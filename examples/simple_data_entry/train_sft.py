from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset
from accelerate import Accelerator, DataLoaderConfiguration
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import wandb
from pydantic import BaseModel
import random
import os
from datetime import datetime

from ui_rl.models.uitars15.dataset import UITARS15_RolloutDataset, UITARS15_ThoughtAugmentedRolloutDataset
from utils import Qwen2_5_VLCollate


class RolloutGroup(BaseModel):
    name: str
    sampling_weight: float = 1.0
    rollouts: list[str]


class TrainConfig(BaseModel):
    model_name: str
    learning_rate: float
    train_rollouts: list[RolloutGroup]
    test_rollouts: list[str]
    accelerate_kwargs: dict = {}
    output_dir: str | None = None
    eval_checkpoint_steps: int | None = None
    lora_adapter_path: str | None = None


def main(config_file: str):
    # Disable as we use multi-worker dataloader 
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with open(config_file) as f:
        config = TrainConfig(**yaml.safe_load(f))

    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
        # Need dispatch_batches=False as "pixel_values" and "image_grid_thw" lacks batch dim
        **config.accelerate_kwargs
    )
    
    processor = AutoProcessor.from_pretrained(config.model_name)
    train_ds = SetGlobalUniqueRandomSeedWrapper(
        StratifiedRolloutDataset(
            rollout_groups=config.train_rollouts,
            processor=processor,
            random_seed=42
        )
    )
    test_ds = torch.utils.data.ConcatDataset(
        [UITARS15_RolloutDataset(processor, path) for path in config.test_rollouts]
    )

    collator = Qwen2_5_VLCollate(processor)
    train_dataloader = DataLoader(train_ds, batch_size=1, collate_fn=collator, num_workers=1)
    test_dataloader = DataLoader(test_ds, batch_size=1, collate_fn=collator, num_workers=1)

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if config.lora_adapter_path is not None:
        # Continue training from an existing LoRA adapter
        print(f"Loading LoRA adapter from: {config.lora_adapter_path}")
        model = PeftModel.from_pretrained(model, config.lora_adapter_path, is_trainable=True)
    else:
        # Start fresh with new LoRA adapter
        lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)

    # Resolve default checkpoint directory (datetime-stamped) and ensure it exists
    if config.output_dir is None:
        repo_root = Path(__file__).parent.parent.parent
        output_root = repo_root / "data" / "checkpoints" / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_root = config.output_dir
    if accelerator.is_main_process:
        os.makedirs(output_root, exist_ok=True)
    accelerator.wait_for_everyone()

    # Initialize Weights & Biases
    if accelerator.is_main_process:
        wandb.init(project="ui-rl", config={"config": config_file})
        wandb.save(config_file)

    accelerator.wait_for_everyone()

    global_step = 0
    for epoch in range(10):
        progress = tqdm(train_dataloader, disable=not accelerator.is_main_process)
        for batch in progress:
            # Evaluate and checkpoint every n steps
            if global_step % config.eval_checkpoint_steps == 0:
                # Save LoRA adapter weights
                checkpoint_dir = os.path.join(output_root, f"step_{global_step}")
                peft_model = accelerator.unwrap_model(model)
                peft_model.save_pretrained(checkpoint_dir, safe_serialization=True)
                if accelerator.is_main_process:
                    wandb.log({"adapter_checkpoint": checkpoint_dir, "epoch": epoch, "step": global_step})
                
                # Run eval
                test_loss = evaluate(model, test_dataloader, accelerator)
                if accelerator.is_main_process:
                    wandb.log({"test_loss": test_loss, "epoch": epoch, "step": global_step})
                    print(f"Step {global_step} - Test loss: {test_loss}")

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                if accelerator.is_main_process:
                    wandb.log({"train_loss": loss.item(), "epoch": epoch, "step": global_step})
                progress.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

    if accelerator.is_main_process:
        wandb.finish()


def evaluate(model, dataloader, accelerator):
    model_was_training = model.training
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        progress = tqdm(dataloader, disable=not accelerator.is_main_process)
        for batch in progress:
            outputs = model(**batch)
            loss = outputs.loss
            gathered = accelerator.gather(loss.detach())
            total_loss += gathered.mean().item()
            num_batches += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
    avg_loss = total_loss / max(num_batches, 1)
    if model_was_training:
        model.train()
    return avg_loss


class StratifiedRolloutDataset(IterableDataset):
    def __init__(
        self, 
        rollout_groups: list[RolloutGroup],
        processor: Qwen2_5_VLProcessor,
        random_seed: int = 0
    ):
        self._group_datasets = defaultdict(list)  # list of datasets
        self._sampling_weights = {group.name: group.sampling_weight for group in rollout_groups}
        self._random_seed = random_seed

        for group in rollout_groups:
            for rollout_path in group.rollouts:
                if rollout_path.endswith("augmented.json"):
                    ds = UITARS15_ThoughtAugmentedRolloutDataset(processor, rollout_path, random_seed=random_seed)
                else:
                    ds = UITARS15_RolloutDataset(processor, rollout_path)
                self._group_datasets[group.name].append(ds)

    def __iter__(self):
        rng = random.Random(self._random_seed)

        # Create iterator for each dataset
        def get_ds_iterator(ds):
            if isinstance(ds, UITARS15_RolloutDataset):
                return self._random_sampler(rng, ds)
            elif isinstance(ds, UITARS15_ThoughtAugmentedRolloutDataset):
                return iter(ds)
            else:
                raise ValueError(f"Invalid dataset: {ds}")

        grouped_dataset_iterators = {
            group: [get_ds_iterator(ds) for ds in ds_lst]
            for group, ds_lst in self._group_datasets.items()
        }

        all_groups = list(grouped_dataset_iterators.keys())
        weights = [self._sampling_weights[group] for group in all_groups]
        while True:
            # Sample a group according to normalized_group_weights
            group = rng.choices(population=all_groups, weights=weights, k=1)[0]
            # Sample uniformly an iterator from that group, and take the next in that group
            yield next(rng.choice(grouped_dataset_iterators[group]))

    @staticmethod
    def _random_sampler(rng: random.Random, dataset: UITARS15_RolloutDataset):
        while True:
            yield rng.choice(dataset)


class SetGlobalUniqueRandomSeedWrapper(IterableDataset):
    """
    Before __iter__, this wrapper updates iterable._random_seed to make it globally unique wrt:
     - torch.distributed.get_rank()
     - torch.utils.data.get_worker_info() 
    """
    def __init__(self, iterable):
        self._iterable = iterable

    def __iter__(self):
        if hasattr(self._iterable, "_random_seed"):
            # 1. Get distributed rank
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0

            # 2. Get DataLoader worker info
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                worker_id = 0
            else:
                worker_id = worker_info.id
            self._iterable._random_seed += 100_000 * rank + 1_000 * worker_id
        
        return iter(self._iterable)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)