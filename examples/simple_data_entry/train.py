from pathlib import Path
import torch
from torch.utils.data import DataLoader, Sampler, random_split
from accelerate import Accelerator
from typing import Any, List, Optional
from collections import defaultdict, namedtuple
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
from scipy.special import softmax
import wandb
import pickle
import random
import os
from datetime import datetime

from ui_rl.models.uitars15 import UITARS15_SFTDataset


def main(
    rollouts: str,
    num_test_rollouts: int,
    grad_accumulation_steps: int = 1, 
    output_dir: Optional[str] = None, 
    eval_checkpoint_steps: int = 100, 
    lora_adapter_path: Optional[str] = None,
    model_name: str = "ByteDance-Seed/UI-TARS-1.5-7B",
    task_error_rates: dict | None = None,
    sampler_temperature: float = 1.0,
):
    with open(rollouts) as f:
        rollout_paths = [path.strip() for path in f]

    train_rollouts, test_rollouts = train_test_split(rollout_paths, test_num_or_ratio=num_test_rollouts)
    print(f"Using {len(train_rollouts)} training and {len(test_rollouts)} test rollouts")
    
    processor = AutoProcessor.from_pretrained(model_name)
    train_ds = UITARS15_SFTDataset(processor, train_rollouts)
    test_ds = UITARS15_SFTDataset(processor, test_rollouts)

    sampler = ErrorBasedTaskUpsampler(
        idx2task={i: r.task_spec["rows"][0] for i, r in train_ds.seqidx2rollout.items()},
        task_error_rates=task_error_rates, 
        temperature=sampler_temperature
    ) if task_error_rates is not None else None

    train_dataloader = DataLoader(train_ds, batch_size=1, collate_fn=lambda x: x[0], num_workers=1, sampler=sampler)
    test_dataloader = DataLoader(test_ds, batch_size=1, collate_fn=lambda x: x[0], num_workers=1)

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if lora_adapter_path is not None:
        # Continue training from an existing LoRA adapter
        print(f"Loading LoRA adapter from: {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path, is_trainable=True)
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
    accelerator = Accelerator(gradient_accumulation_steps=grad_accumulation_steps)
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)

    # Resolve default checkpoint directory (datetime-stamped) and ensure it exists
    if output_dir is None:
        repo_root = Path(__file__).parent.parent.parent
        output_root = repo_root / "data" / "checkpoints" / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_root = output_dir
    if accelerator.is_main_process:
        os.makedirs(output_root, exist_ok=True)
    accelerator.wait_for_everyone()

    # Initialize Weights & Biases
    if accelerator.is_main_process:
        run = wandb.init(project="ui-rl", config={
            "lr": lr,
            "batch_size": 1,
            "grad_accum": grad_accumulation_steps,
            "train_size": len(train_rollouts),
            "test_size": len(test_rollouts),
            "output_dir": output_root,
            "eval_checkpoint_steps": eval_checkpoint_steps,
        })

        # Save list of rollout paths
        rollout_path_file = os.path.join(run.dir, "rollouts.txt")
        with open(rollout_path_file, "w") as f:
            f.write("\n".join(rollout_paths))
        wandb.save(rollout_path_file)

    accelerator.wait_for_everyone()

    global_step = 0
    for epoch in range(10):
        progress = tqdm(train_dataloader, disable=not accelerator.is_main_process)
        for batch in progress:
            # Evaluate and checkpoint every n steps
            if global_step % eval_checkpoint_steps == 0:
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


def train_test_split(data: list, test_num_or_ratio: int | float):
    """
    Split data into train and test sets.

    Args:
        data: List of data items to split
        test_num_or_ratio: If int, the number of test samples. If float, the ratio of test samples (0.0 to 1.0)

    Returns:
        tuple: (train_data, test_data)
    """
    total_size = len(data)

    if isinstance(test_num_or_ratio, float):
        if not 0.0 <= test_num_or_ratio <= 1.0:
            raise ValueError(f"test_num_or_ratio as ratio must be between 0.0 and 1.0, got {test_num_or_ratio}")
        test_size = int(total_size * test_num_or_ratio)
    else:
        test_size = test_num_or_ratio
        if test_size < 0 or test_size > total_size:
            raise ValueError(f"test_num_or_ratio as count must be between 0 and {total_size}, got {test_size}")

    train_size = total_size - test_size

    # Shuffle data with fixed seed for reproducibility
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    train_data = shuffled_data[:train_size]
    test_data = shuffled_data[train_size:]

    return train_data, test_data


class ErrorBasedTaskUpsampler(Sampler):
    def __init__(self, idx2task: dict[int, Any], task_error_rates: dict[Any, float], temperature: float = 1.0, error_margin: float = 0.1):
        self._task_error_rates = task_error_rates
        self._temperature = temperature
        self._error_margin = error_margin
        self._task_indices = defaultdict(list)
        for i, task in idx2task.items():
            self._task_indices[task].append(i)

    def __iter__(self):
        all_tasks = list(self._task_indices.keys())
        weights = softmax([max(self._task_error_rates[task], self._error_margin) / self._temperature for task in all_tasks])
        while True:
            # Sample row 
            row = random.choices(population=all_tasks, weights=weights, k=1)[0]
            # Sample index for that row uniformly
            yield random.choice(self._task_indices[row])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--num-test-rollouts", type=int)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--eval-checkpoint-steps", type=int, default=100)
    parser.add_argument("--lora-adapter-path", type=str, default=None, help="Path to existing LoRA adapter to continue training from")
    parser.add_argument("--model-name", type=str, default="ByteDance-Seed/UI-TARS-1.5-7B")
    parser.add_argument("--task-error-rates")
    parser.add_argument("--sampler-temperature", type=float, default=1.0)
    args = parser.parse_args()

    if args.task_error_rates:
        # Unpickle
        with open(args.task_error_rates, "rb") as f:
            args.task_error_rates = pickle.load(f)

    random.seed(42)

    main(**vars(args))