from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader
from accelerate import Accelerator, DataLoaderConfiguration
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import wandb
from pydantic import BaseModel
import os
from datetime import datetime

from ui_rl.models.uitars15 import UITARS15_RolloutDataset
from utils import Qwen2_5_VLCollate


class TrainConfig(BaseModel):
    model_name: str
    learning_rate: float
    train_rollouts: list[str]
    accelerate_kwargs: dict = {}
    output_dir: str | None = None
    eval_checkpoint_steps: int | None = None
    lora_adapter_path: str | None = None


def reward_fn(rollout: UITARS15_RolloutDataset.Rollout):
    if set(rollout.progress["submitted_row_indices"]) == set([r-2 for r in rollout.task_spec["rows"]]):
        return 1.0
    else:
        return -1.0


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
    
    print("Loading data...")
    processor = AutoProcessor.from_pretrained(config.model_name)
    train_ds = torch.utils.data.ConcatDataset(
        [UITARS15_RolloutDataset(processor, path, reward_fn=reward_fn) for path in tqdm(config.train_rollouts)]
    )

    collator = Qwen2_5_VLCollate(processor)
    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collator, num_workers=2)

    # Load model
    print("Loading model...")
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
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

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
            if global_step % config.eval_checkpoint_steps == 0 and global_step > 0:
                # Save LoRA adapter weights 
                checkpoint_dir = os.path.join(output_root, f"step_{global_step}")
                peft_model = accelerator.unwrap_model(model)
                peft_model.save_pretrained(checkpoint_dir, safe_serialization=True)
                if accelerator.is_main_process:
                    wandb.log({"adapter_checkpoint": checkpoint_dir, "epoch": epoch, "step": global_step})
                
                # Run eval
                # test_loss = evaluate(model, test_dataloader, accelerator)
                # if accelerator.is_main_process:
                #     wandb.log({"test_loss": test_loss, "epoch": epoch, "step": global_step})
                #     print(f"Step {global_step} - Test loss: {test_loss}")

            with accelerator.accumulate(model):
                outputs = model(**batch)
                # Note: Assumes batch_size == 1
                loss = outputs.loss * batch["reward"][0]
                if accelerator.is_main_process:
                    wandb.log({"train_loss": loss.item(), "epoch": epoch, "step": global_step})
                progress.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if accelerator.is_main_process:
                        wandb.log({"grad_norm": grad_norm.item(), "step": global_step})
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)