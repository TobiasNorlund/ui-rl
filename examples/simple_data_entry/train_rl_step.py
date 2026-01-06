import torch
import yaml
from torch.utils.data import DataLoader
from accelerate import Accelerator, DataLoaderConfiguration
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
from pydantic import BaseModel
import os
from ui_rl.models.uitars15.dataset import UITARS15_RolloutDataset
from utils import Qwen2_5_VLCollate


class TrainConfig(BaseModel):
    model_name: str
    learning_rate: float
    train_rollouts: list[str]
    output_dir: str
    lora_adapter: str | None = None


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
    if config.lora_adapter is not None:
        # Continue training from an existing LoRA adapter
        print(f"Loading LoRA adapter from: {config.lora_adapter}")
        model = PeftModel.from_pretrained(model, config.lora_adapter, is_trainable=True)
    else:
        # Start fresh with new LoRA adapter
        lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    # Load optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer_path = os.path.join(config.lora_adapter, "optimizer.pt")
    if config.lora_adapter is not None and os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))

    accelerator = Accelerator(
        # Need dispatch_batches=False as "pixel_values" and "image_grid_thw" lacks batch dim
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False, even_batches=True),
    )
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        
    # Note: We only take a single gradient step over the whole dataset!
    accelerator.gradient_accumulation_steps = len(train_dataloader)

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    progress = tqdm(train_dataloader, disable=not accelerator.is_main_process)
    for batch in progress:
        with accelerator.accumulate(model):
            outputs = model(**batch)
            # Note: Assumes batch_size == 1
            loss = outputs.loss * batch["reward"][0]
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

    peft_model: PeftModel = accelerator.unwrap_model(model)
    peft_model.save_pretrained(config.output_dir, safe_serialization=True)
    torch.save(optimizer.state_dict(), os.path.join(config.output_dir, "optimizer.pt"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args.config)