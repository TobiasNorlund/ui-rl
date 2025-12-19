from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import random
from typing import NamedTuple
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
import wandb
from train import evaluate


def main(
    rollouts: str,
    grad_accumulation_steps: int = 1, 
    output_dir: str | None = None, 
    eval_checkpoint_steps: int = 100, 
    lora_adapter_path: str | None = None,
    model_name: str = "ByteDance-Seed/UI-TARS-1.5-7B",
):
    with open(rollouts) as f:
        rollout_paths = [path.strip() for path in f]
    
    processor = AutoProcessor.from_pretrained(model_name)
    train_ds = AugmentedSFTDataset(processor, rollout_paths, random_seed=42)
    test_ds = AugmentedSFTDataset(processor, rollout_paths, random_seed=1337)

    train_dataloader = DataLoader(train_ds, batch_size=1, num_workers=2)
    test_dataloader = DataLoader(test_ds, batch_size=1, num_workers=2)

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
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


class Span(NamedTuple):
    start: int
    end: int
    role_id: int


class AugmentedSFTDataset(IterableDataset):
    """
    Assumes a rollout dict where each assistant message's "text" block is not str, 
    but a list[str] of alternative completions.

    This transform selects from the alternatives a random completion, and computes 
    "prompt_token_ids" and "generated_token_ids" for it
    """

    def __init__(self, processor: Qwen2_5_VLProcessor, rollout_paths: list[str], random_seed: int=0):
        self._processor = processor
        self._rng = random.Random(random_seed)
        self._rollouts = []
        for rollout_path in rollout_paths:
            with open(rollout_path) as f:
                self._rollouts.append(json.load(f))
        
    def __iter__(self):
        # 1. Sample a rollout
        rollout = self._rng.choice(self._rollouts)

        # 2. Sample a sequence in that rollout
        seqs = defaultdict(list)  # Map sequence (message indices) to list of completion messages
        for completion in rollout["completions"]:
            longest = max(
                (tuple(c["prompt_messages"] + [c["generated_message"]]) for c in rollout["completions"] if c["prompt_messages"][:len(completion["prompt_messages"])] == completion["prompt_messages"]),
                key=lambda x: len(x)
            )
            seqs[longest].append({
                "message_idx": longest.index(completion["completion"]),
            })
        seq, completion_messages = self._rng.choice(list(seqs.items()))

        # 3. In that sequence, sample each completion
        messages = []
        for i in seq:
            message = deepcopy(rollout["messages"][i])
            if i in completion_messages:
                assert message["content"][0]["type"] == "text" and isinstance(message["content"][0]["text"], list)
                message["content"][0]["text"] = self._rng.choice(message["content"][0]["text"])

        # 4. Run through processor
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # 5. Compute labels
        # Mask labels to only train on completed message tokens
        # Find message spans and match to "completion" messages, i.e the messages we want to train on
        spans = self._find_message_spans(inputs["input_ids"][0].tolist())
        assistant_token_id = 77091
        labels = torch.zeros_like(inputs["input_ids"]).fill_(-100)
        message_prefix_len = 3  # Each message starts with three tokens that we don't wanna train on
        for completion in completion_messages:
            span = spans[completion["message_idx"] + 1] # +1 because jinja adds a system message when rendering
            assert span.role_id == assistant_token_id, "Should only complete assistant messages. Something is wrong"
            
            # Modify the span to exactly match the tokens we want to train on
            start = span.start + message_prefix_len
            end = span.end + 1  # Include the end-of-message token
            # assert len(completion["logprobs"]) == end - start, "Tokenized seq doesn't match reference seq"
            completion["span"] = (start, end)
            labels[0, start:end] = inputs["input_ids"][0, start:end]

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "labels": labels,
        }

    @staticmethod
    def _find_message_spans(input_ids: list) -> list[Span]:
        # Note: UITARS specific
        message_start_id = 151644
        message_end_id = 151645
        # Scan tokens to find message spans
        spans = []
        start, role_id = None, None
        for i, id in enumerate(input_ids):
            if id == message_start_id:
                start = i
                role_id = input_ids[i+1]
                continue
            elif id == message_end_id:
                spans.append(Span(start, i, role_id))
                start, role_id = None, None
        return spans
