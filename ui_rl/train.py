import torch
from torch.utils.data import Dataset, DataLoader, random_split
import json
from accelerate import Accelerator
from typing import List, Optional
from collections import defaultdict, namedtuple
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import transformers.loss.loss_utils
from tqdm import tqdm
import wandb
from copy import deepcopy
import os
from datetime import datetime


def main(rollouts: List[str], grad_accumulation_steps: int = 1, output_dir: Optional[str] = None, eval_checkpoint_steps: int = 100):
    # Monkey patch transformers fixed_cross_entropy to allow passing "reduction"
    transformers.loss.loss_utils.fixed_cross_entropy = fixed_cross_entropy

    ds = load_dataset(rollouts)
    train_size = int(0.95 * len(ds))
    test_size = len(ds) - train_size
    train_dataset, test_dataset = random_split(
        ds, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0], num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0], num_workers=1)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "ByteDance-Seed/UI-TARS-1.5-7B", 
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.train()
    lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    accelerator = Accelerator(gradient_accumulation_steps=grad_accumulation_steps)
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)

    # Resolve default checkpoint directory (datetime-stamped) and ensure it exists
    if output_dir is None:
        default_root = "checkpoints"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = os.path.join(default_root, timestamp)
    else:
        output_root = output_dir
    if accelerator.is_main_process:
        os.makedirs(output_root, exist_ok=True)
    accelerator.wait_for_everyone()

    # Initialize Weights & Biases
    if accelerator.is_main_process:
        wandb.init(project="ui-rl", config={
            "lr": lr,
            "batch_size": 1,
            "grad_accum": grad_accumulation_steps,
            "train_size": train_size,
            "test_size": test_size,
            "output_dir": output_root,
            "eval_checkpoint_steps": eval_checkpoint_steps,
        })
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


def load_dataset(rollout_paths: List[str]):
    seqs = []
    for rollout_path in rollout_paths:
        rollout = load_rollout(rollout_path)
        seqs += get_rollout_sequences(rollout)

    return RolloutDataset(seqs)


class RolloutDataset(Dataset):
    def __init__(self, sequences):
        self.processor = AutoProcessor.from_pretrained("ByteDance-Seed/UI-TARS-1.5-7B")
        self.sequences = sequences  # list of (messages, completions)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        messages, completions = self.sequences[idx]

        # Remove all keys with value None, for jinja not to render empty images
        remove_none_values(messages)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Mask labels to only train on completed messages
        # Find message spans and match to "completion" messages, i.e the messages we want to train on
        spans = find_message_spans(inputs["input_ids"][0].tolist())
        assistant_token_id = 77091  # self.processor.tokenizer.convert_tokens_to_ids(["assistant"])
        labels = torch.zeros_like(inputs["input_ids"]).fill_(-100)
        message_prefix_len = 3  # Each message starts with three tokens that we don't wanna train on
        for completion in completions:
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
            "completions": completions,
        }


def load_rollout(rollout_path: str):
    with open(rollout_path) as f:
        rollout = json.load(f)
    # Preprocess messages
    for message in rollout["messages"]:
        # Convert "content" to list if str
        if type(message["content"]) == str:
            message["content"] = [{"type": "text", "text": message["content"]}]
        # Remove all keys except "content" or "role"
        keys_to_remove = [k for k in list(message.keys()) if k not in ("role", "content")]
        for k in keys_to_remove:
            del message[k]
        # Convert "image_url" -> "image"
        for block in message["content"]:
            if block["type"] == "image_url":
                block["type"] = "image"
                block["image"] = block["image_url"]["url"]
                del block["image_url"]
    # Preprocess completions
    for completion in rollout["completions"]:
        for logprobs in completion["logprobs"]:
            del logprobs["bytes"]
            del logprobs["top_logprobs"]
    return rollout


def get_rollout_sequences(rollout):
    seqs = defaultdict(list)
    for completion in rollout["completions"]:
        longest = max(
            (tuple(c["context"] + [c["completion"]]) for c in rollout["completions"] if c["context"][:len(completion["context"])] == completion["context"]),
            key=lambda x: len(x)
        )
        seqs[longest].append({
            "message_idx": longest.index(completion["completion"]),
            "logprobs": [l["logprob"] for l in completion["logprobs"]],
            "tokens": [l["token"] for l in completion["logprobs"]],
        })

    output = []
    for seq, completions_metadata in seqs.items():
        messages = [deepcopy(rollout["messages"][i]) for i in seq]
        output.append((messages, completions_metadata))
    return output


Span = namedtuple("Span", ["start", "end", "role_id"])

def find_message_spans(input_ids: list) -> List[Span]:
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


def remove_none_values(obj):
    if isinstance(obj, dict):
        keys_to_delete = [k for k, v in obj.items() if v is None]
        for k in keys_to_delete:
            del obj[k]
        for v in obj.values():
            remove_none_values(v)
    elif isinstance(obj, list):
        for item in obj:
            remove_none_values(item)


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    reduction = kwargs.get("reduction", reduction)
    loss = torch.nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", nargs="+", default=["runs/20251103_181518/rollout_000.json"])
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--eval-checkpoint-steps", type=int, default=100)
    args = parser.parse_args()
    main(**vars(args))