import torch
from torch.utils.data import Dataset, DataLoader, random_split
import json
from accelerate import Accelerator
from typing import List, Optional
from collections import defaultdict, namedtuple
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model
import transformers.loss.loss_utils
from tqdm import tqdm
import wandb


def main(rollouts: List[str], grad_accumulation_steps: int = 1):
    # Monkey patch transformers fixed_cross_entropy to allow passing "reduction"
    transformers.loss.loss_utils.fixed_cross_entropy = fixed_cross_entropy

    ds = load_dataset(rollouts)
    train_size = int(0.8 * len(ds))
    test_size = len(ds) - train_size
    train_dataset, test_dataset = random_split(
        ds, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0], num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0], num_workers=1)

    #tokenizer = AutoTokenizer.from_pretrained("ByteDance-Seed/UI-TARS-1.5-7B")
    model = load_model()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    accelerator = Accelerator(gradient_accumulation_steps=grad_accumulation_steps)
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)

    # Initialize Weights & Biases
    if accelerator.is_main_process:
        wandb.init(project="ui-rl", config={
            "lr": 1e-4,
            "batch_size": 1,
            "train_size": train_size,
            "test_size": test_size,
        })
    accelerator.wait_for_everyone()

    for epoch in range(10):
        # Evaluate on test set prior to each epoch
        test_loss = evaluate(model, test_dataloader, accelerator)
        if accelerator.is_main_process:
            wandb.log({"test_loss": test_loss, "epoch": epoch})
            print("Test loss:", test_loss)

        progress = tqdm(train_dataloader, disable=not accelerator.is_main_process)
        for batch in progress:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                if accelerator.is_main_process:
                    wandb.log({"train_loss": loss.item(), "epoch": epoch})
                progress.set_postfix({"loss": f"{loss.item():.4f}"})
                accelerator.backward(loss)
                #loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    if accelerator.is_main_process:
        wandb.finish()

    # for i, inputs in tqdm(enumerate(ds), total=len(ds)):
    #     if i < 380:
    #         continue
    #     if inputs is None:
    #         print(f"Couldn't load example {i}")
    #         continue
    #     batch = {k: v.to("cuda") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    #     output = model(reduction="none", **batch)
    #     loss = output.loss.to("cpu")
    #     for j, completion in enumerate(inputs["completions"]):
    #         start = completion["span"][0] - 1  # Span is shifted by one in loss
    #         end = completion["span"][1] - 1
    #         neg_logprobs = -torch.tensor(completion["logprobs"])
    #         diff = loss[start:end] - neg_logprobs
    #         if diff.mean().abs() > 1e-2 or diff.abs().max() > 1.0:
    #             #from tabulate import tabulate
    #             #tokens = [tokenizer.decode(token_id).__repr__() for token_id in inputs["input_ids"][0, start+1:end+1]]
    #             #data = zip(tokens, loss[start:end], neg_logprobs, diff)
    #             #print(tabulate(data, headers=["Token", "Loss", "Neg logprob (ref)", "Diff"]))
    #             print(i, j, "Mean abs diff", diff.mean().abs().tolist(), "Max abs diff", diff.abs().max().tolist())
    #             #breakpoint()


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


def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "ByteDance-Seed/UI-TARS-1.5-7B", 
        #device_map="cuda", 
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    return model


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
            #assert len(completion["logprobs"]) == end - start, "Tokenized seq doesn't match reference seq"
            if len(completion["logprobs"]) != end - start:
                return None
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
        messages = [rollout["messages"][i] for i in seq]
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
    parser.add_argument("--rollouts", nargs="+", default=["runs/20251102_191524/rollout_004.json"])
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    args = parser.parse_args()
    main(**vars(args))