from collections import defaultdict
from copy import deepcopy
import random
from typing import Callable, NamedTuple
import json
import torch
from PIL import Image
import io
import base64
from torch.utils.data import Dataset, IterableDataset
from transformers import Qwen2_5_VLProcessor


class UITARS15_RolloutDataset(Dataset):
    """
    Dataset for training UITARS 1.5 on trajectory/rollout data
    """

    class Span(NamedTuple):
        start: int
        end: int

    class TokenSequence(NamedTuple):
        token_ids: torch.LongTensor
        completions: list["UITARS15_RolloutDataset.Span"]
        base64_images: list[str]

    class Rollout(NamedTuple):
        task_spec: dict
        progress: dict
        sequences: list["UITARS15_RolloutDataset.TokenSequence"]

    def __init__(self, processor: Qwen2_5_VLProcessor, rollout_path: str, reward_fn: Callable[[Rollout], float] | None = None):
        self._processor = processor
        
        rollout = self._load_rollout(rollout_path)
        self._sequences = rollout.sequences
        self._task_spec = rollout.task_spec
        self._reward = reward_fn(rollout) if reward_fn is not None else 0.0
            
    def __len__(self):
        return len(self._sequences)

    @property
    def task_spec(self) -> dict:
        return self._task_spec

    def __getitem__(self, idx: int):
        seq = self._sequences[idx]

        input_ids = seq.token_ids
        attention_mask = torch.ones_like(input_ids)

        # Decode base64 images to PIL and use processor to obtain image inputs
        images = [
            Image.open(io.BytesIO(base64.b64decode(img_b64[img_b64.index(","):])))
            for img_b64 in seq.base64_images
        ]
        image_inputs = self._processor.image_processor(images=images, return_tensors="pt")

        # Construct labels to only train on completed message tokens
        labels = torch.zeros_like(input_ids).fill_(-100)
        for completion in seq.completions:
            labels[completion.start:completion.end] = input_ids[completion.start:completion.end]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "reward": torch.tensor(self._reward, dtype=torch.float32),
            **image_inputs
        }

    def _load_rollout(self, rollout_path: str) -> "UITARS15_RolloutDataset.Rollout":
        with open(rollout_path) as f:
            rollout = json.load(f)

        sequences: list[self.TokenSequence] = []
        for completion in rollout["completions"]:
            token_ids = torch.LongTensor(completion["prompt_token_ids"] + completion["generated_token_ids"])
            base64_images = [
                message_block["image_url"]["url"]
                for message_idx in completion["prompt_messages"]
                for message_block in rollout["messages"][message_idx]["content"]
                if type(message_block) == dict and message_block["type"] == "image_url"
            ]
            sequences.append(self.TokenSequence(
                token_ids=token_ids,
                completions=[self.Span(
                    start=len(completion["prompt_token_ids"]), 
                    end=len(completion["prompt_token_ids"])+len(completion["generated_token_ids"])
                )],
                base64_images=base64_images
            ))
        
        for seq in sequences:
            # Find the longest sequence that starts with seq's token_ids
            longest_seq = max(
                (s for s in sequences if len(s.token_ids) >= len(seq.token_ids) and (s.token_ids[:len(seq.token_ids)] == seq.token_ids).all()),
                key=lambda s: len(s.token_ids)
            )
            # Move seq's completion to the longest seq
            completion = seq.completions.pop(0)
            longest_seq.completions.append(completion)

        return self.Rollout(
            task_spec=rollout["task"],
            progress=rollout["progress"],
            sequences=[seq for seq in sequences if len(seq.completions) > 0]
        )


class UITARS15_ThoughtAugmentedRolloutDataset(IterableDataset):
    """
    Assumes a rollout dict where each assistant message's "text" block is not str, 
    but a list[str] of alternative completions.

    This transform selects from the alternatives a random completion, and computes 
    "prompt_token_ids" and "generated_token_ids" for it
    """

    ASSISTANT_TOKEN_ID = 77091

    class Span(NamedTuple):
        start: int
        end: int
        role_id: int

    def __init__(self, processor: Qwen2_5_VLProcessor, rollout_path: str, random_seed: int = 0):
        self._processor = processor
        self._random_seed = random_seed
        with open(rollout_path) as f:
            self._rollout = json.load(f)
        
        self._sequences = defaultdict(list)  # Map sequence (message indices) to list of completion messages
        for completion in self._rollout["completions"]:
            longest = max(
                (tuple(c["prompt_messages"] + [c["generated_message"]]) for c in self._rollout["completions"] if c["prompt_messages"][:len(completion["prompt_messages"])] == completion["prompt_messages"]),
                key=lambda x: len(x)
            )
            self._sequences[longest].append(longest.index(completion["generated_message"]))
        
    @property
    def task_spec(self) -> dict:
        return self._rollout["task"]

    def __iter__(self):
        rng = random.Random(self._random_seed)
        while True:
            # Sample a sequence
            seq, completion_messages = rng.choice(list(self._sequences.items()))

            # In that sequence, sample each completion
            messages = []
            for message_index in seq:
                message = deepcopy(self._rollout["messages"][message_index])
                for block in message["content"]:
                    # Convert "image_url" -> "image"
                    if block["type"] == "image_url":
                        block["type"] = "image"
                        block["image"] = block["image_url"]["url"]
                        del block["image_url"]
                    # Sample completion
                    if block["type"] == "text" and isinstance(block["text"], list):
                        block["text"] = rng.choice(block["text"])
                messages.append(message)

            # Run through processor
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Compute labels
            #  Train only on completed message tokens
            #  Find message spans and match to "completion" messages, i.e the messages we want to train on
            spans = self._find_message_spans(inputs["input_ids"][0].tolist())
            assistant_token_id = 77091
            labels = torch.zeros_like(inputs["input_ids"]).fill_(-100)
            message_prefix_len = 3  # Each message starts with three tokens that we don't wanna train on
            for completion in completion_messages:
                span = spans[completion + 1] # +1 because jinja adds a system message when rendering
                assert span.role_id == assistant_token_id, "Should only complete assistant messages. Something is wrong"
                
                # Modify the span to exactly match the tokens we want to train on
                start = span.start + message_prefix_len
                end = span.end + 1  # Include the end-of-message token
                labels[0, start:end] = inputs["input_ids"][0, start:end]

            yield {
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "pixel_values": inputs["pixel_values"],
                "image_grid_thw": inputs["image_grid_thw"],
                "labels": labels[0],
            }

    @classmethod
    def _find_message_spans(cls, input_ids: list) -> list[Span]:
        # Note: Specific to Qwen 2.5 VL
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
                spans.append(cls.Span(start, i, role_id))
                start, role_id = None, None
        return spans


if __name__ == "__main__":
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("ByteDance-Seed/UI-TARS-1.5-7B")
    ds = UITARS15_ThoughtAugmentedRolloutDataset(processor, rollout_path="data/rollouts/20251219_augmented/row_002_augmented.json")