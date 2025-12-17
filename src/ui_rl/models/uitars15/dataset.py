from collections import defaultdict
from typing import NamedTuple
import json
import torch
from PIL import Image
import io
import base64
from torch.utils.data import Dataset


class Span(NamedTuple):
    start: int
    end: int


class TokenSequence(NamedTuple):
    token_ids: torch.IntTensor
    completions: list[Span]
    base64_images: list[str]


class Rollout(NamedTuple):
    task_spec: dict
    progress: dict
    sequences: list[TokenSequence]


class UITARS15_SFTDataset(Dataset):
    """
    Dataset for SFT training UITARS 1.5 on trajectory/rollout data
    """

    def __init__(self, processor, rollout_paths: list[str]):
        self._processor = processor
        self._sequences: list[TokenSequence] = []
        self.seqidx2rollout: dict[int, Rollout] = {}  # Mapping from seq idx back to its rollout

        for rollout_path in rollout_paths:
            rollout = _load_rollout(rollout_path)
            self._sequences += rollout.sequences
            for seqidx in range(len(self._sequences)-len(rollout.sequences), len(self._sequences)):
                self.seqidx2rollout[seqidx] = rollout
            
    def __len__(self):
        return len(self._sequences)

    def __getitem__(self, idx: int):
        seq = self._sequences[idx]

        input_ids = seq.token_ids[None, ...]
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
            labels[0, completion.start:completion.end] = input_ids[0, completion.start:completion.end]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            **image_inputs
        }


def _load_rollout(rollout_path: str):
    with open(rollout_path) as f:
        rollout = json.load(f)

    sequences: list[TokenSequence] = []
    for completion in rollout["completions"]:
        token_ids = torch.IntTensor(completion["prompt_token_ids"] + completion["generated_token_ids"])
        base64_images = [
            message_block["image_url"]["url"]
            for message_idx in completion["prompt_messages"]
            for message_block in rollout["messages"][message_idx]["content"]
            if type(message_block) == dict and message_block["type"] == "image_url"
        ]
        sequences.append(TokenSequence(
            token_ids=token_ids,
            completions=[Span(
                start=len(completion["prompt_token_ids"]), 
                end=len(completion["prompt_token_ids"])+len(completion["generated_token_ids"])
            )],
            base64_images=base64_images
        ))
    
    for seq in sequences:
        # Find the longest sequence that starts with seq.token_ids
        longest_seq = max(
            (s for s in sequences if len(s.token_ids) >= len(seq.token_ids) and (s.token_ids[:len(seq.token_ids)] == seq.token_ids).all()),
            key=lambda s: len(s.token_ids)
        )
        # Move seq's completion to the longest seq
        completion = seq.completions.pop(0)
        longest_seq.completions.append(completion)

    return Rollout(
        task_spec=rollout["task"],
        progress=rollout["progress"],
        sequences=[seq for seq in sequences if len(seq.completions) > 0]
    )
