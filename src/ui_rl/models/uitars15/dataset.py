from collections import defaultdict, namedtuple
from copy import deepcopy
import json
import torch
from torch.utils.data import Dataset


Span = namedtuple("Span", ["start", "end", "role_id"])
Rollout = namedtuple("Rollout", ["task", "progress"])

class UITARS15_SFTDataset(Dataset):
    """
    Dataset for SFT training UITARS 1.5 on trajectory/rollout data
    """

    def __init__(self, processor, rollout_paths: list[str]):
        self._processor = processor
        self._rollouts = []
        self._sequences = []  # list of (messages, completions)

        self.seqidx2rollout = {}  # Mapping from seq idx back to its rollout

        for rollout_path in rollout_paths:
            rollout = _load_rollout(rollout_path)
            rollout_meta = Rollout(
                task=rollout["task"], 
                progress=rollout["progress"]
            )
            rollout_seqs = _get_rollout_sequences(rollout)
            self._sequences += rollout_seqs
            self._rollouts.append(rollout_meta)
            for seqidx in range(len(self._sequences)-len(rollout_seqs), len(self._sequences)):
                self.seqidx2rollout[seqidx] = rollout_meta
            
    def __len__(self):
        return len(self._sequences)

    def __getitem__(self, idx):
        messages, completions = self._sequences[idx]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Mask labels to only train on completed message tokens
        # Find message spans and match to "completion" messages, i.e the messages we want to train on
        spans = _find_message_spans(inputs["input_ids"][0].tolist())
        assistant_token_id = 77091
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


def _load_rollout(rollout_path: str):
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
        if "logprobs" in completion:
            del completion["logprobs"]
        #for logprobs in completion["logprobs"]:
        #    del logprobs["bytes"]
        #    del logprobs["top_logprobs"]
    return rollout


def _get_rollout_sequences(rollout):
    seqs = defaultdict(list)
    for completion in rollout["completions"]:
        longest = max(
            (tuple(c["context"] + [c["completion"]]) for c in rollout["completions"] if c["context"][:len(completion["context"])] == completion["context"]),
            key=lambda x: len(x)
        )
        seqs[longest].append({
            "message_idx": longest.index(completion["completion"]),
            #"logprobs": [l["logprob"] for l in completion["logprobs"]],
            #"tokens": [l["token"] for l in completion["logprobs"]],
        })

    output = []
    for seq, completions_metadata in seqs.items():
        messages = [rollout["messages"][i] for i in seq]
        output.append((messages, completions_metadata))
    return output


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