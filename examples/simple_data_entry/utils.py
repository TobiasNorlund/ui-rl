

import torch
from transformers import Qwen2_5_VLProcessor


class Qwen2_5_VLCollate:
    def __init__(self, processor: Qwen2_5_VLProcessor):
        self.processor = processor

    def __call__(self, instances):
        # Extract individual components
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        attention_mask = [instance["attention_mask"] for instance in instances]
        reward = [instance["reward"] for instance in instances]
        
        # pixel_values is a list of tensors, we need to concatenate them
        # image_grid_thw is a list of tensors, we need to stack them
        pixel_values = [instance["pixel_values"] for instance in instances]
        image_grid_thw = [instance["image_grid_thw"] for instance in instances]

        # 1. Pad text sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        # 2. Flatten and concatenate visual data
        # Since Qwen2.5-VL uses dynamic resolution, we concatenate all pixels
        # from all images in the batch into one long tensor.
        pixel_values = torch.cat(pixel_values, dim=0)
        image_grid_thw = torch.cat(image_grid_thw, dim=0)

        # 3. Stack rewards
        reward = torch.stack(reward) 

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "reward": reward
        }