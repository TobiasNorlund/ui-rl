"""Rejection sampling algorithm implementation."""

from typing import List, Dict, Any, Optional, Tuple
import logging
import torch
import torch.nn as nn
from .base import Algorithm
from ...data_utils.trajectory import Trajectory

logger = logging.getLogger(__name__)


class RejectionSampling(Algorithm):
    """
    Rejection Sampling: Only train on trajectories with positive total reward.
    Uses supervised learning (behavior cloning) on successful episodes.

    Design: Simplest possible RL algorithm. Great starting point.
    """

    def __init__(self, reward_threshold: float = 0.0):
        """
        Args:
            reward_threshold: Minimum total reward to keep trajectory
        """
        self.reward_threshold = reward_threshold

    def process_trajectories(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """Filter out trajectories with low total reward."""
        filtered = [t for t in trajectories if t.total_reward() > self.reward_threshold]
        logger.info(f"Rejection sampling: kept {len(filtered)}/{len(trajectories)} trajectories")
        return filtered

    def compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        model_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Supervised learning loss on successful trajectories using causal language modeling.

        For VLMs, we treat action prediction as a text generation task:
        - Input: image + prompt + previous generated text (teacher forcing)
        - Target: generated_text from trajectory (e.g., "click(100, 200)")

        The loss is computed using causal language modeling (next token prediction).
        The model learns to reproduce the exact text it generated during successful episodes.
        """
        prompts = batch['prompts']
        generated_texts = batch['generated_texts']
        images = batch['images']

        conversations = self._build_conversations(images, prompts, generated_texts)
        full_texts = self._render_conversations(model, conversations)
        labels = self._tokenize_conversations(model, full_texts)

        model_outputs = model(
            images=batch['images'],  # List[PIL.Image]
            prompts=full_texts,  # Teacher-forcing text (prompt + generated text)
            return_loss=True,
            labels=labels  # Token IDs for ground truth generated texts
        )

        # HuggingFace models compute loss automatically when labels are provided
        loss = model_outputs['loss']

        if loss is None:
            logger.error("Model did not return loss. Check VLMWrapper implementation.")
            raise ValueError("Model output does not contain loss")

        return loss

    def _build_conversations(
        self,
        images: List[Any],
        prompts: List[str],
        generated_texts: List[str]
    ) -> List[List[Dict[str, Any]]]:
        """
        Build chat-style conversations that pair each image with its prompt and generated text.
        """
        conversations: List[List[Dict[str, Any]]] = []

        for image, prompt, generated_text in zip(images, prompts, generated_texts):
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": generated_text},
                    ],
                },
            ]
            conversations.append(conversation)

        return conversations

    def _render_conversations(
        self,
        model: nn.Module,
        conversations: List[List[Dict[str, Any]]]
    ) -> List[str]:
        """
        Render chat conversations into text prompts using the model's chat template.
        """
        processor = model.processor
        rendered = processor.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=False,
        )

        if isinstance(rendered, str):
            return [rendered]
        return rendered

    def _tokenize_conversations(
        self,
        model: nn.Module,
        full_texts: List[str]
    ) -> torch.Tensor:
        """
        Tokenize chat conversations and build language-model labels that supervise
        only the assistant portion of each message.
        """
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise AttributeError("Model must expose a tokenizer for label construction.")

        try:
            encoded = tokenizer(
                full_texts,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                return_offsets_mapping=True,
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                "The configured tokenizer does not support offset_mapping, "
                "which is required to build supervision masks for assistant tokens."
            ) from exc

        input_ids_list = encoded["input_ids"]
        offsets_list = encoded["offset_mapping"]

        batch_size = len(input_ids_list)
        max_length = max(len(ids) for ids in input_ids_list) if input_ids_list else 0
        labels = torch.full(
            (batch_size, max_length),
            fill_value=-100,
            dtype=torch.long,
        )

        for idx, (full_text, input_ids, offsets) in enumerate(
            zip(full_texts, input_ids_list, offsets_list)
        ):
            assistant_spans = self._extract_assistant_spans(full_text)
            if not assistant_spans:
                logger.warning(
                    "No assistant spans found in conversation; supervising entire sequence."
                )
                assistant_spans = [(0, len(full_text))]

            label_row = torch.tensor(input_ids, dtype=torch.long)
            mask = torch.zeros(len(input_ids), dtype=torch.bool)

            for token_index, (start_char, end_char) in enumerate(offsets):
                if end_char <= start_char:
                    continue  # Skip special tokens or padding with zero-length offsets

                for span_start, span_end in assistant_spans:
                    if start_char < span_end and end_char > span_start:
                        mask[token_index] = True
                        break

            label_row[~mask] = -100
            labels[idx, : len(input_ids)] = label_row

        return labels.to(next(model.parameters()).device)

    @staticmethod
    def _extract_assistant_spans(full_text: str) -> List[Tuple[int, int]]:
        """
        Identify the character spans corresponding to assistant messages within a rendered
        conversation. We rely on the chat template markers to locate assistant content.
        """
        spans: List[Tuple[int, int]] = []
        assistant_tag = "<|im_start|>assistant"
        end_tag = "<|im_end|>"
        search_pos = 0

        while True:
            start_idx = full_text.find(assistant_tag, search_pos)
            if start_idx == -1:
                break

            content_start = full_text.find("\n", start_idx)
            if content_start == -1:
                break
            content_start += 1

            content_end = full_text.find(end_tag, content_start)
            if content_end == -1:
                break

            spans.append((content_start, content_end))
            search_pos = content_end + len(end_tag)

        return spans
