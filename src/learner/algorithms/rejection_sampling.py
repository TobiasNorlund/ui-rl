"""Rejection sampling algorithm implementation."""

from typing import List, Dict, Any, Optional
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
        labels = self._tokenize_conversations(model, conversations)

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
        conversations: List[List[Dict[str, Any]]]
    ) -> torch.Tensor:
        """
        Tokenize chat conversations and build language-model labels that supervise
        only the assistant portion of each message.
        """
        processor = model.processor
        tokenized = processor.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"]
        assistant_masks = tokenized["assistant_masks"]

        labels = input_ids.clone()
        labels[assistant_masks == 0] = -100

        return labels.to(next(model.parameters()).device)
