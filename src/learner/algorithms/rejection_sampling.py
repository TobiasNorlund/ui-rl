"""Rejection sampling algorithm implementation."""

from typing import List, Dict, Any, Optional, Tuple
import logging
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
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

        # Build teacher-forcing sequences so inputs and labels share the same tokens
        full_texts, prompt_prefixes = self._prepare_teacher_forcing_sequences(
            prompts,
            generated_texts
        )

        # Tokenize generated texts to create labels for supervised learning
        labels = self._tokenize_texts(model, full_texts, prompt_prefixes)

        # Run forward pass with labels (VLMWrapper will compute loss internally)
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

    def _prepare_teacher_forcing_sequences(
        self,
        prompts: List[str],
        generated_texts: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Combine prompt and generated text for teacher forcing while keeping track
        of the prompt portion for label masking.
        """
        full_texts = []
        prompt_prefixes = []

        for prompt, generated_text in zip(prompts, generated_texts):
            # Ensure there is a separator between prompt and generated text
            if prompt.endswith("\n"):
                prompt_prefix = prompt
            else:
                prompt_prefix = f"{prompt}\n"

            full_texts.append(f"{prompt_prefix}{generated_text}")
            prompt_prefixes.append(prompt_prefix)

        return full_texts, prompt_prefixes

    def _tokenize_texts(
        self,
        model: nn.Module,
        full_texts: List[str],
        prompt_prefixes: List[str]
    ) -> torch.Tensor:
        """
        Tokenize generated texts to create labels for language modeling.

        Uses label masking to compute loss only on the generated_text tokens,
        not on the prompt tokens. Prompt tokens are set to -100 (ignored by loss).

        Args:
            model: VLMWrapper model (has processor)
            full_texts: List of combined prompt + generated text strings
            prompt_prefixes: Prompt strings (with separator) for masking

        Returns:
            Token IDs tensor for labels [B*T, seq_len] with prompt tokens masked as -100
        """
        # Get processor from VLMWrapper
        processor = model.processor
        tokenizer = processor.tokenizer

        # We need to create labels where:
        # - Prompt tokens are masked (-100, ignored in loss)
        # - Generated text tokens are kept (used for loss computation)

        batch_labels = []

        for full_text, prompt_prefix in zip(full_texts, prompt_prefixes):
            # Tokenize prompt separately (with separator) to know how many tokens to mask
            prompt_tokens = tokenizer(
                prompt_prefix,
                return_tensors="pt",
                add_special_tokens=True  # Include BOS token if needed
            )['input_ids'][0]  # [prompt_len]

            # Tokenize full sequence (prompt + generated_text)
            full_tokens = tokenizer(
                full_text,
                return_tensors="pt",
                add_special_tokens=True,
                max_length=512,
                truncation=True
            )['input_ids'][0]  # [full_len]

            # Create labels: mask prompt tokens with -100
            labels = full_tokens.clone()
            prompt_len = len(prompt_tokens)
            labels[:prompt_len] = -100  # Mask prompt tokens (ignored in loss)

            batch_labels.append(labels)

        # Pad labels to same length
        padded_labels = pad_sequence(
            batch_labels,
            batch_first=True,
            padding_value=-100  # Padding is also ignored in loss
        )

        # Move to same device as model
        padded_labels = padded_labels.to(next(model.parameters()).device)

        return padded_labels
