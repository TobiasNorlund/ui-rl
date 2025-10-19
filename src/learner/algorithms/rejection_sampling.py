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
        - Input: image + prompt ("What action should I take?")
        - Target: action text (e.g., "left_click(100, 200)")

        The loss is computed using causal language modeling (next token prediction).
        """
        # Convert actions to text labels for the model
        action_texts = self._actions_to_text(batch['actions'])

        # Tokenize action texts to get labels
        # This uses the model's processor to convert text to token IDs
        labels = self._tokenize_actions(model, action_texts, batch['prompts'])

        # Run forward pass with labels (VLMWrapper will compute loss internally)
        model_outputs = model(
            images=batch['images'],  # List[PIL.Image]
            prompts=batch['prompts'],  # List[str]
            return_loss=True,
            labels=labels  # Token IDs for ground truth actions
        )

        # HuggingFace models compute loss automatically when labels are provided
        loss = model_outputs['loss']

        if loss is None:
            logger.error("Model did not return loss. Check VLMWrapper implementation.")
            raise ValueError("Model output does not contain loss")

        return loss

    def _actions_to_text(self, actions: List[Dict[str, Any]]) -> List[str]:
        """
        Convert action dicts to text format for supervised learning.

        Args:
            actions: List of action dicts (e.g., {"action_type": "left_click", "x": 100, "y": 200})

        Returns:
            List of action text strings (e.g., "left_click(100, 200)")
        """
        action_texts = []

        for action in actions:
            action_type = action.get('action_type', 'unknown')

            # Format based on action type
            if action_type == 'left_click':
                x = action.get('x', 0)
                y = action.get('y', 0)
                text = f"left_click({x}, {y})"

            elif action_type == 'right_click':
                x = action.get('x', 0)
                y = action.get('y', 0)
                text = f"right_click({x}, {y})"

            elif action_type == 'double_click':
                x = action.get('x', 0)
                y = action.get('y', 0)
                text = f"double_click({x}, {y})"

            elif action_type == 'type':
                typed_text = action.get('text', '')
                text = f"type(\"{typed_text}\")"

            elif action_type == 'key':
                key = action.get('key', '')
                text = f"key({key})"

            elif action_type == 'screenshot':
                text = "screenshot()"

            elif action_type == 'done':
                text = "done()"

            else:
                # Generic format for unknown actions
                text = f"{action_type}()"
                logger.warning(f"Unknown action type: {action_type}")

            action_texts.append(text)

        return action_texts

    def _tokenize_actions(
        self,
        model: nn.Module,
        action_texts: List[str],
        prompts: List[str]
    ) -> torch.Tensor:
        """
        Tokenize action texts to create labels for language modeling.

        Args:
            model: VLMWrapper model (has processor)
            action_texts: List of action text strings
            prompts: List of prompts (for context)

        Returns:
            Token IDs tensor for labels [B*T, seq_len]
        """
        # Get processor from VLMWrapper
        processor = model.processor

        # Create full text sequences: prompt + action
        # This matches how the model will be prompted during inference
        full_texts = [f"{prompt}\n{action}" for prompt, action in zip(prompts, action_texts)]

        # Tokenize
        tokenized = processor.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Extract input_ids as labels
        labels = tokenized['input_ids']

        # Move to same device as model
        labels = labels.to(next(model.parameters()).device)

        return labels
