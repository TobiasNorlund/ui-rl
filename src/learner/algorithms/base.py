"""Base class for RL algorithms."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from ...data_utils.trajectory import Trajectory


class Algorithm(ABC):
    """
    Abstract base class for RL algorithms.

    Design: Composition pattern - algorithms are pluggable components.
    This makes it easy to swap algorithms without changing Trainer code.

    Each algorithm is responsible for:
    1. Processing/filtering trajectories before training
    2. Computing the loss function for training
    """

    @abstractmethod
    def process_trajectories(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """
        Process trajectories before training (filter, augment, compute advantages, etc.)

        Args:
            trajectories: Raw trajectories from actors

        Returns:
            Processed trajectories ready for training
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        model_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute training loss for the given batch.

        Args:
            model: The VLM being trained
            batch: Collated batch from collate_trajectories()
            model_outputs: Optional pre-computed model outputs (for efficiency)

        Returns:
            Scalar loss tensor
        """
        pass


# Alias for backward compatibility
RLAlgorithm = Algorithm
