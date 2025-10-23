"""Collation utilities for batching trajectories."""

from typing import List, Dict, Any
import numpy as np
from .trajectory import Trajectory


def collate_trajectories(trajectories: List[Trajectory]) -> Dict[str, Any]:
    """
    Collate multiple trajectories into a single training batch.

    Design: Standalone function (not part of Trajectory class) for flexibility.
    Can be easily customized without modifying Trajectory.

    Note: Images are kept as PIL Images (not converted to numpy/tensors) to maintain
    compatibility with VLMWrapper which expects PIL Images as input. This avoids
    unnecessary conversions: PIL → numpy → tensor → PIL.

    Args:
        trajectories: List of trajectories to batch together

    Returns:
        Dict containing batched data ready for model training:
        - images: List[PIL.Image] of all observations (kept as PIL for VLMWrapper)
        - generated_texts: [B*T] flattened generated_texts
        - rewards: [B*T] flattened rewards (numpy array)
        - prompts: [B*T] flattened prompts
        - trajectory_lengths: [B] length of each trajectory (for masking if needed)
        - trajectory_indices: [B*T] which trajectory each timestep belongs to
    """
    if not trajectories:
        raise ValueError("Cannot collate empty list of trajectories")

    all_observations = []
    all_generated_texts = []
    all_rewards = []
    all_prompts = []
    trajectory_lengths = []
    trajectory_indices = []

    for traj_idx, traj in enumerate(trajectories):
        # Extend with PIL Images directly (no conversion)
        all_observations.extend(traj.observations)
        all_generated_texts.extend(traj.generated_texts)
        all_rewards.extend(traj.rewards)
        all_prompts.extend(traj.prompts)
        trajectory_lengths.append(len(traj))
        trajectory_indices.extend([traj_idx] * len(traj))

    return {
        'images': all_observations,  # List of PIL Images
        'generated_texts': all_generated_texts,
        'rewards': np.array(all_rewards),
        'prompts': all_prompts,
        'trajectory_lengths': trajectory_lengths,
        'trajectory_indices': trajectory_indices
    }
