"""Trajectory data structure for storing episode information."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image


@dataclass
class Trajectory:
    """
    Represents a single episode trajectory.

    Design: Simple dataclass that just holds episode data.
    No logic here - keeps it easy to serialize/deserialize.

    Note: Observations are stored as PIL Images to avoid unnecessary conversions.
    This keeps images in their native format throughout the pipeline:
    PIL (from VM) → PIL (stored) → PIL (batched) → PIL (to VLM)
    """
    observations: List[Image.Image]  # Screenshots as PIL Images [T]
    generated_texts: List[str]    # Generated texts, e.g. "click(x, y)"
    rewards: List[float]             # Rewards [T]
    prompts: List[str]               # Task prompts [T]
    metadata: Dict[str, Any] = field(default_factory=dict)  # task_id, success, timestamps, etc.

    def __len__(self) -> int:
        """Return the length of the trajectory."""
        return len(self.observations)

    def total_reward(self) -> float:
        """Convenience method for filtering."""
        return sum(self.rewards)

    def to_dict(self) -> Dict[str, Any]:
        """
        For serialization.

        Note: PIL Images are converted to numpy arrays for serialization.
        This is necessary for saving to disk (pickle, npz, etc.).
        """
        observations_np = [np.array(img) for img in self.observations] if self.observations else []
        return {
            'observations': np.stack(observations_np) if observations_np else np.array([]),
            'generated_texts': self.generated_texts,
            'rewards': np.array(self.rewards),
            'prompts': self.prompts,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trajectory':
        """
        For deserialization.

        Note: Numpy arrays are converted back to PIL Images.
        """
        # Convert numpy arrays back to PIL Images
        observations_pil = [Image.fromarray(obs.astype(np.uint8)) for obs in data['observations']]

        return cls(
            observations=observations_pil,
            generated_texts=data['generated_texts'],
            rewards=list(data['rewards']),
            prompts=data['prompts'],
            metadata=data['metadata']
        )
