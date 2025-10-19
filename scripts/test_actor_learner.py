#!/usr/bin/env python3
"""
Test script for integrated Actor-Learner setup with real VM communication.

This script tests the full training loop:
- TaskRunner (Actor) collects trajectories from VM
- Trainer (Learner) waits for n trajectories and runs training steps

Usage:
    python scripts/test_actor_learner.py --vm-url http://VM_IP:8000 --num-trajectories 5
    python scripts/test_actor_learner.py --vm-url http://VM_IP:8000 --num-actors 2 --num-trajectories 10
"""

import argparse
import logging
import sys
import queue
import time
import threading
from typing import Union, List, Dict, Any
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.actor.task_runner import TaskRunner
from src.learner.trainer import Trainer
from src.learner.algorithms.rejection_sampling import RejectionSampling
from src.orchestration import ActorPoolManager


class MockVLM(nn.Module):
    """
    Mock VLM for testing Actor-Learner communication.

    Returns simple predefined actions and supports training interface.
    """

    def __init__(
        self,
        action_type: str = "left_click",
        x: int = 100,
        y: int = 100,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__()
        self.action_type = action_type
        self.x = x
        self.y = y
        self.device = device
        self.step_count = 0

        # Simple dummy parameters to make this a valid nn.Module for training
        self.dummy_linear = nn.Linear(10, 10)

        logging.info(f"MockVLM initialized - action: {action_type} at ({x}, {y})")

    def predict_action(
        self,
        images: Union[torch.Tensor, Image.Image, np.ndarray],
        prompt: str
    ) -> str:
        """Generate action text (mocked)."""
        self.step_count += 1
        action_json = f'{{"action_type": "{self.action_type}", "x": {self.x}, "y": {self.y}}}'
        logging.debug(f"MockVLM step {self.step_count}: {action_json}")
        return action_json

    def forward(
        self,
        images: Union[torch.Tensor, List[Image.Image], np.ndarray],
        prompts: Union[str, List[str]],
        return_loss: bool = True,
        labels: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training (mocked).

        Returns dummy outputs that match expected interface.
        """
        # Create dummy outputs
        batch_size = 1 if isinstance(prompts, str) else len(prompts)

        # Dummy logits
        logits = torch.randn(batch_size, 10, 100)  # [B, seq_len, vocab_size]

        # Dummy loss
        loss = torch.tensor(0.5, requires_grad=True)

        return {
            "logits": logits,
            "loss": loss,
            "action_logits": logits[:, -1, :]  # Last token logits as action
        }

    def parameters(self):
        """Return parameters for optimizer."""
        return self.dummy_linear.parameters()

    def state_dict(self):
        """Return state dict for checkpointing."""
        return self.dummy_linear.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.dummy_linear.load_state_dict(state_dict)


class MockRejectionSampling(RejectionSampling):
    """
    Mock version of RejectionSampling that doesn't require real action processing.
    """

    def compute_loss(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        model_outputs: Dict[str, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Mock loss computation that works with MockVLM.

        Simply returns a dummy loss that can be backpropagated.
        """
        # Call model to get outputs
        if model_outputs is None:
            # For mock, just return a simple loss
            pass

        # Return dummy loss with gradient
        dummy_loss = torch.tensor(0.5, requires_grad=True)
        dummy_loss = dummy_loss * torch.sum(torch.stack([p.sum() for p in model.parameters()]))

        return dummy_loss


# Actor management now handled by ActorPoolManager


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Reduce noise
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="Test integrated Actor-Learner training loop"
    )

    # Required arguments
    parser.add_argument(
        "--vm-url",
        type=str,
        required=True,
        help="URL of ui-verifiers VM (e.g., http://34.123.45.67:8000)"
    )

    parser.add_argument(
        "--num-trajectories",
        type=int,
        required=True,
        help="Number of trajectories to collect before each training step"
    )

    # Optional arguments
    parser.add_argument(
        "--num-actors",
        type=int,
        default=2,
        help="Target number of concurrent actors (default: 2)"
    )

    parser.add_argument(
        "--max-concurrent-per-vm",
        type=int,
        default=2,
        help="Max concurrent sessions per VM (memory limit, default: 2)"
    )

    parser.add_argument(
        "--num-training-steps",
        type=int,
        default=3,
        help="Number of training steps to run (default: 3)"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per episode (default: 50)"
    )

    parser.add_argument(
        "--session-type",
        type=str,
        default="simple_data_entry",
        help="Type of session to create (default: simple_data_entry)"
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (optional)"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    # Print configuration
    logger.info("=" * 60)
    logger.info("Actor-Learner Integration Test")
    logger.info("=" * 60)
    logger.info(f"VM URL: {args.vm_url}")
    logger.info(f"Target concurrent actors: {args.num_actors}")
    logger.info(f"Max concurrent per VM: {args.max_concurrent_per_vm}")
    logger.info(f"Trajectories per training step: {args.num_trajectories}")
    logger.info(f"Number of training steps: {args.num_training_steps}")
    logger.info(f"Max steps per episode: {args.max_steps}")
    logger.info(f"Session type: {args.session_type}")
    logger.info("=" * 60)

    # Create shared trajectory queue
    trajectory_queue = queue.Queue()

    # Create Mock VLM
    logger.info("\nCreating MockVLM...")
    mock_vlm = MockVLM(action_type="left_click", x=100, y=100)

    # Create Trainer (owns the trajectory queue)
    logger.info("Creating Trainer...")
    algorithm = MockRejectionSampling(reward_threshold=0.0)
    trainer = Trainer(
        model=mock_vlm,
        algorithm=algorithm,
        trajectory_queue=trajectory_queue,
        batch_size=args.num_trajectories,
        learning_rate=1e-5,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None
    )

    # Create ActorPoolManager
    logger.info("\nCreating ActorPoolManager...")
    actor_pool = ActorPoolManager(
        target_concurrent_actors=args.num_actors,
        vm_urls=[args.vm_url],  # Can support multiple VMs
        max_concurrent_per_vm=args.max_concurrent_per_vm,
        model=mock_vlm,
        trajectory_queue=trajectory_queue,
        task_prompt="Complete the data entry task",
        session_type=args.session_type,
        max_steps_per_episode=args.max_steps,
        action_format="json"
    )

    # Start actor pool
    logger.info("\nStarting actor pool...")
    actor_pool.start()

    # Run training loop - Trainer handles waiting for trajectories
    logger.info("\nStarting Trainer...")
    try:
        metrics = trainer.train(
            num_steps=args.num_training_steps,
            save_every=2 if args.checkpoint_dir else None
        )
    except KeyboardInterrupt:
        logger.info("\nStopping due to keyboard interrupt...")

    # Stop actor pool
    logger.info("\nStopping actor pool...")
    actor_pool.stop(timeout=30)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Training steps completed: {trainer.step}")
    logger.info(f"Total trajectories processed: {trainer.trajectories_seen}")

    # Actor pool stats
    pool_stats = actor_pool.get_stats()
    logger.info(f"Episodes collected by pool: {pool_stats['total_episodes_collected']}")
    logger.info(f"Total actors spawned: {pool_stats['total_actors_spawned']}")
    logger.info(f"Actors crashed: {pool_stats['actors_crashed']}")
    logger.info(f"Active actors at end: {pool_stats['active_actors']}")

    if trainer.step == args.num_training_steps:
        logger.info("\n✓ Actor-Learner integration test complete!")
        return 0
    else:
        logger.error(f"\n✗ Only completed {trainer.step}/{args.num_training_steps} training steps")
        return 1


if __name__ == "__main__":
    sys.exit(main())
