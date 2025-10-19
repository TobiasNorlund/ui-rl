"""Trainer class for coordinating the training loop."""

from contextlib import nullcontext
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import queue
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from ..data_utils.trajectory import Trajectory
from ..data_utils.collation import collate_trajectories
from .algorithms.base import Algorithm

logger = logging.getLogger(__name__)


class Trainer:
    """
    Learner component: Trains the VLM using trajectories from actors.

    Design decisions:
    1. Owns the trajectory queue - waits for batch_size trajectories
    2. Uses composition (Algorithm class) for flexibility
    3. Delegates algorithm-specific logic to Algorithm class
    4. Handles model checkpointing and logging

    Responsibilities:
    - Wait for batch_size trajectories from queue
    - Process trajectories with algorithm
    - Run training step
    - Save checkpoints
    - Log metrics
    """

    def __init__(
        self,
        model: nn.Module,
        algorithm: Algorithm,
        trajectory_queue: queue.Queue,
        optimizer: Optional[torch.optim.Optimizer] = None,
        batch_size: int = 32,
        learning_rate: Optional[float] = None,
        checkpoint_dir: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        queue_timeout: float = 5.0
    ):
        """
        Args:
            model: VLM to train
            algorithm: Algorithm to use (RejectionSampling, PPO, etc.)
            trajectory_queue: Queue that actors put trajectories into
            optimizer: Optional optimizer (creates AdamW if None)
            batch_size: Number of trajectories per training batch
            learning_rate: Learning rate (auto-detected if None: 2e-4 for LoRA, 1e-5 for full)
            checkpoint_dir: Directory to save checkpoints
            device: Device for training
            queue_timeout: Timeout in seconds when waiting for trajectories
        """
        self.model = model.to(device)
        self.algorithm = algorithm
        self.trajectory_queue = trajectory_queue
        self.batch_size = batch_size
        self.device = device
        self.queue_timeout = queue_timeout

        # Auto-detect learning rate based on whether LoRA is used
        if learning_rate is None:
            if hasattr(model, 'peft_config'):
                # LoRA detected: use higher learning rate (research recommendation)
                learning_rate = 2e-4
                logger.info("Detected LoRA model, using learning_rate=2e-4")
            else:
                # Full fine-tuning: use lower learning rate
                learning_rate = 1e-5
                logger.info("Full fine-tuning model, using learning_rate=1e-5")
        else:
            logger.info(f"Using provided learning_rate={learning_rate}")

        self.learning_rate = learning_rate

        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optimizer

        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.step = 0
        self.trajectories_seen = 0

        logger.info(f"Trainer initialized with {algorithm.__class__.__name__}")
        logger.info(f"Batch size: {batch_size}, Queue timeout: {queue_timeout}s")

    def train_step(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """
        Run one training step on a batch of trajectories.

        Args:
            trajectories: Batch of trajectories to train on

        Returns:
            Dict of training metrics
        """
        # Let algorithm process trajectories (filter, compute advantages, etc.)
        processed_trajectories = self.algorithm.process_trajectories(trajectories)

        if len(processed_trajectories) == 0:
            logger.warning("No trajectories after algorithm processing, skipping training step")
            return {'loss': 0.0, 'trajectories_used': 0}

        # Collate into batch
        batch = collate_trajectories(processed_trajectories)

        # Move batch to device
        batch = self._batch_to_device(batch)

        model_lock = getattr(self.model, "execution_lock", None)
        lock_context = model_lock if model_lock is not None else nullcontext()

        with lock_context:
            self.model.train()
            self.optimizer.zero_grad()
            loss = self.algorithm.compute_loss(self.model, batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            loss_value = loss.detach().item()

        # Update state
        self.step += 1
        self.trajectories_seen += len(processed_trajectories)

        # Metrics
        metrics = {
            'loss': loss_value,
            'trajectories_used': len(processed_trajectories),
            'trajectories_filtered': len(trajectories) - len(processed_trajectories),
            'avg_trajectory_length': float(np.mean(batch['trajectory_lengths'])),
            'avg_trajectory_reward': float(np.mean([t.total_reward() for t in processed_trajectories]))
        }

        logger.info(f"Step {self.step}: loss={metrics['loss']:.4f}, "
                   f"trajs={metrics['trajectories_used']}")

        return metrics

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch tensors to training device.

        Note: Images are kept as PIL Images (List[Image.Image]) since VLMWrapper
        expects PIL Images as input. Only numpy arrays and tensors are moved to device.
        """
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                device_batch[key] = torch.from_numpy(value).to(self.device)
            elif isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                # Keep as-is (e.g., PIL Images, lists, dicts)
                device_batch[key] = value
        return device_batch

    def save_checkpoint(self, filename: Optional[str] = None):
        """
        Save model checkpoint.

        Args:
            filename: Optional filename, uses step number if None
        """
        if self.checkpoint_dir is None:
            logger.warning("No checkpoint_dir configured, skipping save")
            return

        if filename is None:
            filename = f"checkpoint_step_{self.step}.pt"

        filepath = self.checkpoint_dir / filename

        checkpoint = {
            'step': self.step,
            'trajectories_seen': self.trajectories_seen,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        try:
            torch.save(checkpoint, filepath)
            logger.info(f"Saved checkpoint to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, filepath: Path):
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.step = checkpoint['step']
            self.trajectories_seen = checkpoint['trajectories_seen']
            logger.info(f"Loaded checkpoint from {filepath} at step {self.step}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

    def _collect_batch(self) -> List[Trajectory]:
        """
        Collect batch_size trajectories from the queue.

        Waits until enough trajectories are available.

        Returns:
            List of trajectories (may be less than batch_size if queue closes)
        """
        trajectories = []

        logger.info(f"Waiting for {self.batch_size} trajectories from queue...")

        while len(trajectories) < self.batch_size:
            try:
                trajectory = self.trajectory_queue.get(timeout=self.queue_timeout)
                trajectories.append(trajectory)
                logger.debug(f"Collected trajectory {len(trajectories)}/{self.batch_size} "
                           f"(reward: {trajectory.total_reward():.2f}, "
                           f"steps: {len(trajectory.observations)})")
            except queue.Empty:
                # Queue is empty, continue waiting
                if len(trajectories) > 0:
                    logger.debug(f"Queue empty, waiting... ({len(trajectories)}/{self.batch_size} collected)")
                continue

        logger.info(f"Collected {len(trajectories)} trajectories for training")
        return trajectories

    def train(
        self,
        num_steps: Optional[int] = None,
        save_every: Optional[int] = None
    ) -> List[Dict[str, float]]:
        """
        Main training loop: waits for trajectories and trains.

        This method blocks and continuously waits for trajectories from the queue,
        runs training steps when enough trajectories are available.

        Args:
            num_steps: Number of training steps to run (None = infinite)
            save_every: Save checkpoint every N steps (None = don't save)

        Returns:
            List of metrics dictionaries from each training step
        """
        logger.info("=" * 60)
        logger.info("Starting training loop")
        logger.info(f"Training steps: {num_steps or '∞'}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info("=" * 60)

        all_metrics = []
        step = 0

        try:
            while num_steps is None or step < num_steps:
                logger.info(f"\n--- Training Step {step + 1}/{num_steps or '∞'} ---")

                # Collect batch of trajectories from queue
                trajectories = self._collect_batch()

                if len(trajectories) == 0:
                    logger.warning("No trajectories collected, stopping training")
                    break

                # Run training step
                metrics = self.train_step(trajectories)
                all_metrics.append(metrics)

                # Save checkpoint if configured
                if save_every and (step + 1) % save_every == 0:
                    self.save_checkpoint()

                step += 1

        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")

        logger.info("\n" + "=" * 60)
        logger.info("Training Complete")
        logger.info("=" * 60)
        logger.info(f"Total steps: {self.step}")
        logger.info(f"Total trajectories seen: {self.trajectories_seen}")

        return all_metrics
