#!/usr/bin/env python3
"""
Test script for validating TaskRunner-to-VM communication.

This script creates 1..n TaskRunner instances with a Mock VLM to test
the communication between TaskRunners and the ui-verifiers VM without
requiring a real VLM model.

Usage:
    python scripts/test_vm_connection.py --vm-url http://VM_IP:8000
    python scripts/test_vm_connection.py --vm-url http://VM_IP:8000 --num-runners 3
    python scripts/test_vm_connection.py --vm-url http://VM_IP:8000 --max-steps 20
"""

import argparse
import logging
import sys
import queue
import time
import threading
from typing import Union, List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.actor.task_runner import TaskRunner


class MockVLM(nn.Module):
    """
    Mock VLM for testing VM communication without loading a real model.

    This mock mimics the VLMWrapper interface but returns simple predefined actions.
    You can easily modify the actions it returns to test different scenarios.
    """

    def __init__(
        self,
        action_type: str = "left_click",
        x: int = 100,
        y: int = 100,
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize Mock VLM.

        Args:
            action_type: Type of action to return (e.g., "left_click", "screenshot")
            x: X coordinate for click actions
            y: Y coordinate for click actions
            device: Device (ignored, always CPU for mock)
            **kwargs: Other arguments (ignored, for compatibility)
        """
        super().__init__()
        self.action_type = action_type
        self.x = x
        self.y = y
        self.device = device
        self.step_count = 0

        # Simple dummy parameter to make this a valid nn.Module
        self.dummy_param = nn.Parameter(torch.zeros(1))

        logging.info(f"MockVLM initialized - will return: {action_type} at ({x}, {y})")

    def predict_action(
        self,
        images: Union[torch.Tensor, Image.Image, np.ndarray],
        prompt: str
    ) -> str:
        """
        Generate action text (mocked).

        Returns a JSON string representing the action to take.
        TaskRunner will decode this using ActionsDecoder.

        Args:
            images: Input image (ignored)
            prompt: Task prompt (ignored)

        Returns:
            JSON string with action
        """
        self.step_count += 1

        # Return action as JSON string that ActionsDecoder can parse
        # Format: {"action_type": "...", "x": ..., "y": ...}
        action_json = f'{{"action_type": "{self.action_type}", "x": {self.x}, "y": {self.y}}}'

        logging.debug(f"MockVLM step {self.step_count}: returning {action_json}")
        return action_json

    def eval(self):
        """Set to eval mode (no-op for mock)."""
        return self

    def train(self, mode=True):
        """Set to train mode (no-op for mock)."""
        return self


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Reduce noise from some libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def run_single_episode(
    runner_id: int,
    vm_url: str,
    max_steps: int,
    session_type: str,
    action_type: str,
    x: int,
    y: int,
    trajectory_queue: queue.Queue,
    results_queue: queue.Queue
):
    """
    Run a single episode with one TaskRunner (thread-safe).

    Args:
        runner_id: ID for this runner (for logging)
        vm_url: URL of ui-verifiers VM
        max_steps: Maximum steps per episode
        session_type: Type of session to create
        action_type: Type of action for MockVLM to return
        x: X coordinate for click actions
        y: Y coordinate for click actions
        trajectory_queue: Queue to put completed trajectory
        results_queue: Queue to put (runner_id, success) result
    """
    logger = logging.getLogger(f"Runner-{runner_id}")
    logger.info(f"Starting TaskRunner {runner_id}")

    try:
        # Create Mock VLM
        mock_vlm = MockVLM(
            action_type=action_type,
            x=x,
            y=y
        )

        # Create TaskRunner
        runner = TaskRunner(
            ui_env_url=vm_url,
            model=mock_vlm,
            task_prompt="Complete the data entry task",
            session_type=session_type,
            max_steps_per_episode=max_steps,
            action_delay=1.0,
            action_format="json"
        )

        logger.info(f"Runner {runner_id}: Starting episode")
        start_time = time.time()

        # Run one complete episode
        trajectory = runner.run_episode()
        trajectory_queue.put(trajectory)

        elapsed = time.time() - start_time

        # Log results
        logger.info(f"Runner {runner_id}: Episode completed!")
        logger.info(f"  Duration: {elapsed:.2f}s")
        logger.info(f"  Steps: {len(trajectory.observations)}")
        logger.info(f"  Total reward: {trajectory.total_reward():.3f}")
        logger.info(f"  Success: {trajectory.metadata.get('success', False)}")
        logger.info(f"  Termination: {trajectory.metadata.get('termination_reason', 'unknown')}")

        results_queue.put((runner_id, True))

    except Exception as e:
        logger.error(f"Runner {runner_id}: Failed with error: {e}", exc_info=True)
        results_queue.put((runner_id, False))


def main():
    parser = argparse.ArgumentParser(
        description="Test TaskRunner-to-VM communication with Mock VLM"
    )

    # Required arguments
    parser.add_argument(
        "--vm-url",
        type=str,
        required=True,
        help="URL of ui-verifiers VM (e.g., http://34.123.45.67:8000)"
    )

    # Optional arguments
    parser.add_argument(
        "--num-runners",
        type=int,
        default=1,
        help="Number of TaskRunner instances to create (default: 1)"
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
        "--action-type",
        type=str,
        default="left_click",
        choices=["left_click", "right_click", "double_click", "triple_click", "mouse_move", "screenshot"],
        help="Type of action for MockVLM to return (default: left_click)"
    )

    parser.add_argument(
        "--x",
        type=int,
        default=100,
        help="X coordinate for click actions (default: 100)"
    )

    parser.add_argument(
        "--y",
        type=int,
        default=100,
        help="Y coordinate for click actions (default: 100)"
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
    logger.info("VM-TaskRunner Communication Test")
    logger.info("=" * 60)
    logger.info(f"VM URL: {args.vm_url}")
    logger.info(f"Number of runners: {args.num_runners}")
    logger.info(f"Max steps per episode: {args.max_steps}")
    logger.info(f"Session type: {args.session_type}")
    logger.info(f"Mock action: {args.action_type} at ({args.x}, {args.y})")
    logger.info("=" * 60)

    # Create shared queues
    trajectory_queue = queue.Queue()
    results_queue = queue.Queue()

    # Run episodes
    if args.num_runners == 1:
        # Sequential execution for single runner
        logger.info("Running single episode...")
        run_single_episode(
            runner_id=1,
            vm_url=args.vm_url,
            max_steps=args.max_steps,
            session_type=args.session_type,
            action_type=args.action_type,
            x=args.x,
            y=args.y,
            trajectory_queue=trajectory_queue,
            results_queue=results_queue
        )
    else:
        # Parallel execution with threading for multiple runners
        logger.info(f"Running {args.num_runners} episodes in parallel using threads...")

        threads = []
        for i in range(1, args.num_runners + 1):
            thread = threading.Thread(
                target=run_single_episode,
                kwargs={
                    'runner_id': i,
                    'vm_url': args.vm_url,
                    'max_steps': args.max_steps,
                    'session_type': args.session_type,
                    'action_type': args.action_type,
                    'x': args.x,
                    'y': args.y,
                    'trajectory_queue': trajectory_queue,
                    'results_queue': results_queue
                },
                name=f"Runner-{i}"
            )
            threads.append(thread)
            thread.start()
            logger.info(f"Started thread for Runner {i}")

        # Wait for all threads to complete
        logger.info(f"Waiting for {len(threads)} threads to complete...")
        for thread in threads:
            thread.join()

        logger.info("All threads completed!")

    # Collect results from queue
    results = {}
    while not results_queue.empty():
        runner_id, success = results_queue.get()
        results[runner_id] = success

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total episodes: {len(results)}")
    logger.info(f"Successful: {sum(results.values())}")
    logger.info(f"Failed: {len(results) - sum(results.values())}")
    logger.info(f"Trajectories collected: {trajectory_queue.qsize()}")

    # Show per-runner results
    if len(results) > 1:
        logger.info("\nPer-runner results:")
        for runner_id in sorted(results.keys()):
            status = "✓ Success" if results[runner_id] else "✗ Failed"
            logger.info(f"  Runner {runner_id}: {status}")

    if all(results.values()):
        logger.info("\n✓ All episodes completed successfully!")
        return 0
    else:
        logger.error(f"\n✗ {len(results) - sum(results.values())} episode(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
