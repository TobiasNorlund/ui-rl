#!/usr/bin/env python3
"""
Test script for running a single episode with VLM inference.

This script:
- Creates a Kubernetes pod for a single session
- Runs VLM inference to generate actions
- Executes actions in the environment
- Saves the complete trajectory to file for analysis
- NO training - just inference and data collection

Usage:
    # Basic usage with default VLM
    python scripts/test_single_episode.py --cluster-host 34.123.45.67

    # Specify custom VLM model
    python scripts/test_single_episode.py \
        --cluster-host 34.123.45.67 \
        --model Qwen/Qwen2.5-VL-3B-Instruct

    # Custom number of steps and output directory
    python scripts/test_single_episode.py \
        --cluster-host 34.123.45.67 \
        --max-steps 20 \
        --output-dir ./test_trajectories
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.actor.task_runner import TaskRunner
from src.models.vlm_wrapper import VLMWrapper

# Import pod manifest functions
sys.path.insert(0, str(Path(__file__).parent.parent / "ui_rl"))
from simple_data_entry import simple_data_entry_pod_manifest


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Reduce noise from libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('kubernetes').setLevel(logging.WARNING)


def save_trajectory_analysis(trajectory, output_dir: Path):
    """
    Save trajectory data in multiple formats for analysis.

    Args:
        trajectory: Trajectory object with observations, actions, rewards
        output_dir: Directory to save analysis files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = trajectory.metadata.get('session_id', 'unknown')

    # 1. Save metadata as JSON
    metadata_file = output_dir / f"metadata_{timestamp}_{session_id}.json"
    with open(metadata_file, 'w') as f:
        json.dump(trajectory.metadata, f, indent=2)
    logging.info(f"Saved metadata to {metadata_file}")

    # 2. Save action sequence as JSON for easy reading
    actions_file = output_dir / f"actions_{timestamp}_{session_id}.json"
    actions_data = {
        'session_id': session_id,
        'timestamp': timestamp,
        'total_steps': len(trajectory.generated_texts),
        'total_reward': trajectory.total_reward(),
        'actions': [
            {
                'step': i,
                'generated_text': text,
                'reward': reward,
                'prompt': prompt
            }
            for i, (text, reward, prompt) in enumerate(
                zip(trajectory.generated_texts, trajectory.rewards, trajectory.prompts)
            )
        ]
    }
    with open(actions_file, 'w') as f:
        json.dump(actions_data, f, indent=2)
    logging.info(f"Saved action sequence to {actions_file}")

    # 3. Save screenshots
    screenshots_dir = output_dir / f"screenshots_{timestamp}_{session_id}"
    screenshots_dir.mkdir(exist_ok=True)
    for i, obs in enumerate(trajectory.observations):
        screenshot_file = screenshots_dir / f"step_{i:03d}.png"
        obs.save(screenshot_file)
    logging.info(f"Saved {len(trajectory.observations)} screenshots to {screenshots_dir}")

    # 4. Save full trajectory as numpy archive (original format)
    npz_file = output_dir / f"trajectory_{timestamp}_{session_id}.npz"
    trajectory_dict = trajectory.to_dict()
    # Save with numpy (observations are PIL Images, convert to arrays)
    import numpy as np
    obs_arrays = [np.array(obs) for obs in trajectory.observations]
    np.savez_compressed(
        npz_file,
        observations=obs_arrays,
        generated_texts=trajectory.generated_texts,
        rewards=trajectory.rewards,
        prompts=trajectory.prompts,
        metadata=json.dumps(trajectory.metadata)
    )
    logging.info(f"Saved trajectory archive to {npz_file}")

    # 5. Print summary to console
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Session ID: {session_id}")
    print(f"Timestamp: {timestamp}")
    print(f"Total Steps: {len(trajectory.generated_texts)}")
    print(f"Total Reward: {trajectory.total_reward():.2f}")
    print(f"Success: {trajectory.metadata.get('success', False)}")
    print(f"Termination: {trajectory.metadata.get('termination_reason', 'unknown')}")
    print("\nActions taken:")
    for i, text in enumerate(trajectory.generated_texts):
        print(f"  Step {i}: {text} (reward: {trajectory.rewards[i]:.2f})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Test VLM inference on a single episode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--cluster-host",
        type=str,
        required=True,
        help="Kubernetes cluster proxy server host/IP"
    )

    # Optional VLM configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-VL-3B-Instruct)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA fine-tuning (requires lora-checkpoint)"
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        help="Path to LoRA checkpoint directory"
    )

    # Episode configuration
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum steps per episode (default: 10)"
    )
    parser.add_argument(
        "--task-prompt",
        type=str,
        default="Complete the data entry task",
        help="Task prompt for VLM (default: 'Complete the data entry task')"
    )
    parser.add_argument(
        "--session-type",
        type=str,
        default="simple_data_entry",
        help="Session type / pod manifest to use (default: simple_data_entry)"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Kubernetes namespace (default: default)"
    )
    parser.add_argument(
        "--session-timeout",
        type=int,
        default=300,
        help="Pod startup timeout in seconds (default: 300)"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_episodes",
        help="Directory to save trajectory files (default: ./test_episodes)"
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
    logger.info("Single Episode Test")
    logger.info("=" * 60)
    logger.info(f"Cluster host: {args.cluster_host}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Task prompt: {args.task_prompt}")
    logger.info(f"Session type: {args.session_type}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)

    # Get pod manifest function
    POD_MANIFEST_REGISTRY = {
        "simple_data_entry": simple_data_entry_pod_manifest
    }

    if args.session_type not in POD_MANIFEST_REGISTRY:
        logger.error(
            f"Unknown session type: {args.session_type}. "
            f"Available: {list(POD_MANIFEST_REGISTRY.keys())}"
        )
        return 1

    pod_manifest_fn = POD_MANIFEST_REGISTRY[args.session_type]

    # Create VLM
    logger.info("Loading VLM...")
    try:
        vlm = VLMWrapper(
            model_name=args.model,
            device=args.device,
            use_lora=args.use_lora,
            lora_checkpoint=args.lora_checkpoint if args.use_lora else None
        )
        logger.info("VLM loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load VLM: {e}")
        return 1

    # Create TaskRunner
    logger.info("Creating TaskRunner...")
    try:
        runner = TaskRunner(
            cluster_host=args.cluster_host,
            pod_manifest_fn=pod_manifest_fn,
            model=vlm,
            task_prompt=args.task_prompt,
            namespace=args.namespace,
            max_steps_per_episode=args.max_steps,
            session_timeout=args.session_timeout,
            action_format="json"
        )
        logger.info("TaskRunner created successfully")
    except Exception as e:
        logger.error(f"Failed to create TaskRunner: {e}")
        return 1

    # Run episode
    logger.info("\n" + "=" * 60)
    logger.info("Starting episode...")
    logger.info("=" * 60)

    try:
        trajectory = runner.run_episode()
        logger.info("Episode completed successfully!")
    except Exception as e:
        logger.error(f"Episode failed: {e}", exc_info=True)
        return 1

    # Save trajectory analysis
    logger.info("\nSaving trajectory analysis...")
    try:
        output_dir = Path(args.output_dir)
        save_trajectory_analysis(trajectory, output_dir)
        logger.info(f"\n✓ All files saved to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to save trajectory: {e}", exc_info=True)
        return 1

    logger.info("\n✓ Test completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
