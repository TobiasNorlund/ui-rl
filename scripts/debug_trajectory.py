#!/usr/bin/env python3
"""Debug script for inspecting saved trajectories."""

import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils.trajectory import Trajectory


def load_trajectory(trajectory_path: Path) -> Trajectory:
    """Load trajectory from JSON file."""
    with open(trajectory_path, 'r') as f:
        data = json.load(f)
    return Trajectory.from_dict(data)


def print_trajectory_summary(trajectory: Trajectory):
    """Print summary of trajectory."""
    print("=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Length: {len(trajectory)} steps")
    print(f"Total reward: {sum(trajectory.rewards):.4f}")
    print(f"Average reward: {sum(trajectory.rewards) / len(trajectory.rewards) if trajectory.rewards else 0:.4f}")
    print(f"Metadata: {trajectory.metadata}")
    print()


def print_trajectory_details(trajectory: Trajectory, max_steps: int = None):
    """Print detailed step-by-step trajectory information."""
    print("=" * 60)
    print("TRAJECTORY DETAILS")
    print("=" * 60)

    steps_to_show = len(trajectory) if max_steps is None else min(max_steps, len(trajectory))

    for i in range(steps_to_show):
        print(f"\n--- Step {i+1} ---")

        # Action
        print(f"Action: {trajectory.actions[i]}")

        # Reward
        print(f"Reward: {trajectory.rewards[i]:.4f}")

        # Observation (abbreviated)
        obs = trajectory.observations[i+1] if i+1 < len(trajectory.observations) else None
        if obs:
            print(f"Next observation keys: {list(obs.keys())}")

    if max_steps and len(trajectory) > max_steps:
        print(f"\n... ({len(trajectory) - max_steps} more steps)")


def print_reward_distribution(trajectory: Trajectory):
    """Print reward distribution statistics."""
    rewards = trajectory.rewards

    if not rewards:
        print("No rewards in trajectory")
        return

    print("=" * 60)
    print("REWARD DISTRIBUTION")
    print("=" * 60)
    print(f"Min reward: {min(rewards):.4f}")
    print(f"Max reward: {max(rewards):.4f}")
    print(f"Mean reward: {sum(rewards) / len(rewards):.4f}")
    print(f"Total reward: {sum(rewards):.4f}")

    # Simple histogram
    print("\nReward histogram:")
    bins = 10
    min_r, max_r = min(rewards), max(rewards)
    if min_r == max_r:
        print(f"All rewards are {min_r:.4f}")
    else:
        bin_width = (max_r - min_r) / bins
        for i in range(bins):
            bin_start = min_r + i * bin_width
            bin_end = bin_start + bin_width
            count = sum(1 for r in rewards if bin_start <= r < bin_end or (i == bins-1 and r == bin_end))
            bar = '*' * count
            print(f"[{bin_start:6.2f}, {bin_end:6.2f}): {bar} ({count})")


def main():
    """Main debug entry point."""
    parser = argparse.ArgumentParser(description='Debug saved trajectories')
    parser.add_argument(
        'trajectory_file',
        type=str,
        help='Path to trajectory JSON file'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show summary only'
    )
    parser.add_argument(
        '--details',
        action='store_true',
        help='Show step-by-step details'
    )
    parser.add_argument(
        '--rewards',
        action='store_true',
        help='Show reward distribution'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Maximum number of steps to show in details'
    )
    args = parser.parse_args()

    # Load trajectory
    trajectory_path = Path(args.trajectory_file)
    if not trajectory_path.exists():
        print(f"Error: Trajectory file not found: {trajectory_path}")
        sys.exit(1)

    print(f"Loading trajectory from {trajectory_path}")
    trajectory = load_trajectory(trajectory_path)

    # If no specific flags, show everything
    show_all = not (args.summary or args.details or args.rewards)

    # Print requested information
    if show_all or args.summary:
        print_trajectory_summary(trajectory)

    if show_all or args.rewards:
        print_reward_distribution(trajectory)

    if show_all or args.details:
        print_trajectory_details(trajectory, max_steps=args.max_steps)


if __name__ == '__main__':
    main()
