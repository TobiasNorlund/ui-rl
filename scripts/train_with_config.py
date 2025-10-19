#!/usr/bin/env python3
"""
Main training script using configuration files.

This script loads all settings from a YAML config file, making it easy to:
- Run experiments with different hyperparameters
- Compare LoRA targeting strategies
- Scale to multiple VMs
- Reproduce results

Usage:
    # Use default config
    python scripts/train_with_config.py --config config/qwen25_vl_lora.yaml

    # Use custom cluster host (override config)
    python scripts/train_with_config.py --config config/qwen25_vl_lora.yaml --cluster-host 34.123.45.67

    # Run LoRA ablation experiments
    python scripts/train_with_config.py --config config/experiments/lora_ablation_attention_only.yaml
    python scripts/train_with_config.py --config config/experiments/lora_ablation_mlp_only.yaml

    # Kubernetes distributed training
    python scripts/train_with_config.py --config config/multi_vm_distributed.yaml
"""

import argparse
import logging
import sys
import queue
import torch
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.models.vlm_wrapper import VLMWrapper
from src.learner.trainer import Trainer
from src.learner.algorithms.rejection_sampling import RejectionSampling
from src.orchestration import ActorPoolManager

# Import pod manifest functions
# Note: User should create their own pod manifest functions for their tasks
# This is just importing the example from ui_rl for simple_data_entry
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "ui_rl"))
    from simple_data_entry import simple_data_entry_pod_manifest
    POD_MANIFEST_REGISTRY = {
        "simple_data_entry": simple_data_entry_pod_manifest
    }
except ImportError:
    # If ui_rl module not available, user must provide their own manifests
    POD_MANIFEST_REGISTRY = {}


def setup_logging(config: Config):
    """Setup logging from config."""
    level = logging.DEBUG if config.logging.verbose else getattr(logging, config.logging.log_level)

    # Create log directory
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Log to both file and console
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=handlers
    )

    # Reduce noise from third-party libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")


def create_model_from_config(config: Config) -> VLMWrapper:
    """Create VLM from config."""
    logger = logging.getLogger(__name__)
    logger.info("Loading VLM from config...")

    # Convert torch_dtype string to torch.dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.model.torch_dtype, torch.float16)

    # Prepare LoRA config if enabled
    lora_config = None
    if config.model.use_lora:
        lora_config = {
            "r": config.model.lora_rank,
            "lora_alpha": config.model.lora_alpha,
            "lora_dropout": config.model.lora_dropout,
            "bias": config.model.lora_bias,
        }
        if config.model.lora_custom_modules:
            lora_config["custom_modules"] = config.model.lora_custom_modules

    model = VLMWrapper(
        model_name=config.model.name,
        device=config.model.device,
        torch_dtype=torch_dtype,
        use_lora=config.model.use_lora,
        lora_config=lora_config,
        lora_target_option=config.model.lora_target_option if config.model.use_lora else None,
        max_new_tokens=config.model.max_new_tokens,
        temperature=config.model.temperature,
        do_sample=config.model.do_sample,
    )

    # Apply vision encoder freezing if requested
    if config.model.freeze_vision_encoder:
        logger.info("Freezing vision encoder...")
        model.freeze_vision_encoder()

    return model


def create_trainer_from_config(config: Config, model: VLMWrapper, trajectory_queue: queue.Queue) -> Trainer:
    """Create Trainer from config."""
    logger = logging.getLogger(__name__)
    logger.info("Creating Trainer from config...")

    # Create algorithm
    if config.trainer.algorithm == "rejection_sampling":
        algorithm = RejectionSampling(reward_threshold=config.trainer.reward_threshold)
    else:
        raise ValueError(f"Unsupported algorithm: {config.trainer.algorithm}")

    # Create checkpoint directory
    checkpoint_dir = Path(config.logging.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        algorithm=algorithm,
        trajectory_queue=trajectory_queue,
        batch_size=config.trainer.batch_size,
        learning_rate=config.trainer.learning_rate,
        checkpoint_dir=checkpoint_dir,
        device=config.model.device,
        queue_timeout=config.trainer.queue_timeout
    )

    return trainer


def create_actor_pool_from_config(
    config: Config,
    model: VLMWrapper,
    trajectory_queue: queue.Queue
) -> ActorPoolManager:
    """Create ActorPoolManager from config."""
    logger = logging.getLogger(__name__)
    logger.info("Creating ActorPoolManager from config...")

    # Get pod manifest function from registry
    session_type = config.actor.session_type
    if session_type not in POD_MANIFEST_REGISTRY:
        raise ValueError(
            f"No pod manifest function found for session_type '{session_type}'. "
            f"Available types: {list(POD_MANIFEST_REGISTRY.keys())}. "
            f"Please add your pod manifest function to POD_MANIFEST_REGISTRY."
        )
    pod_manifest_fn = POD_MANIFEST_REGISTRY[session_type]

    actor_pool = ActorPoolManager(
        target_concurrent_actors=config.actor_pool.target_concurrent_actors,
        cluster_host=config.environment.cluster_host,
        pod_manifest_fn=pod_manifest_fn,
        model=model,
        trajectory_queue=trajectory_queue,
        task_prompt=config.actor.task_prompt,
        namespace=config.environment.namespace,
        max_steps_per_episode=config.actor.max_steps_per_episode,
        action_format=config.actor.action_format,
        data_dir=config.actor.data_dir,
        monitor_interval=config.actor_pool.monitor_interval,
        session_timeout=config.environment.session_timeout
    )

    return actor_pool


def main():
    parser = argparse.ArgumentParser(
        description="Train VLM for UI tasks using config files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    # Optional overrides
    parser.add_argument(
        "--cluster-host",
        type=str,
        help="Override cluster host from config"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Override number of training steps"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Override checkpoint directory"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    try:
        config = Config.from_yaml(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1

    # Apply command-line overrides
    if args.cluster_host:
        config.environment.cluster_host = args.cluster_host

    if args.num_steps:
        config.trainer.num_training_steps = args.num_steps

    if args.checkpoint_dir:
        config.logging.checkpoint_dir = args.checkpoint_dir

    if args.verbose:
        config.logging.verbose = True

    # Validate config
    try:
        config.validate()
    except ValueError as e:
        print(f"Config validation error: {e}")
        return 1

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    # Print configuration
    logger.info("\n" + str(config))

    # Save config to experiment directory
    config_save_path = Path(config.logging.checkpoint_dir) / "config.yaml"
    config.to_yaml(config_save_path)

    # Create shared trajectory queue
    trajectory_queue = queue.Queue()

    # Create model
    try:
        model = create_model_from_config(config)
    except Exception as e:
        logger.error(f"Failed to create model: {e}", exc_info=True)
        return 1

    # Create trainer
    try:
        trainer = create_trainer_from_config(config, model, trajectory_queue)
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}", exc_info=True)
        return 1

    # Create actor pool
    try:
        actor_pool = create_actor_pool_from_config(config, model, trajectory_queue)
    except Exception as e:
        logger.error(f"Failed to create actor pool: {e}", exc_info=True)
        return 1

    # Start actor pool
    logger.info("\nStarting actor pool...")
    actor_pool.start()

    # Run training
    logger.info("\nStarting training...")
    start_time = datetime.now()

    try:
        metrics = trainer.train(
            num_steps=config.trainer.num_training_steps,
            save_every=config.trainer.save_every
        )
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nTraining failed: {e}", exc_info=True)
    finally:
        # Always stop actor pool gracefully
        logger.info("\nStopping actor pool...")
        actor_pool.stop(timeout=30)

    # Training complete
    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    logger.info(f"Training steps completed: {trainer.step}")
    logger.info(f"Total trajectories processed: {trainer.trajectories_seen}")

    # Actor pool stats
    pool_stats = actor_pool.get_stats()
    logger.info(f"Episodes collected: {pool_stats['total_episodes_collected']}")
    logger.info(f"Total actors spawned: {pool_stats['total_actors_spawned']}")
    logger.info(f"Actors crashed: {pool_stats['actors_crashed']}")

    if trainer.step > 0:
        logger.info(f"Average trajectories per step: {trainer.trajectories_seen / trainer.step:.1f}")
        logger.info(f"Steps per second: {trainer.step / elapsed:.2f}")

    # Final checkpoint
    logger.info("\nSaving final checkpoint...")
    trainer.save_checkpoint(filename="final_checkpoint.pt")

    # Save model
    final_model_path = Path(config.logging.checkpoint_dir) / "final_model"
    logger.info(f"Saving final model to {final_model_path}...")
    model.save_pretrained(str(final_model_path))

    logger.info("\n✓ Training complete!")
    logger.info(f"Model saved to: {final_model_path}")
    logger.info(f"Checkpoints saved to: {config.logging.checkpoint_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
