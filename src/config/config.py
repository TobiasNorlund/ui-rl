"""Configuration dataclasses for ui-rl training system."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """VLM model configuration."""
    # Model selection
    name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    device: str = "cuda"
    torch_dtype: str = "float16"  # "float16", "bfloat16", "float32"

    # LoRA configuration
    use_lora: bool = True
    lora_target_option: str = "attention+mlp"  # "attention", "mlp", "attention+mlp", "all-linear", "custom"
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_custom_modules: Optional[List[str]] = None  # Only used if lora_target_option="custom"

    # Generation parameters
    max_new_tokens: int = 128
    temperature: float = 0.7
    do_sample: bool = True

    # Training options
    freeze_vision_encoder: bool = False


@dataclass
class TrainerConfig:
    """Trainer/Learner configuration."""
    batch_size: int = 4
    learning_rate: Optional[float] = None  # Auto-detected if None (2e-4 for LoRA, 1e-5 for full)
    num_training_steps: int = 100
    save_every: int = 10
    queue_timeout: float = 5.0

    # Algorithm selection
    algorithm: str = "rejection_sampling"  # "rejection_sampling", "ppo"
    reward_threshold: float = 0.0  # For rejection sampling


@dataclass
class ActorConfig:
    """Actor/TaskRunner configuration."""
    max_steps_per_episode: int = 50
    action_format: str = "json"  # "json", "text", "coordinates"
    screenshot_size: List[int] = field(default_factory=lambda: [224, 224])
    task_prompt: str = "Complete the data entry task"
    session_type: str = "simple_data_entry"
    data_dir: Optional[str] = None  # Optional directory to save raw trajectories
    action_delay: float = 1.0  # Delay in seconds after each action


@dataclass
class ActorPoolConfig:
    """ActorPoolManager configuration."""
    target_concurrent_actors: int = 2
    max_concurrent_per_vm: int = 2  # Memory limit per VM
    monitor_interval: float = 2.0


@dataclass
class EnvironmentConfig:
    """Environment (VM) configuration."""
    vm_urls: List[str] = field(default_factory=lambda: ["http://localhost:8000"])
    timeout: int = 30
    max_retries: int = 3


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""
    log_level: str = "INFO"
    log_dir: str = "experiments/default/logs"
    checkpoint_dir: str = "checkpoints"
    trajectory_dir: Optional[str] = None  # Optional trajectory saving
    verbose: bool = False


@dataclass
class Config:
    """
    Main configuration class for ui-rl training.

    This combines all component configs into a single structure.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    actor_pool: ActorPoolConfig = field(default_factory=ActorPoolConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Experiment metadata
    experiment_name: str = "default"

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML config file

        Returns:
            Config instance
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create Config from dictionary.

        Args:
            config_dict: Dictionary with config values

        Returns:
            Config instance
        """
        # Extract nested configs
        model_config = ModelConfig(**config_dict.get('model', {}))
        trainer_config = TrainerConfig(**config_dict.get('trainer', {}))
        actor_config = ActorConfig(**config_dict.get('actor', {}))
        actor_pool_config = ActorPoolConfig(**config_dict.get('actor_pool', {}))
        environment_config = EnvironmentConfig(**config_dict.get('environment', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))

        experiment_name = config_dict.get('experiment_name', 'default')

        return cls(
            model=model_config,
            trainer=trainer_config,
            actor=actor_config,
            actor_pool=actor_pool_config,
            environment=environment_config,
            logging=logging_config,
            experiment_name=experiment_name
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'model': asdict(self.model),
            'trainer': asdict(self.trainer),
            'actor': asdict(self.actor),
            'actor_pool': asdict(self.actor_pool),
            'environment': asdict(self.environment),
            'logging': asdict(self.logging),
        }

    def to_yaml(self, yaml_path: Union[str, Path]):
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML config file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved config to {yaml_path}")

    def validate(self):
        """
        Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate model config
        if self.model.lora_target_option not in ["attention", "mlp", "attention+mlp", "all-linear", "custom"]:
            raise ValueError(f"Invalid lora_target_option: {self.model.lora_target_option}")

        if self.model.lora_target_option == "custom" and not self.model.lora_custom_modules:
            raise ValueError("lora_custom_modules required when lora_target_option='custom'")

        # Validate trainer config
        if self.trainer.algorithm not in ["rejection_sampling", "ppo"]:
            raise ValueError(f"Invalid algorithm: {self.trainer.algorithm}")

        if self.trainer.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Validate actor pool config
        if self.actor_pool.target_concurrent_actors <= 0:
            raise ValueError("target_concurrent_actors must be positive")

        if self.actor_pool.max_concurrent_per_vm <= 0:
            raise ValueError("max_concurrent_per_vm must be positive")

        # Validate environment config
        if not self.environment.vm_urls:
            raise ValueError("At least one VM URL required")

        logger.info("Configuration validated successfully")

    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = [
            "=" * 60,
            f"Experiment: {self.experiment_name}",
            "=" * 60,
            "",
            "Model Configuration:",
            f"  Name: {self.model.name}",
            f"  Device: {self.model.device}",
            f"  LoRA: {self.model.use_lora}",
        ]

        if self.model.use_lora:
            lines.extend([
                f"  LoRA Targeting: {self.model.lora_target_option}",
                f"  LoRA Rank: {self.model.lora_rank}",
                f"  LoRA Alpha: {self.model.lora_alpha}",
            ])

        lines.extend([
            "",
            "Training Configuration:",
            f"  Algorithm: {self.trainer.algorithm}",
            f"  Batch Size: {self.trainer.batch_size}",
            f"  Learning Rate: {self.trainer.learning_rate or 'auto-detect'}",
            f"  Training Steps: {self.trainer.num_training_steps}",
            "",
            "Actor Pool Configuration:",
            f"  Target Concurrent Actors: {self.actor_pool.target_concurrent_actors}",
            f"  Max Concurrent per VM: {self.actor_pool.max_concurrent_per_vm}",
            f"  VMs: {len(self.environment.vm_urls)}",
            "",
            "Actor Configuration:",
            f"  Max Steps per Episode: {self.actor.max_steps_per_episode}",
            f"  Session Type: {self.actor.session_type}",
            f"  Action Format: {self.actor.action_format}",
            "",
            "Logging Configuration:",
            f"  Log Level: {self.logging.log_level}",
            f"  Checkpoint Dir: {self.logging.checkpoint_dir}",
            "=" * 60,
        ])

        return "\n".join(lines)
