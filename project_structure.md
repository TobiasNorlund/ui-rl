ui-vlm-training/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
│
├── config/                          # Configuration files
│   ├── __init__.py
│   ├── default_config.yaml          # Default training config
│   └── algorithms/                  # Algorithm-specific configs
│       ├── rejection_sampling.yaml
│       └── ppo.yaml
│
├── src/                             # Source code
│   ├── __init__.py
│   │
│   ├── data/                        # Data structures and utilities
│   │   ├── __init__.py
│   │   ├── trajectory.py            # Trajectory class
│   │   ├── collation.py             # collate_trajectories()
│   │   └── action_parser.py         # ActionParser class
│   │
│   ├── actor/                       # Actor (TaskRunner) components
│   │   ├── __init__.py
│   │   ├── task_runner.py           # TaskRunner class
│   │   └── env_client.py            # UI environment communication
│   │
│   ├── learner/                     # Learner (Trainer) components
│   │   ├── __init__.py
│   │   ├── trainer.py               # Trainer class
│   │   └── algorithms/              # RL algorithms
│   │       ├── __init__.py
│   │       ├── base.py              # Algorithm ABC
│   │       ├── rejection_sampling.py
│   │       └── ppo.py
│   │
│   ├── models/                      # Model wrappers
│   │   ├── __init__.py
│   │   └── vlm_wrapper.py           # VLM interface wrapper
│   │
│   ├── orchestration/               # Multi-VM orchestration
│   │   ├── __init__.py
│   │   ├── vm_manager.py            # GCP VM management
│   │   ├── env_pool.py              # Pool of UI environments
│   │   └── distributed_runner.py    # Distributed training coordinator
│   │
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── logging_utils.py
│       └── checkpoint_utils.py
│
├── scripts/                         # Executable scripts
│   ├── train.py                     # Main training script
│   ├── eval.py                      # Evaluation script
│   └── debug_trajectory.py          # Debug saved trajectories
│
├── tests/                           # Unit tests
│   ├── __init__.py
│   ├── test_trajectory.py
│   ├── test_action_parser.py
│   ├── test_task_runner.py
│   ├── test_trainer.py
│   ├── test_algorithms.py
│   └── test_integration.py
│
├── notebooks/                       # Jupyter notebooks for analysis
│   └── analyze_trajectories.ipynb
│
├── experiments/                     # Experiment configs and results
│   └── exp_001_rejection_sampling/
│       ├── config.yaml
│       ├── checkpoints/
│       ├── trajectories/
│       └── logs/
│
└── docker/                          # Docker files
    ├── Dockerfile.trainer
    └── Dockerfile.actor
