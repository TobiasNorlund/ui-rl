# UI-RL: Reinforcement Learning for Vision-Language Model UI Agents

This project implements an **Actor-Learner** RL training system for teaching VLMs to perform UI tasks.

### Current Implementation Status

✅ **Implemented:**
- TaskRunner (actor) for UI environment interaction
- Trainer (learner) with trajectory queue management
- VLMWrapper with LoRA support (attention, MLP, attention+mlp targeting)
- ActorPoolManager for multi-actor orchestration
- Rejection Sampling algorithm
- Configuration system with YAML files
- PIL-based image pipeline

## TODO
### High Priority
- [x] Migrate from VM sessions to Kubernetes pod-based sessions
- [x] Implement `/progress` endpoint for reward and done signals
- [ ] Current ActionsDecoder is generic and needs updating to match Qwen2.5-VL output format
- [ ] W&B integration for metrics
- [ ] Implement evaluation script and metrics

### Medium Priority
- [ ] **PPO Algorithm**: Implement Proximal Policy Optimization (see `src/learner/algorithms/base.py` for interface)

### Future Improvements
- [ ] Trajectory replay buffer for off-policy learning
- [ ] Multi-task training support


## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Configuration (YAML)                      │
│         model, trainer, actor, actor_pool, environment       │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────┐     ┌──────────────────┐
│  Kubernetes Cluster   │     │ ActorPoolManager │
│  ┌─────────────────┐  │     │                  │
│  │  Proxy Server   │  │     │  Manages pool    │
│  │  (Routes by     │  │     │  of actors       │
│  │   session_id)   │  │     │                  │
│  └────────┬────────┘  │     └────────┬─────────┘
│           │           │              │
│  ┌────────┴────────┐  │              │ spawns/monitors
│  │ Session Pods    │  │              │
│  │ ┌─────┐ ┌─────┐ │  │              │
│  │ │Pod 1│ │Pod 2│ │  │              ▼
│  │ └─────┘ └─────┘ │  │     ┌─────────────┐
│  └─────────────────┘  │     │ TaskRunner  │──┐
└───────────┬───────────┘     │  (Actor)    │  │
            │                 └─────────────┘  │
            │                                  │
            │ screenshots/actions              │
            │ via /proxy/{session_id}/*        │ Trajectories
            │                                  │ via Queue
            │                 ┌─────────────┐  │
            └─────────────────┤ TaskRunner  │──┤
                              │  (Actor)    │  │
                              └─────────────┘  │
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │     Trainer      │
                                    │    (Learner)     │
                                    │                  │
                                    │  ┌────────────┐  │
                                    │  │ VLMWrapper │  │
           model updates ←──────────┼──│ (w/ LoRA)  │  │
                                    │  └────────────┘  │
                                    └──────────────────┘
```

### Component Responsibilities

#### **TaskRunner** (src/actor/task_runner.py)
- Creates Kubernetes pods for each episode
- Runs VLM inference to get actions from screenshots
- Executes actions via proxy server routing
- Collects complete trajectories (observations, actions, rewards)
- Deletes pods when episode completes
- Puts finished trajectories in queue for training
- **Lifecycle**: Runs ONE episode (create pod → run → delete pod), then exits

#### **Learner: Trainer** (src/learner/trainer.py)
- Owns the trajectory queue
- Waits for `batch_size` trajectories from actors
- Processes trajectories with algorithm (e.g., filter by reward)
- Computes loss and runs training step
- Handles checkpointing and metrics logging
- **Auto-detects learning rate**: 2e-4 for LoRA, 1e-5 for full fine-tuning based on https://thinkingmachines.ai/blog/lora/. 

#### **VLMWrapper** (src/models/vlm_wrapper.py)
- Loads HuggingFace VLMs (`AutoModelForImageTextToText`)
- Applies LoRA with flexible layer targeting:
  - `"attention"`: q, k, v, o projections
  - `"mlp"`: gate, up, down projections
  - `"attention+mlp"`: both (recommended)
  - `"all-linear"`: all linear layers
  - `"custom"`: user-specified modules
- Handles inference (`predict_action`) and training (`forward`)
- Accepts PIL Images directly (no unnecessary conversions) as that is expected input to HF processor().

#### **ActorPoolManager** (src/orchestration/actor_pool_manager.py)
- Maintains pool of concurrent actors (TaskRunners)
- Automatically spawns replacement actors when episodes finish
- Monitor thread for health checks and crash recovery
- Kubernetes handles pod scheduling and resource management
- One actor = one episode = one Kubernetes pod

#### **Config System** (src/config/)
- YAML-based configuration for all components
- Dataclass-based validation
- Easy experimentation with different hyperparameters

## Installation

### Setup with UV (Recommended)

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone <repository-url>
cd ui-rl

# Create virtual environment and install dependencies
uv sync

# Activate environment
source .venv/bin/activate  # Unix
# or
.venv\Scripts\activate  # Windows
```

See [UV_USAGE.md](UV_USAGE.md) for more details.

### Setup with pip

```bash
pip install -e .
```

## Quick Start

### How to run Kubernetes

```bash
# 1. Setup kubctl
gcloud container clusters get-credentials simple-data-entry-cluster --region=europe-north2
 
# 2. Make sure the ui-verifiers proxy server is deployed
(cd /path/to/ui-verifiers/proxy; make deploy)

# 3. Find out public/external IP of proxy server
kubectl get services

# 4. Run demo script
uv run -m ui_rl.main <proxy ip>
```

### 1. Configure Your Training

Edit `config/qwen25_vl_lora.yaml` or create your own config:

```yaml
experiment_name: "my_experiment"

model:
  name: "Qwen/Qwen2.5-VL-3B-Instruct"
  use_lora: true
  lora_target_option: "attention+mlp"  # Recommended
  lora_rank: 32

trainer:
  batch_size: 4
  learning_rate: null  # Auto-detect
  num_training_steps: 100

actor:
  max_steps_per_episode: 50
  task_prompt: "Complete the data entry task"
  session_type: "simple_data_entry"  # Maps to pod manifest function

actor_pool:
  target_concurrent_actors: 2

environment:
  cluster_host: "your-proxy-server-ip"  # Kubernetes proxy server
  namespace: "default"
  session_timeout: 300
```

### 2. Run Training

```bash
# Train with config file
python scripts/train_with_config.py --config config/qwen25_vl_lora.yaml

# Override cluster host
python scripts/train_with_config.py \
    --config config/qwen25_vl_lora.yaml \
    --cluster-host 34.123.45.67

# Override training steps
python scripts/train_with_config.py \
    --config config/qwen25_vl_lora.yaml \
    --num-steps 500
```


## Project Structure

```
ui-rl/
├── src/
│   ├── actor/
│   │   ├── __init__.py
│   │   └── task_runner.py         # Episode runner, VLM inference, action execution
│   ├── learner/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training loop, queue management
│   │   └── algorithms/
│   │       ├── base.py             # Algorithm interface
│   │       └── rejection_sampling.py  # Rejection sampling implementation
│   ├── models/
│   │   ├── __init__.py
│   │   └── vlm_wrapper.py          # HuggingFace VLM wrapper with LoRA
│   ├── orchestration/
│   │   ├── __init__.py
│   │   └── actor_pool_manager.py   # Multi-actor lifecycle management
│   ├── data_utils/
│   │   ├── __init__.py
│   │   ├── trajectory.py           # Trajectory data structure
│   │   ├── collation.py            # Batching trajectories
│   │   └── actions_decoder.py      # Parse VLM outputs to actions
│   └── config/
│       ├── __init__.py
│       └── config.py               # Configuration dataclasses
├── config/                          # YAML configuration files
│   ├── README.md                   # Config documentation
│   ├── qwen25_vl_lora.yaml        # Recommended baseline
│   ├── qwen25_vl_full_finetune.yaml
│   ├── multi_vm_distributed.yaml
│   └── experiments/                # Ablation study configs
│       ├── lora_ablation_attention_only.yaml
│       └── lora_ablation_mlp_only.yaml
├── scripts/
│   ├── train_with_config.py       # Main training script
│   ├── test_vm_connection.py      # Test connection between TaskRunner and VM
│   └── test_actor_learner.py      # Test that TaskRunner and Trainer works together with the VM
└── docs/                           # Additional documentation
```

## Usage Examples

### Basic Training Loop

```python
from src.config import Config
from src.models.vlm_wrapper import VLMWrapper
from src.learner.trainer import Trainer
from src.learner.algorithms.rejection_sampling import RejectionSampling
from src.orchestration import ActorPoolManager
from ui_rl.simple_data_entry import simple_data_entry_pod_manifest
import queue

# Load config
config = Config.from_yaml("config/qwen25_vl_lora.yaml")

# Create model with LoRA
model = VLMWrapper(
    model_name=config.model.name,
    use_lora=config.model.use_lora,
    lora_target_option=config.model.lora_target_option,
    lora_rank=config.model.lora_rank
)

# Create trainer
trajectory_queue = queue.Queue()
algorithm = RejectionSampling(reward_threshold=0.0)
trainer = Trainer(
    model=model,
    algorithm=algorithm,
    trajectory_queue=trajectory_queue,
    batch_size=config.trainer.batch_size
)

# Create actor pool
actor_pool = ActorPoolManager(
    target_concurrent_actors=config.actor_pool.target_concurrent_actors,
    cluster_host=config.environment.cluster_host,
    pod_manifest_fn=simple_data_entry_pod_manifest,
    model=model,
    trajectory_queue=trajectory_queue,
    task_prompt=config.actor.task_prompt,
    namespace=config.environment.namespace,
    max_steps_per_episode=config.actor.max_steps_per_episode
)

# Start training
actor_pool.start()
trainer.train(num_steps=100, save_every=10)
actor_pool.stop()
```

### Run Single Episode

```python
from src.actor.task_runner import TaskRunner
from src.models.vlm_wrapper import VLMWrapper
from ui_rl.simple_data_entry import simple_data_entry_pod_manifest

vlm = VLMWrapper(model_name="Qwen/Qwen2.5-VL-3B-Instruct")

runner = TaskRunner(
    cluster_host="your-proxy-ip",
    pod_manifest_fn=simple_data_entry_pod_manifest,
    model=vlm,
    task_prompt="Complete the data entry form",
    namespace="default"
)

trajectory = runner.run_episode()
print(f"Steps: {len(trajectory)}, Reward: {trajectory.total_reward()}")
```


## Development

### Code Structure Guidelines

Current structure follows "flat is better than nested":
- Single file per directory until complexity demands splitting
- Clear separation of concerns
- Simple import paths

**When to split a file:**
- Exceeds ~600 lines
- Multiple unrelated classes used independently
- 3+ implementations of same interface

### Running Tests

```bash
# Test single episode with VLM inference (saves trajectories for analysis)
python scripts/test_single_episode.py --cluster-host your-proxy-ip

# Test pod connection
python scripts/test_vm_connection.py --cluster-host your-proxy-ip

# Test full actor-learner loop
python scripts/test_actor_learner.py \
    --cluster-host your-proxy-ip \
    --num-trajectories 5 \
    --num-actors 2
```

See [scripts/TEST_SINGLE_EPISODE.md](scripts/TEST_SINGLE_EPISODE.md) for detailed usage of the single episode test.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes following the existing code style
4. Test your changes
5. Commit with clear messages
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- HuggingFace Transformers for VLM infrastructure
- PEFT library for LoRA implementation
- PyTorch team for the ML framework
- Research papers on LoRA fine-tuning strategies
