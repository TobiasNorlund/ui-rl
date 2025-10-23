# Test Single Episode Script

This script tests VLM inference on a single episode without any training. Perfect for:
- Testing your VLM model's action generation
- Debugging pod connectivity
- Analyzing trajectory data before training
- Validating your reward function

## Quick Start

```bash
# Basic test with default settings
python scripts/test_single_episode.py --cluster-host 34.123.45.67

# Custom model and more steps
python scripts/test_single_episode.py \
    --cluster-host 34.123.45.67 \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --max-steps 20

# Test with LoRA checkpoint
python scripts/test_single_episode.py \
    --cluster-host 34.123.45.67 \
    --use-lora \
    --lora-checkpoint ./checkpoints/step_100
```

## Output Files

The script saves trajectory data in multiple formats for analysis:

```
test_episodes/
├── metadata_20251019_145530_a1b2c3d4.json          # Episode metadata
├── actions_20251019_145530_a1b2c3d4.json           # Action sequence (easy to read)
├── trajectory_20251019_145530_a1b2c3d4.npz         # Full trajectory archive
└── screenshots_20251019_145530_a1b2c3d4/           # Screenshots from episode
    ├── step_000.png
    ├── step_001.png
    └── ...
```

### 1. Metadata JSON
Contains episode-level information:
```json
{
  "task_id": "2025-10-19T14:55:30.123456",
  "session_id": "a1b2c3d4",
  "pod_name": "session-a1b2c3d4",
  "success": true,
  "episode_length": 10,
  "total_reward": 5.0,
  "complete": true,
  "termination_reason": "success"
}
```

### 2. Actions JSON
Detailed step-by-step action log:
```json
{
  "session_id": "a1b2c3d4",
  "timestamp": "20251019_145530",
  "total_steps": 10,
  "total_reward": 5.0,
  "actions": [
    {
      "step": 0,
      "generated_text": "{\"action_type\": \"left_click\", \"x\": 150, \"y\": 200}",
      "reward": 0.0,
      "prompt": "Complete the data entry task"
    },
    ...
  ]
}
```

### 3. Screenshots
PNG images of each observation state, named sequentially:
- `step_000.png` - Initial state
- `step_001.png` - After first action
- `step_002.png` - After second action
- etc.

### 4. Trajectory Archive (NPZ)
Numpy compressed archive with all data:
- `observations` - Screenshot arrays
- `generated_texts` - VLM outputs
- `rewards` - Reward per step
- `prompts` - Task prompts
- `metadata` - Metadata JSON string

## Command Line Options

### Required
- `--cluster-host` - Kubernetes cluster proxy server IP/hostname

### VLM Configuration
- `--model` - HuggingFace model name (default: Qwen/Qwen2.5-VL-3B-Instruct)
- `--device` - Device for inference: cuda/cpu (default: auto-detect)
- `--use-lora` - Enable LoRA fine-tuning
- `--lora-checkpoint` - Path to LoRA checkpoint directory

### Episode Configuration
- `--max-steps` - Maximum steps per episode (default: 10)
- `--task-prompt` - Task description for VLM (default: "Complete the data entry task")
- `--session-type` - Session type (default: simple_data_entry)
- `--namespace` - Kubernetes namespace (default: default)
- `--session-timeout` - Pod startup timeout in seconds (default: 300)

### Output Configuration
- `--output-dir` - Directory to save files (default: ./test_episodes)
- `--verbose`, `-v` - Enable DEBUG logging

## Example Usage Scenarios

### 1. Quick Connectivity Test
```bash
# Just verify everything works
python scripts/test_single_episode.py --cluster-host 34.123.45.67 --max-steps 3
```

### 2. Analyze VLM Behavior
```bash
# Run longer episode and analyze actions
python scripts/test_single_episode.py \
    --cluster-host 34.123.45.67 \
    --max-steps 20 \
    --output-dir ./analysis/baseline_model \
    --verbose
```

### 3. Test Fine-tuned Model
```bash
# Test your LoRA checkpoint
python scripts/test_single_episode.py \
    --cluster-host 34.123.45.67 \
    --use-lora \
    --lora-checkpoint ./checkpoints/step_500 \
    --output-dir ./analysis/checkpoint_500
```

### 4. Compare Different Models
```bash
# Run same episode with different models
python scripts/test_single_episode.py \
    --cluster-host 34.123.45.67 \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --output-dir ./comparison/3B

python scripts/test_single_episode.py \
    --cluster-host 34.123.45.67 \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output-dir ./comparison/7B
```

## Analyzing Results

### Python Analysis
```python
import json
import numpy as np
from PIL import Image

# Load action sequence
with open('test_episodes/actions_20251019_145530_a1b2c3d4.json') as f:
    data = json.load(f)

# Print action sequence
for action in data['actions']:
    print(f"Step {action['step']}: {action['generated_text']}")
    print(f"  Reward: {action['reward']}")

# Load full trajectory
traj = np.load('test_episodes/trajectory_20251019_145530_a1b2c3d4.npz')
observations = traj['observations']
rewards = traj['rewards']

# Analyze rewards
print(f"Total reward: {rewards.sum()}")
print(f"Average reward: {rewards.mean()}")
print(f"Reward distribution: {np.histogram(rewards)}")

# View screenshots
for i in range(len(observations)):
    img = Image.fromarray(observations[i])
    img.save(f'analysis/step_{i}.png')
```

### Command Line Analysis
```bash
# Count total steps
jq '.total_steps' test_episodes/actions_*.json

# Sum rewards
jq '.total_reward' test_episodes/actions_*.json

# Extract all action types
jq '.actions[].generated_text' test_episodes/actions_*.json | grep action_type

# View screenshots
open test_episodes/screenshots_*/step_*.png
```

## Troubleshooting

### Pod Creation Fails
- Check Kubernetes credentials: `kubectl get pods`
- Verify namespace exists: `kubectl get namespaces`
- Check proxy server is running: `kubectl get services`

### VLM Loading Fails
- Verify model name is correct on HuggingFace
- Check GPU availability: `nvidia-smi`
- Ensure enough memory for model

### Episode Hangs
- Increase `--session-timeout` if pod takes long to start
- Check pod logs: `kubectl logs session-<session_id>`
- Verify proxy server is accessible: `curl http://<cluster-host>:8000/health`

### No Screenshots Saved
- Check output directory permissions
- Verify disk space available
- Check logs for PIL/image saving errors
