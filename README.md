<div align="center">
    <img alt="ui-rl: Reinforcement Fine-tuning for Computer Use models" src="assets/uirl.svg">
</div>

---

[![Build](https://github.com/TobiasNorlund/ui-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/TobiasNorlund/ui-rl/actions/workflows/ci.yml) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

`ui-rl` is a **framework for fine-tuning Computer Use agent models**.
It provides utilities for scalable training on verifiable UI tasks to improve model reliability on targeted domains and tasks.
This allows you to focus more on building verifiable tasks and optimizing model performance, and less on boilerplate code such as agent loops, rollout generations, torch data loading etc.

## The Why

Computer Use deployments require very robust model performance. You typically want (very close to) 100% success rate, and never any fatal mistakes. Few of today's models offer that reliability for arbitrary tasks.
The best way to achieve such high reliability is to do task-specific fine-tuning. 

However, performing such fine-tuning is complex. A typical pipeline consists of starting from an open source model, performing some Supervised Fine-Tuning (SFT) until a decent success rate is achieved, and finally use Reinforcement Learning from Verified Reward (RLVR) for the last-mile performance boost.
Getting all this up-and-running requires significant setup, and is where `ui-rl` aims to simplify.


## Key features
 - Perform CUA rollouts at scale in custom task environments, using state-of-the-art open models
 - Rollout serialization/deserialization out of the box
 - CUA specific Data Augmentation techniques for SFT
 - Training framework agnostic — pick your favorite torch-compatible trainer
 - Pre-built torch `Dataset` implementations for rollouts — start training in a just few lines


## Installation

```bash
pip install ui-rl
```


## Use Case: Generate a rollout for a custom UI task

1. Start by building a containerized task environment. For an example, see [examples/simple_data_entry/env](examples/simple_data_entry/env)
2. Implement a `TaskSpec` that specifies the task prompt, and how to run it (e.g. in a local docker container).  

```python
from ui_rl.task import TaskSpec
from ui_rl.runtime.docker import DockerSessionRuntime

class SimpleDataEntryTaskSpec(TaskSpec[DockerSessionRuntime]):
    def __init__(self, rows: List[int]):
        self.rows = rows

    def get_task_instruction(self):
        return f"""Your task is to submit data from a spreadsheet (seen on the left) into a form (seen on the right). Specifically, the following rows (as numbered in the left margin) from the spreadsheet are to be submitted: {", ".join(str(i) for i in self.rows)}.
Note: You may need to scroll to make the row visible in the sheet.
The form has to be submitted separately for each row. When the form has been submitted, return to the form to submit the next row. 
Submit a row by selecting each cell individually, copy its content by sending keys "ctrl+c", select the target form text input and paste using "ctrl+v".
Finally, submit the form and continue with the next row. Only finish when all rows have been successfully submitted"""

    def create_session(self, runtime: DockerSessionRuntime) -> str:
        return runtime.create_session(
            image="ui-rl/simple-data-entry:latest",
        )

```

3. Start vLLM to use as inference engine. We will use the [UITARS 1.5 7B](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B) model.

**Note:** You probably need a GPU with at least 40GB of VRAM for running this model

```bash
# TODO
docker run ...
```

4. Instantiate the TaskSpec, and create a rollout object to manage and record the CUA model calls:

```python
import httpx
from ui_rl.models.uitars15.rollout import UITARS15_Rollout

# Create shared httpx client for communicating with vLLM and task environment
httpx_client = httpx.Client()

# Create a task spec for submitting row no 2 from the spreadsheet
task_spec = SimpleDataEntryTaskSpec(rows=[2])

# Create rollout object
rollout = UITARS15_Rollout(
    task_spec=task_spec,
    model_host="localhost:8000",
    model_name="ByteDance-Seed/UI-TARS-1.5-7B",
    httpx_client=httpx_client,
    max_images_in_context=10,
    max_tokens=200,
    temperature=0.1
)
```

5. Create a `SessionRuntime` that manages the rollout session containers. We'll run them locally using `DockerSessionRuntime` (but Kubernetes is supported via `KubernetesSessionRuntime`).

```python
runtime = DockerSessionRuntime(httpx_client=httpx_client)
```

6. Run and serialize the rollout

```python
from ui_rl.agent import run_cua_rollout

try:
    run_cua_rollout(
        task_spec=task_spec,
        rollout=rollout,
        runtime=runtime,
        max_steps=20,
    )
    print("Rollout finished successfully!")
    print("Progress:", rollout.progress)
    rollout.save("rollout_1.json")
except Exception as e:
    print(f"Error when generating rollout: {e}")
```

**Note:** A successful rollout has a "progress", a dict specifying task-specific progress data. In our example, it contains:

```json
{
    "submitted_row_indices": [0],
    "num_incorrect_submissions": 0
}
```

This can later be used to compute a reward during RL training. 


## Use Case: Train on rollouts

Saved rollouts can be used directly for training. 
Perhaps the simplest case is when doing [Rejection Sampling](https://rlhfbook.com/c/10-rejection-sampling).
`ui-rl` comes with built in torch `Dataset`s for directly loading rollouts into trainable token sequences:

```python
from transformers import AutoProcessor
from ui_rl.models.uitars15.dataset import UITARS15_RolloutDataset

processor = AutoProcessor.from_pretrained("ByteDance-Seed/UI-TARS-1.5-7B")
ds = UITARS15_RolloutDataset(
    processor=processor,
    rollout_path="rollout_1.json"
)

print(ds[0])
```

Which will print something like:

```
{'input_ids': tensor([151644,   8948,    198,  ...,    272,    863, 151645]),
 'attention_mask': tensor([1, 1, 1,  ..., 1, 1, 1]),
 'labels': tensor([  -100,   -100,   -100,  ...,    272,    863, 151645]),
 'reward': tensor(0.),
 'pixel_values': tensor([[1.4340, 1.4340, 1.4340,  ..., 2.1459, 2.1459, 2.1459],
         [1.4340, 1.4340, 1.4340,  ..., 2.1459, 2.1459, 2.1459],
         [1.4340, 1.4340, 1.4340,  ..., 2.1459, 2.1459, 2.1459],
         ...,
         [1.8865, 1.8865, 1.8865,  ..., 2.1032, 2.1032, 2.1032],
         [1.7114, 1.7114, 1.7114,  ..., 2.0464, 2.0464, 2.1032],
         [1.8865, 1.8865, 1.9157,  ..., 2.1032, 2.1032, 2.1032]]),
 'image_grid_thw': tensor([[ 1, 58, 92],
         [ 1, 58, 92],
         [ 1, 58, 92],
         [ 1, 58, 92],
         [ 1, 58, 92],
         [ 1, 58, 92],
         [ 1, 58, 92],
         [ 1, 58, 92],
         [ 1, 58, 92],
         [ 1, 58, 92]])}
```

Here, there are some things worth noting:

 - The dataset groups together LLM completions that extends the same sequence. If all rollout turns keeps extending the same token sequence, then `len(ds)` will be 1.
 - The `labels` are constructed such that only generated tokens are trained.

Finally, a `reward_fn` can be provided in the constructor to support RL training, see code for more details:

```python
def reward_fn(rollout: UITARS15_RolloutDataset.Rollout) -> float:
    if rollout.task_spec["rows"] == rollout.progress["submitted_row_indices"]:
        return 1.0
    else:
        return -1.0

ds = UITARS15_RolloutDataset(
    processor=processor,
    rollout_path="rollout_1.json",
    reward_fn=reward_fn
)
```