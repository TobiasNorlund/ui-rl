<div align="center">
    <img alt="ui-rl: Reinforcement Fine-tuning for Computer Use models" src="assets/uirl.svg">
</div>

---

[![Build](https://github.com/TobiasNorlund/ui-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/TobiasNorlund/ui-rl/actions/workflows/ci.yml) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

`ui-rl` is a **framework for fine-tuning Computer Use agent models**.
It provides utilities for scalable training on verifiable UI tasks to improve model reliability on targeted domains and tasks.
This allows you to focus more on building verifiable tasks and optimizing model performance, and less on boilerplate code such as agent loops, rollout generations, torch data loading etc.

## The Why

Computer Use deployments require very robust model performance. You typically want (very close to) 100% success rate on your specific task(s), and never any fatal mistakes. Even the best models seldom offer that reliability for arbitrary tasks.
The best way to achieve such high reliability is to do task-specific fine-tuning. 

However, performing such fine-tuning is complex. A typical pipeline consists of starting from an open source model, performing some Supervised Fine-Tuning (SFT) until a decent success rate is achieved, and finally use Reinforcement Learning from Verified Reward (RLVR) for the last-mile performance boost.

Getting all this up-and-running requires significant setup, and is where `ui-rl` aims to simplify"."


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

## How to use:

We exemplify usage of `ui-rl` via common use cases:


### Use Case: Generate a rollout for a custom UI task

1. Start by building a containerized task environment. For an example, see [examples/simple_data_entry/env](examples/simple_data_entry/env) or [this blog post](https://medium.com/@tobias.norlund/does-computer-use-actually-make-sense-60f0449e3770) for a thorough walkthrough
2. Implement a `TaskSpec` that specifies the task prompt, and how to run it using a so called `SessionRuntime` (local docker container in this case).  

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

3. Start vLLM to use as inference engine, or use a cloud hosting. We will use the [UITARS 1.5 7B](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B) model.

**Note:** You probably need a GPU with at least 40GB of VRAM for running this model

```bash
docker run -it --rm --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 vllm/vllm-openai:latest "ByteDance-Seed/UI-TARS-1.5-7B" --limit-mm-per-prompt '{"image":10,"video":0}'
```

4. Instantiate the TaskSpec, and create a rollout object to manage and record the CUA model calls:

```python
import httpx
from ui_rl.models.uitars15.rollout import UITARS15_Rollout

# Create shared httpx client for communicating with vLLM and task environment
httpx_client = httpx.Client()

# Create a task spec for submitting row no 2 (the first data row) from the spreadsheet
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

**Note:** A successful rollout has a "progress", a dict specifying task-specific progress data. For our example task, it contains:

```json
{
    "submitted_row_indices": [0],
    "num_incorrect_submissions": 0
}
```

This can later be used to compute a reward during RL training. 


### Use Case: Train on rollouts

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

 - The dataset iterates over _sequences_, all unique token sequences used by the agent throughout the rollout. This means that LLM completions that extends the same sequence is represented by a single sequence. If all LLM completions keep extending the same token sequence, then `len(ds)` will be 1.
 - The `labels` are constructed such that only generated tokens are trained.

Finally, a `reward_fn` can be provided in the constructor to support RL training, see code for more details:

```python
def reward_fn(rollout: UITARS15_RolloutDataset.Rollout) -> float:
    # Minor detail: The data starts on row 2 in the spreadsheet in this example
    if [row - 2 for row in rollout.task_spec["rows"]] == rollout.progress["submitted_row_indices"]:
        return 1.0
    else:
        return -1.0

ds = UITARS15_RolloutDataset(
    processor=processor,
    rollout_path="rollout_1.json",
    reward_fn=reward_fn
)
```

The reward is returned when iterating over the dataset.


### Use case: Generate multiple rollouts in parallel

`ui-rl` supports utilities for generating rollouts in parallel via `run_rollouts(...)`. It takes three arguments:

 - `strategy: ui_rl.RolloutStrategy`: Defines what task specs to run and in what order
 - `rollout_worker: ui_rl.RolloutWorker`: Defines how to run a rollout (e.g. creating a rollout object followed by `run_cua_agent(...)`, like above). 
 - `max_parallel: int`: How many rollouts to run in parallel.

Under the hood, `run_rollouts()` uses a `ProcessPoolExecutor` to run the rollout workers concurrently. We start by creating a `RolloutWorker` for our task and model: 

```python
import logging
from ui_rl import RolloutWorker, RolloutResult

class SimpleDataEntryRolloutWorker(RolloutWorker):
    _runtime = None

    @classmethod
    def init_worker(cls, log_queue: Queue):
        # init_worker is run at init for each process pool worker

        # Configures logging for multiprocessing, and creates a reusable 
        # httpx client for each process worker at cls._httpx_client
        super().init_worker(log_queue)

        # Initialize a docker runtime for this worker
        cls._runtime = DockerSessionRuntime(httpx_client=cls._httpx_client, session_timeout=60)

    def run(self, rollout_id: int, task_spec: SimpleDataEntryTaskSpec) -> RolloutResult:
        # `run` takes a rollout_id and task_spec, executes a rollout and returns a `RolloutResult` 
        # It is executed for each task_spec returned by the rollout strategy

        rollout = UITARS15_Rollout(
            task_spec=task_spec,
            model_host="localhost:8000",
            model_name="ByteDance-Seed/UI-TARS-1.5-7B",
            httpx_client=self._httpx_client,
            max_images_in_context=10,
            max_tokens=200,
            temperature=0.1
        )
        logging.info(f"Starting rollout for task: {task_spec}")
        try:
            run_cua_rollout(
                task_spec=task_spec,
                rollout=rollout,
                runtime=self._runtime,
                max_steps=20,
            )
            logging.info(f"Rollout {rollout_id} completed")

            # Save rollout
            result = RolloutResult(rollout_id, task_spec, rollout.progress, None)
            rollout.save(f"rollout_{rollout_id:03d}.json")
            return result

        except Exception as e:
            logging.error(f"Rollout {rollout_id} was stopped due to an error: {e}")
            return RolloutResult(rollout_id, task_spec, rollout.progress, e)
```

As we can see, the rollout worker simply wraps the rollout logic we created earlier. Now we are ready to run:

```python
from ui_rl import run_rollouts, FixedStrategy

# Rollout strategy for running our SimpleDataEntry task once for rows 2, 3 and 4
strategy = FixedStrategy(
    tasks=[SimpleDataEntryTaskSpec(rows=[i]) for i in range(2, 5)]
)

# Create rollout worker
worker = SimpleDataEntryRolloutWorker()

# Run
run_rollouts(
    strategy=strategy,
    rollout_worker=worker,
    max_parallel=3
)
```