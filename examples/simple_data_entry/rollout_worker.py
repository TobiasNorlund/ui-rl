from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
import signal
import requests
from multiprocessing import Queue, Process
from pathlib import Path

from ui_rl.runner import RolloutStrategy, run_rollouts
from simple_data_entry import SimpleDataEntryRolloutWorker
from launch_vllm import launch as launch_vllm, await_vllm_ready


@dataclass
class RolloutBatchRequest:
    output_dir: Path
    rollout_strategy: RolloutStrategy
    max_steps: int
    model_name: str | None
    lora_name: str | None = None
    lora_path: str | None = None


def run_rollout_worker(
    gpus: list[int], 
    model_name: str, 
    max_parallel_rollouts: int,
    task_queue: Queue,
    vllm_mounts: list[str],
    vllm_args: list[str]
):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("ui_rl.agent").setLevel(logging.WARNING)

    with launch_vllm(
        gpus=gpus, 
        model_name=model_name,
        mounts=vllm_mounts,
        vllm_args=vllm_args,
        detach=True
    ):
        # Wait until ready
        logging.info("Starting vLLM...")
        await_vllm_ready()

        try:
            while (rollout_batch_request := task_queue.get()) is not None:
                logging.info("Starting to process rollout request")

                # Create requested dir
                rollout_batch_request.output_dir.mkdir(parents=True, exist_ok=True)

                # Optionally load lora
                if rollout_batch_request.lora_name is not None and rollout_batch_request.lora_path is not None:
                    logging.info(f"Loading lora: {rollout_batch_request.lora_name} from path {rollout_batch_request.lora_path}")
                    resp = requests.post(
                        "http://localhost:8000/v1/load_lora_adapter", 
                        json={
                            "lora_name": rollout_batch_request.lora_name,
                            "lora_path": str(rollout_batch_request.lora_path)
                        }
                    )
                    resp.raise_for_status()

                worker = SimpleDataEntryRolloutWorker(
                    model_host="localhost:8000",
                    model_name=rollout_batch_request.lora_name if rollout_batch_request.lora_path is not None else rollout_batch_request.model_name,
                    max_steps=rollout_batch_request.max_steps,
                    output_dir=rollout_batch_request.output_dir
                )
                logging.info(f"Running rollouts with parallelism: {max_parallel_rollouts}")
                run_rollouts(
                    strategy=rollout_batch_request.rollout_strategy,
                    rollout_worker=worker,
                    max_parallel=max_parallel_rollouts,
                )

                Path(rollout_batch_request.output_dir, ".done").touch() 

        except KeyboardInterrupt:
            logging.info("Stopping rollout worker...")


@contextmanager
def rollout_worker(
    gpus: list[int], 
    model_name: str, 
    max_parallel_rollouts: int,
    vllm_mounts: list[str],
    vllm_args: list[str]
):
    task_queue = Queue()
    proc = Process(target=run_rollout_worker, kwargs={
        "gpus": gpus,
        "model_name": model_name,
        "max_parallel_rollouts": max_parallel_rollouts,
        "task_queue": task_queue,
        "vllm_mounts": vllm_mounts,
        "vllm_args": vllm_args
    })
    proc.start()

    yield task_queue

    os.kill(proc.pid, signal.SIGINT)
    proc.join()