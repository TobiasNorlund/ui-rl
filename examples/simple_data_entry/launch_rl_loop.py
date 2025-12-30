from pathlib import Path
from rollout_uitars15_docker import SimpleDataEntryRolloutWorker
from launch_vllm import launch, get_gpu_count

import requests
import time
import logging

from ui_rl.runner import FixedStrategy, NSuccessfulStrategy, run_rollouts
from task import SimpleDataEntryTaskSpec


DEFAULT_VLLM_ARGS = ["--skip-mm-profiling", "--limit-mm-per-prompt", "{\"image\":10,\"video\":0}", "--max-num-seqs", "8", "--max-lora-rank", "64"]

DEFAULT_MOUNTS = ["/tmp/vllm-cache:/root/.cache"]


def main(
    max_parallel_rollouts: int,
    max_rollout_steps: int,
    rollout_output_dir: Path,
    checkpoint_output_dir: Path,
    lora_path: str | None = None, 
    mounts: list[str] = DEFAULT_MOUNTS, 
    vllm_args: list[str] = DEFAULT_VLLM_ARGS
):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(),
        ],
        force=True
    )
    # Disable verbose logging from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if lora_path is not None:
        vllm_args += ["--enable-lora", "--lora-modules", f"lora_model={lora_path}"]

    base_model_name = "ByteDance-Seed/UI-TARS-1.5-7B"

    # Launch vLLM on all gpus
    gpus = range(get_gpu_count())
    with launch(
        gpus=gpus, 
        model_name=base_model_name,
        mounts=mounts,
        vllm_args=vllm_args
    ):
        # Wait until ready
        logging.info("Starting vLLM...")
        await_vllm_ready()

        logging.info(f"Starting generation of rollouts using {max_parallel_rollouts} parallel workers")
        strategy = FixedStrategy(
            tasks=[SimpleDataEntryTaskSpec(rows=[2])] * 20,
            rerun_on_error=True
        )
        generate_rollout_batch(
            model_name="lora_model" if lora_path is not None else base_model_name,
            strategy=strategy,
            max_parallel=max_parallel_rollouts,
            max_steps=max_rollout_steps,
            output_dir=rollout_output_dir,
        )

    # Launch training on this batch
    print("Batch ready!")

    # Loop

    pass


def await_vllm_ready():
    while True:
        try:
            resp = requests.get("http://localhost:8000/health")
            if resp.status_code == 200:
                break
            else:
                time.sleep(5)
        except:
            time.sleep(5)


def generate_rollout_batch(
    model_name: str,
    strategy: FixedStrategy | NSuccessfulStrategy,
    max_parallel: int,
    max_steps: int,
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create rollout worker
    worker = SimpleDataEntryRolloutWorker(
        model_host="localhost:8000",
        model_name=model_name,
        max_steps=max_steps,
        output_dir=output_dir
    )

    # Run rollouts
    run_rollouts(
        strategy=strategy,
        rollout_worker=worker,
        max_parallel=max_parallel,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-parallel-rollouts", type=int, default=20)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--mount", nargs="+")
    args, vllm_args = parser.parse_known_args()

    main(
        max_parallel_rollouts=args.max_parallel_rollouts,
        max_rollout_steps=20,
        rollout_output_dir=Path("/tmp/rollouts"),
        checkpoint_output_dir=Path("/tmp/checkpoint"),
        lora_path=args.lora_path,
        mounts=DEFAULT_MOUNTS + args.mount,
        vllm_args=DEFAULT_VLLM_ARGS + vllm_args
    )