from collections import defaultdict
from pathlib import Path
from rollout_uitars15_docker import SimpleDataEntryRolloutWorker
from launch_vllm import launch as launch_vllm, get_gpu_count

import requests
import subprocess
import time
import logging
import yaml
import shutil
import wandb

from ui_rl.runner import FixedStrategy, NSuccessfulStrategy, run_rollouts
from task import SimpleDataEntryTaskSpec


DEFAULT_VLLM_ARGS = ["--skip-mm-profiling", "--limit-mm-per-prompt", "{\"image\":10,\"video\":0}", "--max-num-seqs", "8", "--max-lora-rank", "64"]

DEFAULT_MOUNTS = ["/tmp/vllm-cache:/root/.cache", "/tmp:/tmp"]


def main(
    max_parallel_rollouts: int,
    max_rollout_steps: int,
    rollout_output_dir: Path,
    checkpoint_output_dir: Path,
    lora_path: str | None = None,
    vllm_lora_path: str | None = None, 
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

    if vllm_lora_path is not None:
        vllm_args += ["--enable-lora", "--lora-modules", f"lora_model={vllm_lora_path}"]

    base_model_name = "ByteDance-Seed/UI-TARS-1.5-7B"

    wandb.init(project="ui-rl", group="rl")

    while True:
        # Launch vLLM on all gpus and generate a batch of rollouts
        gpus = range(get_gpu_count())
        with launch_vllm(
            gpus=gpus, 
            model_name=base_model_name,
            mounts=mounts,
            vllm_args=vllm_args,
            detach=True
        ):
            # Wait until ready
            logging.info("Starting vLLM...")
            await_vllm_ready()

            logging.info(f"Starting generation of rollouts using {max_parallel_rollouts} parallel workers")
            strategy = FixedStrategy(
                tasks=[SimpleDataEntryTaskSpec(rows=[2])] * 20,
                rerun_on_error=True
            )
            shutil.rmtree(rollout_output_dir)
            generate_rollout_batch(
                model_name="lora_model" if lora_path is not None else base_model_name,
                strategy=strategy,
                max_parallel=max_parallel_rollouts,
                max_steps=max_rollout_steps,
                output_dir=rollout_output_dir,
            )

        success_rate = get_success_rate(rollout_output_dir)
        wandb.log({"success_rate": success_rate})

        # Launch training on this batch
        with open("/tmp/train_config.yaml", "w") as f:
            yaml.dump({
                "model_name": base_model_name,
                "learning_rate": 1e-5,
                "lora_adapter": str(lora_path),
                "output_dir": str(checkpoint_output_dir),
                "train_rollouts": [
                    str(path) for path in Path(rollout_output_dir).glob("row_*.json")
                ]
            }, f)
        subprocess.run(["accelerate", "launch", "train_rl.py", "/tmp/train_config.yaml"])

        # checkpoint_output_dir now contains the newest lora
        lora_path = checkpoint_output_dir
        vllm_lora_path = checkpoint_output_dir


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
    worker = SimpleDataEntryRolloutWorker(
        model_host="localhost:8000",
        model_name=model_name,
        max_steps=max_steps,
        output_dir=output_dir
    )
    run_rollouts(
        strategy=strategy,
        rollout_worker=worker,
        max_parallel=max_parallel,
    )


def get_success_rate(rollout_dir: Path) -> float:
    # Compute success rate for each row
    n_success = defaultdict(lambda: 0)
    n_tot = defaultdict(lambda: 0)
    for rollout in rollout_dir.glob("row_*.json"):
        _, row, res, _ = rollout.name.split("_")
        row = int(row)
        n_tot[row] += 1
        if res == "success":
            n_success[row] += 1
    return sum(n_success[row] / n_tot[row] for row in n_tot.keys()) / len(n_tot)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-parallel-rollouts", type=int, default=20)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--vllm-lora-path", default=None)
    parser.add_argument("--mount", nargs="+", default=[])
    args, vllm_args = parser.parse_known_args()

    main(
        max_parallel_rollouts=args.max_parallel_rollouts,
        max_rollout_steps=20,
        rollout_output_dir=Path("/tmp/rollouts"),
        checkpoint_output_dir=Path("/tmp/checkpoint"),
        lora_path=args.lora_path,
        vllm_lora_path=args.vllm_lora_path,
        mounts=DEFAULT_MOUNTS + args.mount,
        vllm_args=DEFAULT_VLLM_ARGS + vllm_args
    )