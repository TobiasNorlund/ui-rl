from collections import defaultdict
from pathlib import Path
import re
import plotly.graph_objects as go

import subprocess
import logging
import yaml
import shutil
import wandb

from ui_rl.runner import FixedStrategy, NSuccessfulStrategy, run_rollouts
from generate_rollouts import SimpleDataEntryRolloutWorker
from simple_data_entry import SimpleDataEntryTaskSpec, rows_submitted_correctly
from launch_vllm import launch as launch_vllm, get_gpu_count, await_vllm_ready


DEFAULT_VLLM_ARGS = ["--skip-mm-profiling", "--limit-mm-per-prompt", "{\"image\":10,\"video\":0}", "--max-num-seqs", "8", "--max-lora-rank", "64"]

DEFAULT_MOUNTS = ["/tmp/vllm-cache:/root/.cache", "/tmp:/tmp"]


def main(
    max_parallel_rollouts: int,
    max_rollout_steps: int,
    rollout_output_dir: Path,
    checkpoint_output_dir: Path,
    eval_every_n_step: int,
    lora_path: str | None = None,
    vllm_lora_path: str | None = None, 
    mounts: list[str] = DEFAULT_MOUNTS, 
    vllm_args: list[str] = DEFAULT_VLLM_ARGS
):
    """
    Alternates between generating rollouts and training on all available GPUs.
    """

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

    step = 0
    while True:
        # =================================================================
        # STAGE 1: Launch vLLM on all gpus and generate a batch of rollouts
        # =================================================================
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

            # Decide what rollouts to run
            if step % eval_every_n_step == 0:
                # Evalulation batch: Rollout all rows
                strategy = NSuccessfulStrategy(
                    tasks=[SimpleDataEntryTaskSpec(rows=[i]) for i in range(2, 102)],
                    min_successful=1,
                    min_attempts=5,
                    max_attempts=10,
                    is_rollout_successful=rows_submitted_correctly
                )
            else:
                # Training batch: Rollout all rows that didn't have 100% success rate in the latest eval batch
                strategy = NSuccessfulStrategy(
                    tasks=[SimpleDataEntryTaskSpec(rows=[i]) for i in done_rows],
                    min_successful=1,
                    min_attempts=5,
                    max_attempts=10,
                    is_rollout_successful=rows_submitted_correctly
                )

            logging.info(f"Starting generation of rollouts")
            shutil.rmtree(rollout_output_dir)
            rollout_output_dir.mkdir(parents=True, exist_ok=True)
            worker = SimpleDataEntryRolloutWorker(
                model_host="localhost:8000",
                model_name="lora_model" if lora_path is not None else base_model_name,
                max_steps=max_rollout_steps,
                output_dir=rollout_output_dir
            )
            run_rollouts(
                strategy=strategy,
                rollout_worker=worker,
                max_parallel=max_parallel_rollouts,
            )

        # Rollout result
        n_success, n_tot = get_rollout_result(rollout_output_dir)
        success_rate = {row: n_success[row] / n_tot[row] for row in n_tot.keys()}
        fig = go.Figure(data=[go.Bar(x=list(range(2, 102)), y=[success_rate[row] for row in range(2, 102)])])
        if step % eval_every_n_step == 0:
            # Report eval metrics
            wandb.log({
                "eval_success_rate": sum(success_rate.values()) / len/(success_rate),
                "eval_row_success_rate": wandb.Plotly(fig)
            })

            # Only train on rows that doesn't have 100% success rate
            done_rows = [row for row in range(2, 102) if success_rate[row] < 1.0]
            train_rollouts = [
                str(path) 
                for path in Path(rollout_output_dir).glob("row_*.json") \
                    if int(re.search(r"row_(\d+)", path.name).group(1)) not in done_rows
            ]
        else:
            # Report training metrics
            wandb.log({
                "success_rate": sum(success_rate.values()) / len/(success_rate),
                "row_success_rate": wandb.Plotly(fig)
            })

            # Train on all rollouts
            train_rollouts = [str(path) for path in Path(rollout_output_dir).glob("row_*.json")]


        # ======================================
        # STAGE 2: Launch training on this batch
        # ======================================
        with open("/tmp/train_config.yaml", "w") as f:
            yaml.dump({
                "model_name": base_model_name,
                "learning_rate": 1e-5,
                "lora_adapter": str(lora_path),
                "output_dir": str(checkpoint_output_dir),
                "train_rollouts": train_rollouts
            }, f)
        subprocess.run(["accelerate", "launch", "train_rl_step.py", "/tmp/train_config.yaml"])

        # checkpoint_output_dir now contains the newest lora
        lora_path = checkpoint_output_dir
        vllm_lora_path = checkpoint_output_dir



def get_rollout_result(rollout_dir: Path) -> float:
    # Compute success rate for each row
    n_success = defaultdict(lambda: 0)
    n_tot = defaultdict(lambda: 0)
    for rollout in rollout_dir.glob("row_*.json"):
        _, row, res, _ = rollout.name.split("_")
        row = int(row)
        n_tot[row] += 1
        if res == "success":
            n_success[row] += 1
    return n_success, n_tot


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-parallel-rollouts", type=int, default=20)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--vllm-lora-path", default=None)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--mount", nargs="+", default=[])
    args, vllm_args = parser.parse_known_args()

    main(
        max_parallel_rollouts=args.max_parallel_rollouts,
        max_rollout_steps=20,
        rollout_output_dir=Path("/tmp/rollouts"),
        checkpoint_output_dir=Path("/tmp/checkpoint"),
        eval_every_n_step=args.eval_every,
        lora_path=args.lora_path,
        vllm_lora_path=args.vllm_lora_path,
        mounts=DEFAULT_MOUNTS + args.mount,
        vllm_args=DEFAULT_VLLM_ARGS + vllm_args
    )