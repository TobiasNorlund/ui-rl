from pathlib import Path
import re
import plotly.graph_objects as go

import subprocess
import logging
import yaml
import shutil
import random
import wandb

from ui_rl.runner import FixedStrategy, run_rollouts
from generate_rollouts import SimpleDataEntryRolloutWorker
from simple_data_entry import SimpleDataEntryTaskSpec
from launch_vllm import launch as launch_vllm, get_gpu_count, await_vllm_ready, DEFAULT_VLLM_ARGS, DEFAULT_VLLM_MOUNTS
from utils import get_rollout_result


def main(
    max_parallel_rollouts: int,
    max_rollout_steps: int,
    rollout_output_dir: Path,
    checkpoint_output_dir: Path,
    lora_path: str | None = None,
    mounts: list[str] = DEFAULT_VLLM_MOUNTS, 
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
    # Disable verbose logging from httpx and rollouts
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("ui_rl.agent").setLevel(logging.WARNING)

    if lora_path is not None:
        vllm_args += ["--enable-lora", "--lora-modules", f"lora_model={lora_path}"]

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

            logging.info(f"Generating rollout batch")
            strategy = FixedStrategy([SimpleDataEntryTaskSpec(rows=[i]) for i in list(range(2, 7)) * 3])
            
            if rollout_output_dir.exists():
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
        fig = go.Figure(data=[go.Bar(x=list(success_rate.keys()), y=[success_rate[row] for row in success_rate.keys()])])
        # Report training metrics
        wandb.log({
            "success_rate": sum(success_rate.values()) / len(success_rate),
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

        step += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-parallel-rollouts", type=int, default=15)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--mount", nargs="+", default=[])
    args, vllm_args = parser.parse_known_args()

    main(
        max_parallel_rollouts=args.max_parallel_rollouts,
        max_rollout_steps=20,
        rollout_output_dir=Path("/tmp/rollouts"),
        checkpoint_output_dir=args.checkpoint_dir,
        lora_path=args.checkpoint_dir,
        mounts=DEFAULT_VLLM_MOUNTS + args.mount,
        vllm_args=DEFAULT_VLLM_ARGS + vllm_args
    )