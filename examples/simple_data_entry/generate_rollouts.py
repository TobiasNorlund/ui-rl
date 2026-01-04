from collections import defaultdict
import logging
import wandb
from datetime import datetime
from pathlib import Path

from ui_rl import FixedStrategy, NSuccessfulStrategy, run_rollouts

from simple_data_entry import SimpleDataEntryRolloutWorker, parse_strategy


def main(
    vllm_host: str,
    model_name: str,
    strategy: FixedStrategy | NSuccessfulStrategy,
    max_parallel: int,
    max_steps: int,
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "output.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w')
        ],
        force=True
    )
    # Disable verbose logging from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.info(f"Starting generation of rollouts from model '{model_name}', using {max_parallel} parallel workers")
    logging.info(f"Logs will be saved to: {output_dir}")

    wandb.init(project="ui-rl", config={
        "model_name": model_name,
        "max_parallel": max_parallel,
        "max_steps": max_steps,
        "output_dir": output_dir
    })

    # Create worker
    worker = SimpleDataEntryRolloutWorker(
        model_host=vllm_host,
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

    # Compute success rate for each row and log to w&b
    n_success = defaultdict(lambda: 0)
    n_tot = defaultdict(lambda: 0)
    for rollout in output_dir.glob("row_*.json"):
        _, row, res, _ = rollout.name.split("_")
        row = int(row)
        n_tot[row] += 1
        if res == "success":
            n_success[row] += 1
    
    table = wandb.Table(
        data=[[row, n_success[row] / n_tot[row]] for row in n_tot.keys()],
        columns=["row", "success rate"]
    )
    wandb.log({"result": wandb.plot.bar(table, "row", "success rate", title="Success Rate")})
    wandb.finish()
            

if __name__ == "__main__":
    import argparse
    
    class WideHelpFormatter(argparse.RawDescriptionHelpFormatter):
        def __init__(self, prog):
            super().__init__(prog, width=120, max_help_position=50)
    
    parser = argparse.ArgumentParser(
        description="Generate a batch of SimpleDataEntry rollouts using Docker runtime and a vLLM model host",
        formatter_class=WideHelpFormatter
    )
    parser.add_argument("--vllm-host", required=True, help="vLLM host")
    parser.add_argument("--model-name", required=True, help="The model name")
    parser.add_argument("--strategy", required=True, help="Rollout strategy to use")
    parser.add_argument("--max-parallel", type=int, default=1, help="Maximum number of parallel rollouts")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum steps per rollout")
    parser.add_argument("--output-dir", type=Path, help="Dir to save rollouts and logs")
    args = parser.parse_args()

    if args.output_dir is None:
        repo_root = Path(__file__).parent.parent.parent
        output_dir = repo_root / "data" / "rollouts" / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_dir = args.output_dir

    rollout_strategy = parse_strategy(args.strategy)

    try:
        main(
            vllm_host=args.vllm_host,
            model_name=args.model_name,
            strategy=rollout_strategy,
            max_parallel=args.max_parallel,
            max_steps=args.max_steps,
            output_dir=output_dir
        )
    except KeyboardInterrupt:
        logging.info("Script interrupted by user (Ctrl+C)")
        exit(0)
