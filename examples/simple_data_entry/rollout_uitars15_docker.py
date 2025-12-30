from collections import defaultdict
import re
import logging
import wandb
from datetime import datetime
from pathlib import Path
from queue import Queue

from ui_rl import RolloutResult, RolloutStrategy, FixedStrategy, NSuccessfulStrategy, run_rollouts, RolloutWorker
from ui_rl.agent import run_cua_rollout
from ui_rl.models.uitars15.rollout import UITARS15_Rollout
from ui_rl.runtime.docker import DockerSessionRuntime

from task import SimpleDataEntryTaskSpec


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


class SimpleDataEntryRolloutWorker(RolloutWorker):
    _runtime = None

    def __init__(self, model_host: str, model_name: str, max_steps: int, output_dir: Path):
        self._model_host = model_host
        self._model_name = model_name
        self._max_steps = max_steps
        self._output_dir = output_dir

    @classmethod
    def init_worker(cls, log_queue: Queue):
        super().init_worker(log_queue)

        # Initialize docker runtime for this worker
        cls._runtime = DockerSessionRuntime(httpx_client=cls._httpx_client, session_timeout=60)

    def run(self, rollout_id: int, task_spec: SimpleDataEntryTaskSpec):
        rollout = UITARS15_Rollout(
            task_spec=task_spec,
            model_host=self._model_host,
            model_name=self._model_name,
            httpx_client=self._httpx_client,
            max_images_in_context=10,
            max_tokens=200,
            temperature=0.1
        )
        logging.info(f"Starting UITARS rollout for task: {task_spec}")
        try:
            run_cua_rollout(
                task_spec=task_spec,
                rollout=rollout,
                runtime=self._runtime,
                max_steps=self._max_steps,
            )
            logging.info(f"Rollout {rollout_id} completed")

            # Save rollout
            result = RolloutResult(rollout_id, task_spec, rollout.progress, None)
            file_name = f"row_{task_spec.rows[0]:03d}_success_{rollout_id:04d}.json" if rows_submitted_correctly(result) else f"row_{task_spec.rows[0]:03d}_fail_{rollout_id:04d}.json"
            rollout.save(self._output_dir / file_name)
            return result

        except Exception as e:
            logging.error(f"Rollout {rollout_id} was stopped due to an error: {e}")
            return RolloutResult(rollout_id, task_spec, rollout.progress, e)


def rows_submitted_correctly(result: RolloutResult) -> bool:
    """
    Checks if all the instructed spreadsheet rows were actually submitted
    Note: In the spreadsheet, the first data row starts at no 2, but the "submitted_row_indices"
          start from 0, so we need to subtract 2 from the instructed row when matching
    """
    assert isinstance(result.task_spec, SimpleDataEntryTaskSpec)
    return result.error is None and result.progress is not None and \
        set(result.progress["submitted_row_indices"]) == set([r-2 for r in result.task_spec.rows])


def parse_strategy(strategy: str) -> RolloutStrategy:
    def _get_ids(ids: str):
        all_ids = list[int]()
        for id_group in ids.split(","):
            if "-" in id_group:
                start, stop = id_group.split("-")
                all_ids += list(range(int(start), int(stop)+1))
            else:
                all_ids.append(int(id_group))
        return all_ids

    match strategy:
        case s if (m := re.match(r"fixed\((?P<ids>\S+)\)", s)):
            ids = _get_ids(m.group("ids"))

            return FixedStrategy(tasks=[
                SimpleDataEntryTaskSpec(rows=[id])
                for id in ids
            ])
        case s if (m := re.match(r"nsuccessful\((?P<ids>\S+);(?P<min_successful>\d+);(?P<min_attempts>\d+);(?P<max_attempts>\d+)\)", s)):
            ids = _get_ids(m.group("ids"))
            return NSuccessfulStrategy(
                tasks=[
                    SimpleDataEntryTaskSpec(rows=[id])
                    for id in ids
                ], 
                min_successful=int(m.group("min_successful")),
                is_rollout_success=rows_submitted_correctly,
                min_attempts=int(m.group("min_attempts")),
                max_attempts=int(m.group("max_attempts"))
            )
        case _:
            raise ValueError("Invalid strategy")


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
