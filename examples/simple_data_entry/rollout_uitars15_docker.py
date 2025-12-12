import asyncio
import re
import httpx
import logging
from datetime import datetime
from pathlib import Path
from functools import partial

from ui_rl import RolloutResult, RolloutStrategy, FixedStrategy, NSuccessfulStrategy, run_rollouts
from ui_rl.runtime.docker import DockerSessionRuntime

from task import SimpleDataEntryTaskSpec


async def main(
    vllm_host: str, 
    model_name: str, 
    strategy: RolloutStrategy, 
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
            logging.FileHandler(log_file, mode='a')
        ]
    )
    # Disable verbose logging from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.info(f"Starting generation of rollouts, using {max_parallel} parallel workers")
    logging.info(f"Logs will be saved to: {output_dir}")

    # Use a global httpx session
    async with httpx.AsyncClient(timeout=30) as httpx_client:
        # Create docker session runtime 
        runtime = DockerSessionRuntime(httpx_client=httpx_client)

        # Run rollouts
        await run_rollouts(
            vllm_host=vllm_host,
            model_name=model_name,
            strategy=strategy,
            runtime=runtime,
            max_parallel=max_parallel,
            max_steps=max_steps,
            httpx_client=httpx_client,
            on_rollout_finish=partial(on_rollout_finish, output_dir=output_dir)
        )


def on_rollout_finish(result: RolloutResult, output_dir: Path):
    """
    Save rollouts that finished without error
    """
    if result.error is None:
        if is_rollout_correct(result):
            result.rollout.save(output_dir / f"rollout_{result.rollout_id:04d}_success.json")
        else:
            result.rollout.save(output_dir / f"rollout_{result.rollout_id:04d}_fail.json")


def is_rollout_correct(result: RolloutResult) -> bool:
    """
    Checks if all the instructed spreadsheet rows were actually submitted
    Note: In the spreadsheet, the first data row starts at no 2, but the "submitted_row_indices"
          start from 0, so we need to subtract 2 from the instructed row when matching
    """
    assert isinstance(result.task_spec, SimpleDataEntryTaskSpec)
    return set(result.rollout.progress["submitted_row_indices"]) == set([r-2 for r in result.task_spec.rows])


class NCorrectRowsStrategy(NSuccessfulStrategy):
    def on_rollout_finish(self, result: RolloutResult):
        """
        Override to mark rollout successful when the submitted rows match the expected
        """
        self._inflight_counter[result.task_spec] -= 1
        if result.error is None and is_rollout_correct(result):
            self._success_counter[result.task_spec] += 1


def parse_strategy(strategy: str) -> RolloutStrategy:
    def _get_ids(ids: str):
        all_ids = set[int]()
        for id_group in ids.split(","):
            if "-" in id_group:
                start, stop = id_group.split("-")
                all_ids.update(range(int(start), int(stop)+1))
            else:
                all_ids.add(int(id_group))
        return all_ids

    match strategy:
        case s if (m := re.match(r"fixed\((?P<ids>\S+)\)", s)):
            ids = _get_ids(m.group("ids"))
            return FixedStrategy(tasks=[
                SimpleDataEntryTaskSpec(rows=[id])
                for id in ids
            ])
        case s if (m := re.match(r"ncorrect\((?P<ids>\S+);(?P<min_successful>\d+);(?P<max_inflight_per_task>\d+);(?P<max_attempts>\d+)\)", s)):
            ids = _get_ids(m.group("ids"))
            return NCorrectRowsStrategy(
                tasks=[
                    SimpleDataEntryTaskSpec(rows=[id])
                    for id in ids
                ], 
                min_successful=int(m.group("min_successful")), 
                max_inflight_per_task=int(m.group("max_inflight_per_task")),
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
    parser.add_argument("--strategy", required=True, help="Rollout strategy to use")
    parser.add_argument("--max-parallel", type=int, default=1, help="Maximum number of parallel rollouts")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum steps per rollout")
    parser.add_argument("--output-dir", help="Dir to save rollouts and logs")
    args = parser.parse_args()

    if args.output_dir is None:
        repo_root = Path(__file__).parent.parent.parent
        output_dir = repo_root / "data" / "rollouts" / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_dir = args.output_dir

    rollout_strategy = parse_strategy(args.strategy)

    try:
        asyncio.run(main(
            vllm_host=args.vllm_host,
            model_name=args.model_name,
            strategy=rollout_strategy,
            max_parallel=args.max_parallel,
            max_steps=args.max_steps,
            output_dir=output_dir
        ))
    except KeyboardInterrupt:
        logging.info("Script interrupted by user (Ctrl+C)")
        exit(0)
