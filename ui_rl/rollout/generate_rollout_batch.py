import logging
import asyncio
import re
import httpx
from pathlib import Path
from datetime import datetime
from simple_data_entry import SimpleDataEntryTask
from cua import run_cua_rollout, load_kube_config
from strategy import Strategy, FixedStrategy, NSuccessfulStrategy
from uitars import UITARSRollout


async def run_single_rollout(
    rollout_id: int,
    task: SimpleDataEntryTask,
    cluster_host: str,
    model_host: str,
    model_name: str,
    max_steps: int,
    httpx_client: httpx.AsyncClient
):
    """Run a single rollout and return the result"""
    logging.info(f"Starting UITARS rollout for task: {task}")

    rollout = UITARSRollout(
        task=task,
        model_host=model_host,
        model_name=model_name,
        httpx_client=httpx_client,
        temperature=0.1
    )

    try:
        await run_cua_rollout(
            rollout=rollout,
            cluster_host=cluster_host,
            max_steps=max_steps,
            httpx_client=httpx_client
        )
        logging.info(f"Rollout {rollout_id} completed")
        return rollout_id, rollout, None
    except asyncio.CancelledError:
        logging.info(f"Rollout {rollout_id} cancelled")
        raise
    except Exception as e:
        logging.error(f"Rollout {rollout_id} failed with error: {e}")
        return rollout_id, None, e


async def main(cluster_host: str, model_host: str, model_name: str, strategy_str: str, max_parallel: int, max_steps: int):
    """
    Run single-row Simple Data Entry rollouts until at least N successful rollouts are generated for each row
    """
    load_kube_config()

    # Create log directory with timestamp
    repo_root = Path(__file__).parent.parent.parent
    base_run_dir = repo_root / "data" / "rollouts" / datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_dir.mkdir(parents=True, exist_ok=True)

    log_file = base_run_dir / "output.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='a')
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    strategy = parse_strategy(strategy_str)

    logging.info(f"Starting generation of rollouts, using {max_parallel} parallel workers")
    logging.info(f"Logs will be saved to: {base_run_dir}")

    # Create global httpx session
    async with httpx.AsyncClient(timeout=30) as session:
        # Create initial tasks
        asyncio_tasks = set()
        for next_rollout_id in range(1, max_parallel+1):
            if (next_task := strategy.next_task()) is not None:
                asyncio_tasks.add(
                    asyncio.create_task(
                        run_single_rollout(next_rollout_id, next_task, cluster_host, model_host, model_name, max_steps, session)
                    )
                )
            else:
                break

        try:
            while True:
                # Wait for the next task to complete
                done, asyncio_tasks = await asyncio.wait(asyncio_tasks, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    rollout_id, rollout, error = await task
                    if error is None:
                        rollout_row = rollout.task.rows[0]
                        # If success
                        if set(rollout.progress["submitted_row_indices"]) == set([rollout_row-2]):
                            strategy.on_success(rollout.task)
                            rollout.save(base_run_dir / f"rollout_{rollout_id:04d}_success.json")
                            logging.info(f"Successful rollout for task {rollout.task}")
                        else:
                            logging.info(f"Failed rollout for task {rollout.task}")
                            rollout.save(base_run_dir / f"rollout_{rollout_id:04d}_fail.json")
                    else:
                        logging.warning(f"Rollout {rollout_id} was interrupted due to an unrecoverable error")
                        strategy.on_error(rollout.task)
                    
                    # Start next rollout
                    if (next_task := strategy.next_task()) is not None:
                        next_rollout_id += 1
                        asyncio_tasks.add(
                            asyncio.create_task(
                                run_single_rollout(next_rollout_id, next_task, cluster_host, model_host, model_name, max_steps, session)
                            )
                        )
                    else:
                        break
                else:
                    continue
                if len(asyncio_tasks) == 0:
                    break # Only hits this if inner loop breaks out and there are no more remaining tasks

        except asyncio.CancelledError:
            logging.info("Received cancellation signal, cancelling all running tasks...")
            # Cancel all remaining tasks
            for task in asyncio_tasks:
                if not task.done():
                    task.cancel()
            # Wait for all tasks to finish cancelling
            await asyncio.gather(*asyncio_tasks, return_exceptions=True)
            raise

        # Summary
        logging.info("="*60)
        logging.info(f"Batch rollout generation complete!")


def parse_strategy(strategy: str) -> Strategy:
    def _get_ids(ids: str):
        all_ids = set()
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
                SimpleDataEntryTask(rows=[id])
                for id in ids
            ])
        case s if (m := re.match(r"nsuccess\((?P<ids>\S+);(?P<min_successful>\d+);(?P<max_attempts>\d+)\)", s)):
            ids = _get_ids(m.group("ids"))
            return NSuccessfulStrategy(
                tasks=[
                    SimpleDataEntryTask(rows=[id])
                    for id in ids
                ], 
                min_successful=int(m.group("min_successful")), 
                max_attempts=int(m.group("max_attempts"))
            )
        case _:
            raise ValueError("Invalid strategy")
            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate batch rollouts of SimpleDataEntryTask")
    parser.add_argument("--cluster-host", default="34.51.223.83:8000", help="Cluster host address")
    parser.add_argument("--vllm-host", default="localhost:8000", help="Model host address")
    parser.add_argument("--model-name", default="ByteDance-Seed/UI-TARS-1.5-7B", help="Model name in vLLM")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--max-parallel", type=int, default=1, help="Maximum number of parallel rollouts")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum steps per rollout")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.cluster_host, args.vllm_host, args.model_name, args.strategy, args.max_parallel, args.max_steps))
    except KeyboardInterrupt:
        logging.info("Script interrupted by user (Ctrl+C)")
        exit(0)
