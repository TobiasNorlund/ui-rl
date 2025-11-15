import logging
import asyncio
import random
import httpx
from pathlib import Path
from datetime import datetime
from simple_data_entry import SimpleDataEntryTask
from cua import run_cua_rollout, load_kube_config
from uitars import UITARSRollout


async def run_single_rollout(
    rollout_id: int,
    cluster_host: str,
    model_host: str,
    model_name: str,
    base_run_dir: Path,
    max_steps: int,
    httpx_client: httpx.AsyncClient
):
    """Run a single rollout and return the result"""
    logging.info(f"Starting rollout {rollout_id}")

    task = SimpleDataEntryTask(
        rows=[random.randint(2, 101)]
    )
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
            task=task,
            cluster_host=cluster_host,
            max_steps=max_steps,
            httpx_client=httpx_client
        )
        rollout.save(base_run_dir / f"rollout_{rollout_id:03d}.json")
        logging.info(f"Rollout {rollout_id} completed")
        return rollout_id, rollout, None
    except asyncio.CancelledError:
        logging.info(f"Rollout {rollout_id} cancelled")
        raise
    except Exception as e:
        logging.error(f"Rollout {rollout_id} failed with error: {e}")
        return rollout_id, None, e


async def main(cluster_host: str, model_host: str, model_name: str, n: int, max_parallel: int, max_steps: int):
    """
    Generate N rollouts of the SimpleDataEntryTask using asyncio with at most M parallel workers
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)

    load_kube_config()

    # Create log directory with timestamp
    repo_root = Path(__file__).parent.parent
    base_run_dir = repo_root / "rollouts" / datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting {n} rollouts with max {max_parallel} parallel workers")
    logging.info(f"Logs will be saved to: {base_run_dir}")

    # Collect results
    results = []
    failed = []

    # Create a semaphore to limit parallelism
    semaphore = asyncio.Semaphore(max_parallel)

    # Create global httpx session
    async with httpx.AsyncClient(timeout=30) as session:
        async def run_with_semaphore(rollout_id):
            async with semaphore:
                return await run_single_rollout(rollout_id, cluster_host, model_host, model_name, base_run_dir, max_steps, session)

        # Create initial tasks
        tasks = {asyncio.create_task(run_with_semaphore(rollout_id)) for rollout_id in range(n)}
        total_attempts = n

        try:
            # Keep going until we have n successful rollouts
            while len(results) < n:
                # Wait for the next task to complete
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    rollout_id, rollout, error = await task
                    if error is None:
                        results.append((rollout_id, rollout))
                        logging.info(f"Progress: {len(results)}/{n} successful rollouts")
                    else:
                        failed.append((rollout_id, error))
                        # Start a new task to replace the failed one
                        if len(results) < n:
                            logging.warning(f"Rollout {rollout_id} failed, restarting it...")
                            new_task = asyncio.create_task(run_with_semaphore(rollout_id))
                            tasks.add(new_task)
                            total_attempts += 1

        except asyncio.CancelledError:
            logging.info("Received cancellation signal, cancelling all running tasks...")
            # Cancel all remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for all tasks to finish cancelling
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        # Summary
        logging.info("="*60)
        logging.info(f"Batch rollout generation complete!")
        logging.info(f"Requested rollouts: {n}")
        logging.info(f"Successful: {len(results)}")
        logging.info(f"Failed: {len(failed)}")
        logging.info(f"Total attempts: {total_attempts}")

        if results:
            # TODO: SimpleDataEntry specific!
            rewards = [set(i-2 for i in rollout._task.rows) == set(rollout.progress["submitted_row_indices"]) for _, rollout in results]
            logging.info(f"Average reward: {sum(rewards) / len(rewards):.2f}")
            logging.info(f"Min reward: {min(rewards):.2f}")
            logging.info(f"Max reward: {max(rewards):.2f}")

        if failed:
            logging.info(f"Failed rollout IDs: {[rollout_id for rollout_id, _ in failed]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate batch rollouts of SimpleDataEntryTask")
    parser.add_argument("--cluster-host", default="34.51.223.83:8000", help="Cluster host address")
    parser.add_argument("--vllm-host", default="localhost:8000", help="Model host address")
    parser.add_argument("--model-name", default="ByteDance-Seed/UI-TARS-1.5-7B", help="Model name in vLLM")
    parser.add_argument("-n", "--n", type=int, default=1, help="Number of rollouts to generate")
    parser.add_argument("-m", "--max-parallel", type=int, default=1, help="Maximum number of parallel rollouts")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum steps per rollout")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.cluster_host, args.vllm_host, args.model_name, args.n, args.max_parallel, args.max_steps))
    except KeyboardInterrupt:
        logging.info("Script interrupted by user (Ctrl+C)")
        exit(0)
