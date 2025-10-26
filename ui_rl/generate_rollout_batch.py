import logging
import json
import base64
import asyncio
from io import BytesIO
from pathlib import Path
from datetime import datetime
from functools import partial
from simple_data_entry import SimpleDataEntryTask
from cua import run_cua_session, Rollout, State, Action
from uitars import predict_next_action


def serialize_rollout(rollout: Rollout) -> dict:
    """Convert a Rollout to a JSON-serializable dict with base64-encoded images"""
    def image_to_base64(image):
        """Convert PIL Image to base64 string"""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {
        "task": rollout.task,
        "states": [
            {"screenshot": image_to_base64(state.screenshot)}
            for state in rollout.states
        ],
        "actions": [
            {
                "action_type": action.action_type.value,
                "x": action.x,
                "y": action.y,
                "text": action.text,
                "keys": action.keys
            }
            for action in rollout.actions
        ],
        "response_messages": rollout.response_messages,
        "reward": rollout.reward,
        "progress": rollout.progress
    }


async def run_single_rollout(rollout_id: int, cluster_host: str, model_host: str, base_run_dir: Path, max_steps: int):
    """Run a single rollout and return the result"""
    logging.info(f"Starting rollout {rollout_id}")

    # Create subdirectory for this rollout
    run_dir = base_run_dir / f"rollout_{rollout_id:03d}"

    try:
        rollout = await run_cua_session(
            task=SimpleDataEntryTask(),
            predict_next_action=partial(predict_next_action, model_host=model_host),
            cluster_host=cluster_host,
            max_steps=max_steps,
            log_dir=run_dir
        )

        # Save rollout as JSON
        rollout_json = serialize_rollout(rollout)
        json_path = run_dir / "rollout.json"
        with open(json_path, 'w') as f:
            json.dump(rollout_json, f, indent=2)

        logging.info(f"Rollout {rollout_id} completed with reward: {rollout.reward}")
        return rollout_id, rollout, None
    except asyncio.CancelledError:
        logging.info(f"Rollout {rollout_id} cancelled")
        raise
    except Exception as e:
        logging.error(f"Rollout {rollout_id} failed with error: {e}")
        return rollout_id, None, e


async def main(cluster_host: str, model_host: str, n: int, max_parallel: int, max_steps: int):
    """
    Generate N rollouts of the SimpleDataEntryTask using asyncio with at most M parallel workers
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Create log directory with timestamp
    repo_root = Path(__file__).parent.parent
    base_run_dir = repo_root / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting {n} rollouts with max {max_parallel} parallel workers")
    logging.info(f"Logs will be saved to: {base_run_dir}")

    # Collect results
    results = []
    failed = []

    # Create a semaphore to limit parallelism
    semaphore = asyncio.Semaphore(max_parallel)

    async def run_with_semaphore(rollout_id):
        async with semaphore:
            return await run_single_rollout(rollout_id, cluster_host, model_host, base_run_dir, max_steps)

    # Create all tasks
    tasks = [asyncio.create_task(run_with_semaphore(rollout_id)) for rollout_id in range(n)]

    try:
        # Wait for all tasks to complete
        for task in asyncio.as_completed(tasks):
            rollout_id, rollout, error = await task
            if error is None:
                results.append((rollout_id, rollout))
            else:
                failed.append((rollout_id, error))

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
    logging.info(f"Total rollouts: {n}")
    logging.info(f"Successful: {len(results)}")
    logging.info(f"Failed: {len(failed)}")

    if results:
        rewards = [rollout.reward for _, rollout in results]
        logging.info(f"Average reward: {sum(rewards) / len(rewards):.2f}")
        logging.info(f"Min reward: {min(rewards):.2f}")
        logging.info(f"Max reward: {max(rewards):.2f}")

    if failed:
        logging.info(f"Failed rollout IDs: {[rollout_id for rollout_id, _ in failed]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate batch rollouts of SimpleDataEntryTask")
    parser.add_argument("--cluster_host", default="34.51.229.41:8000", help="Cluster host address")
    parser.add_argument("--vllm_host", default="35.204.184.155:8000", help="Model host address")
    parser.add_argument("-n", "--n", type=int, default=10, help="Number of rollouts to generate")
    parser.add_argument("-m", "--max_parallel", type=int, default=5, help="Maximum number of parallel rollouts")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum steps per rollout")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.cluster_host, args.vllm_host, args.n, args.max_parallel, args.max_steps))
    except KeyboardInterrupt:
        logging.info("Script interrupted by user (Ctrl+C)")
        exit(0)
