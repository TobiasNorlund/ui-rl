import logging
import asyncio
from pathlib import Path
from datetime import datetime
from functools import partial
from simple_data_entry import SimpleDataEntryTask
from cua import run_cua_session
from uitars import predict_next_action



async def main(cluster_host: str, model_host: str):
    """
    This demo launches a simple-data-entry session pod, and generates a rollout
    using a mocked CUA inference model
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Create log directory with timestamp
    repo_root = Path(__file__).parent.parent
    run_dir = repo_root / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")

    rollout = await run_cua_session(
        task=SimpleDataEntryTask(),
        predict_next_action=partial(predict_next_action, model_host=model_host),
        cluster_host=cluster_host,
        max_steps=20,
        log_dir=run_dir
    )
    logging.info(f"Final reward: {rollout.reward}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_host", default="34.51.229.41:8000")
    parser.add_argument("--model_host", default="35.204.184.155:8000")
    args = parser.parse_args()
    asyncio.run(main(args.cluster_host, args.model_host))