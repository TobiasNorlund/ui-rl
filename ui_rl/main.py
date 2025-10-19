import logging
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import sleep
from .simple_data_entry import simple_data_entry_pod_manifest, reward_fn
from .cua import run_cua_session, Action, ActionType


def inference_worker(inference_queue: mp.Queue):
    logging.info("Starting inference worker")
    while (input := inference_queue.get()):
        rollout, res_queue = input
        logging.info("Got inference job")
        rollout.states[-1].screenshot.save(f"{len(rollout.states)}.png")
        sleep(1)
        res_queue.put(Action(
            action_type=ActionType.LeftClick,
            x=random.randint(100, 400),
            y=random.randint(300, 700)
        ))
    logging.info("Received sentinel, exiting inference worker")


def main(cluster_host: str):
    """
    This demo launches a simple-data-entry session pod, and generates a rollout
    using a mocked CUA inference model
    """

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    
    with mp.Manager() as manager:
        # Start inference worker
        inference_queue = manager.Queue()
        inference_proc = mp.Process(
            target=inference_worker,
            args=(inference_queue,)
        )
        inference_proc.start()

        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    run_cua_session, 
                    pod_manifest_fn=simple_data_entry_pod_manifest,
                    cua_inference_queue=inference_queue,
                    reward_fn=reward_fn,
                    cluster_host=cluster_host,
                    max_steps=10
                )
                for _ in range(3)
            ]

            # Collect rollouts
            for future in as_completed(futures):
                if err := future.exception():
                    logging.error(err)
                else:
                    logging.info(future.result())

        # Close inference worker
        inference_queue.put(None)
        inference_proc.join()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster_host")
    args = parser.parse_args()
    main(args.cluster_host)