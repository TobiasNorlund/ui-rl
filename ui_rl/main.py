import logging
import random
import multiprocessing as mp
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


def main():
    """
    This demo launches a simple-data-entry session pod, and generates a rollout
    using a mocked CUA inference model
    """

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    
    inference_queue = mp.Queue()
    inference_proc = mp.Process(
        target=inference_worker,
        args=(inference_queue,)
    )
    inference_proc.start()

    rollout_queue = mp.Queue()
    session_proc = mp.Process(
        target=run_cua_session,
        kwargs={
            "pod_manifest_fn": simple_data_entry_pod_manifest,
            "cua_inference_queue": inference_queue,
            "rollout_queue": rollout_queue,
            "reward_fn": reward_fn,
            "cluster_host": "34.51.245.237",
            "max_steps": 10
        }
    )
    # Start CUA session
    session_proc.start()

    # Wait until we get a finished rollout
    rollout = rollout_queue.get()

    # Join processes and clean up
    session_proc.join()
    inference_queue.put(None)
    inference_proc.join()

    # Print final result
    print(rollout)


if __name__ == "__main__":
    main()