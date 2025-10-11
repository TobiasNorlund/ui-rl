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


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    inference_queue = mp.Queue()
    inference_proc = mp.Process(
        target=inference_worker,
        args=(inference_queue,)
    )
    inference_proc.start()

    session_proc = mp.Process(
        target=run_cua_session,
        kwargs={
            "pod_manifest_fn": simple_data_entry_pod_manifest,
            "cua_inference_queue": inference_queue,
            "reward_fn": reward_fn,
            "cluster_host": "34.51.245.237"
        }
    )
    session_proc.start()
    session_proc.join()
    inference_queue.put(None)
    inference_proc.join()


if __name__ == "__main__":
    main()