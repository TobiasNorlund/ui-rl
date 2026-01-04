import threading
import logging
import requests
from queue import Queue
from pydantic import BaseModel, ConfigDict, field_validator, ValidationError
from http.server import HTTPServer, BaseHTTPRequestHandler
from functools import partial
from pathlib import Path

from ui_rl.runner import RolloutStrategy, run_rollouts
from simple_data_entry import SimpleDataEntryRolloutWorker, parse_strategy
from launch_vllm import launch as launch_vllm, await_vllm_ready


class RolloutBatchRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model_name: str
    output_dir: Path
    rollout_strategy: RolloutStrategy
    max_steps: int
    lora_name: str | None = None
    lora_path: str | None = None

    @field_validator('rollout_strategy', mode='before')
    @classmethod
    def parse_strategy(cls, v):
        if isinstance(v, str):
            return parse_strategy(v)
        return v


def main(
    port: int, 
    gpus: list[int], 
    model_name: str, 
    max_parallel_rollouts: int,
    mounts: list[str],
    vllm_args: list[str]
):

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    
    with launch_vllm(
        gpus=gpus, 
        model_name=model_name,
        mounts=mounts,
        vllm_args=vllm_args,
        detach=True
    ):
        # Wait until ready
        logging.info("Starting vLLM...")
        await_vllm_ready()

        # Start server
        task_queue = Queue[RolloutBatchRequest]()
        service = RolloutService(task_queue=task_queue, port=port)
        logging.info(f"Simple Data Entry Rollout Service running on port {port}...")
        service.start()

        try:
            while (rollout_batch_request := task_queue.get()) is not None:
                logging.info("Starting to process rollout request")

                # Create requested dir
                rollout_batch_request.output_dir.mkdir(parents=True, exist_ok=True)

                # Optionally load lora
                if rollout_batch_request.lora_name is not None and rollout_batch_request.lora_path is not None:
                    logging.info(f"Loading lora: {rollout_batch_request.lora_name}")
                    requests.post("http://localhost:8000/v1/load_lora_adapter", data={
                        "lora_name": rollout_batch_request.lora_name,
                        "lora_path": rollout_batch_request.lora_path
                    })

                worker = SimpleDataEntryRolloutWorker(
                    model_host="localhost:8000",
                    model_name=rollout_batch_request.model_name,
                    max_steps=rollout_batch_request.max_steps,
                    output_dir=rollout_batch_request.output_dir
                )
                logging.info(f"Running rollouts with parallelism: {max_parallel_rollouts}")
                run_rollouts(
                    strategy=rollout_batch_request.rollout_strategy,
                    rollout_worker=worker,
                    max_parallel=max_parallel_rollouts,
                )

        except KeyboardInterrupt:
            logging.info("Stopping...")

        finally:
            service.stop()
            service.join()


class RolloutServiceHandler(BaseHTTPRequestHandler):
    def __init__(self, task_queue: Queue, *args, **kwargs):
        self._task_queue = task_queue
        super().__init__(*args, **kwargs)
    
    def do_POST(self):
        if self.path != "/":
            self.send_response(404)
            return
        try:
            content_length = int(self.headers['Content-Length'])
            payload = self.rfile.read(content_length).decode('utf-8')
            self._task_queue.put(RolloutBatchRequest.model_validate_json(payload))
            self.send_response(200)
        except ValidationError as e:
            self.send_response(400)
            print(f"ERROR: {e}")


class RolloutService(threading.Thread):
    def __init__(self, task_queue: Queue, host="127.0.0.1", port=8001):
        super().__init__()
        self._server = HTTPServer((host, port), partial(RolloutServiceHandler, task_queue))

    def run(self):
        self._server.serve_forever()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--max-parallel-rollouts", type=int, default=20)
    parser.add_argument("--mount", nargs="+", default=[])
    args, vllm_args = parser.parse_known_args()

    main(**vars(args))
