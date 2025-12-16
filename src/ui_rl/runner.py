import logging
import logging.handlers
from typing import Callable, Type
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import httpx
from dataclasses import dataclass
from .task import TaskSpec
from .agent import run_cua_rollout
from .models.uitars15 import UITARS15_Rollout
from .runtime import CUASessionRuntime


logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    rollout_id: int 
    task_spec: TaskSpec
    rollout: UITARS15_Rollout
    error: Exception | None


class RolloutStrategy(ABC):

    @abstractmethod
    def next_task(self) -> TaskSpec | None:
        pass

    def on_rollout_finish(self, result: RolloutResult):
        pass


class FixedStrategy(RolloutStrategy):
    """
    Generate rollouts for a fixed set of tasks. Optionally restart on error
    """

    def __init__(self, tasks: list[TaskSpec], rerun_on_error=False):
        self._tasks = list(tasks)
        self._rerun_on_error = rerun_on_error

    def next_task(self) -> TaskSpec | None:
        if len(self._tasks) > 0:
            return self._tasks.pop(0)
        else:
            return None

    def on_rollout_finish(self, result: RolloutResult):
        if result.error is not None and self._rerun_on_error:
            self._tasks.insert(0, result.task_spec)


class NSuccessfulStrategy(RolloutStrategy):
    """
    Generate rollouts until all tasks have at least `min_successful` succeeded rollouts.
    `next_task()` returns the same task until `min_successful` rollouts have completed without error.
    At most `min_successful` attempts per task can be in-flight at any given time.
    """

    def __init__(self, tasks: list[TaskSpec], min_successful: int, max_inflight_per_task: int = 100, max_attempts: int = 100):
        self._tasks = tasks
        self._min_successful = min_successful
        self._max_inflight_per_task = max_inflight_per_task
        self._max_attempts = max_attempts
        self._success_counter = {
            task: 0
            for task in tasks
        }
        self._attempt_counter = {
            task: 0
            for task in tasks
        }
        self._inflight_counter = {
            task: 0
            for task in tasks
        }

    def next_task(self):
        for task in self._tasks:
            if (self._success_counter[task] < self._min_successful
                and self._attempt_counter[task] < self._max_attempts
                and self._inflight_counter[task] < self._max_inflight_per_task):
                self._attempt_counter[task] += 1
                self._inflight_counter[task] += 1
                return task
        else:
            return None

    def on_rollout_finish(self, result: RolloutResult):
        """
        Treat rollout as successful if there was no error.
        Decrement the in-flight counter for the task.
        """
        self._inflight_counter[result.task_spec] -= 1
        if result.error is None:
            self._success_counter[result.task_spec] += 1


def run_rollout(
    rollout_id: int,
    rollout: UITARS15_Rollout,
    task_spec: TaskSpec,
    runtime: CUASessionRuntime,
    max_steps: int,
) -> RolloutResult:
    """
    Run a single rollout and return the result
    """
    logger.info(f"Starting UITARS rollout for task: {task_spec}")
    try:
        run_cua_rollout(
            task_spec=task_spec,
            rollout=rollout,
            runtime=runtime,
            max_steps=max_steps,
        )
        logger.info(f"Rollout {rollout_id} completed")
        return RolloutResult(rollout_id, task_spec, rollout, None)
    except Exception as e:
        logger.error(f"Rollout {rollout_id} was stopped due to an error: {e}")
        return RolloutResult(rollout_id, task_spec, rollout, e)


# Process-local storage for worker resources
_worker_httpx_client = None
_worker_runtime = None


def _worker_init(log_queue, model_host: str, runtime_class: Type[CUASessionRuntime], runtime_kwargs: dict):
    """
    Initialize worker process with logging configuration and reusable resources

    Args:
        log_queue: Multiprocessing queue for logging
        model_host: vLLM host address
        runtime_class: The runtime class to instantiate (e.g., DockerSessionRuntime or KubernetesSessionRuntime)
        runtime_kwargs: Additional kwargs for runtime initialization
    """
    global _worker_httpx_client, _worker_runtime

    # Configure logging to use QueueHandler
    queue_handler = logging.handlers.QueueHandler(log_queue)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    root_logger.addHandler(queue_handler)

    # Initialize httpx client for this worker
    limits = httpx.Limits(max_keepalive_connections=1000, max_connections=1000)
    timeout = httpx.Timeout(60.0, pool=None)
    _worker_httpx_client = httpx.Client(limits=limits, timeout=timeout)

    # Initialize runtime for this worker
    _worker_runtime = runtime_class(httpx_client=_worker_httpx_client, **runtime_kwargs)


def _run_rollout_worker(
    rollout_id: int,
    task_spec: TaskSpec,
    model_host: str,
    model_name: str,
    max_steps: int,
    rollout_kwargs: dict,
):
    """
    Worker function that runs a single rollout in a separate process.
    Uses the worker's shared httpx client and runtime.
    """
    global _worker_httpx_client, _worker_runtime

    from .models.uitars15 import UITARS15_Rollout

    # Create rollout for this task using worker's shared httpx client
    rollout = UITARS15_Rollout(
        task_spec=task_spec,
        model_host=model_host,
        model_name=model_name,
        httpx_client=_worker_httpx_client,
        **rollout_kwargs
    )

    # Run the rollout using worker's shared runtime
    result = run_rollout(rollout_id, rollout, task_spec, _worker_runtime, max_steps)
    return result


def run_rollouts(
    strategy: RolloutStrategy,
    model_host: str,
    model_name: str,
    max_parallel: int,
    max_steps: int,
    on_rollout_finish: Callable[[RolloutResult], None],
    runtime_class: Type[CUASessionRuntime],
    runtime_kwargs: dict = None,
    rollout_kwargs: dict = None,
):
    """
    Run and save rollouts using a rollout strategy with ProcessPoolExecutor

    Args:
        strategy: RolloutStrategy to determine which tasks to run
        model_host: vLLM host address
        model_name: Model name to use
        max_parallel: Maximum number of parallel worker processes
        max_steps: Maximum steps per rollout
        on_rollout_finish: Callback function to call when a rollout finishes
        runtime_class: The runtime class to use (e.g., DockerSessionRuntime or KubernetesSessionRuntime)
        runtime_kwargs: Additional kwargs for runtime initialization
        rollout_kwargs: Additional kwargs for rollout initialization
    """
    runtime_kwargs = runtime_kwargs or {}
    rollout_kwargs = rollout_kwargs or {}

    # Set up multiprocess logging
    manager = Manager()
    log_queue = manager.Queue()

    # Start log listener in main process
    queue_listener = logging.handlers.QueueListener(
        log_queue,
        *logging.getLogger().handlers,
        respect_handler_level=True
    )
    queue_listener.start()

    try:
        with ProcessPoolExecutor(
            max_workers=max_parallel,
            initializer=_worker_init,
            initargs=(log_queue, model_host, runtime_class, runtime_kwargs)
        ) as executor:
            futures = {}
            next_rollout_id = 1

            # Submit initial batch of rollouts
            for _ in range(max_parallel):
                if (next_task_spec := strategy.next_task()) is not None:
                    future = executor.submit(
                        _run_rollout_worker,
                        next_rollout_id,
                        next_task_spec,
                        model_host,
                        model_name,
                        max_steps,
                        rollout_kwargs,
                    )
                    futures[future] = next_rollout_id
                    next_rollout_id += 1
                else:
                    break

            # Process completed rollouts and submit new ones
            while futures:
                # Wait for next completed rollout
                for future in as_completed(futures):
                    result: RolloutResult = future.result()
                    del futures[future]

                    # Call callbacks
                    on_rollout_finish(result)
                    strategy.on_rollout_finish(result)

                    # Submit next rollout if available
                    if (next_task_spec := strategy.next_task()) is not None:
                        new_future = executor.submit(
                            _run_rollout_worker,
                            next_rollout_id,
                            next_task_spec,
                            model_host,
                            model_name,
                            max_steps,
                            rollout_kwargs,
                        )
                        futures[new_future] = next_rollout_id
                        next_rollout_id += 1

                    break  # Only process one completion at a time

        # Summary
        logger.info("="*60)
        logger.info(f"Batch rollout generation complete!")

    finally:
        queue_listener.stop()
