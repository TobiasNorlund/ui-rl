import logging
import logging.handlers
from queue import Queue
from typing import Callable
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import httpx
from dataclasses import dataclass
from .task import TaskSpec


logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    rollout_id: int 
    task_spec: TaskSpec
    progress: dict | None
    error: Exception | None


class RolloutWorker(ABC):
    """
    Manages and hold the necessary state for running rollouts in a worker process
    """
    _httpx_client = None

    @classmethod
    def init_worker(cls, log_queue: Queue):
        # Configure logging to use QueueHandler
        queue_handler = logging.handlers.QueueHandler(log_queue)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers = []
        root_logger.addHandler(queue_handler)

        # Initialize httpx client for this worker
        limits = httpx.Limits(max_keepalive_connections=100, max_connections=100)
        timeout = httpx.Timeout(60.0, pool=None)
        cls._httpx_client = httpx.Client(limits=limits, timeout=timeout)

    @abstractmethod
    def run(self, rollout_id: int, task_spec: TaskSpec) -> RolloutResult:
        """
        Creates a rollout, runs it and returns a RolloutResult
        """
        pass


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

    def __init__(
        self, 
        tasks: list[TaskSpec], 
        min_successful: int, 
        is_rollout_success: Callable[[RolloutResult], bool],
        min_attempts: int = 1,
        max_attempts: int = 100
    ):
        self._tasks = tasks
        self._min_successful = min_successful
        self._is_rollout_success = is_rollout_success
        self._min_attempts = min_attempts
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
        # Prioritize: 1) Least successes 2) Least attempts 3) Least in-flights
        tasks = [
            task for task in self._tasks if \
                self._attempt_counter[task] < self._min_attempts or \
                (
                    self._success_counter[task] < self._min_successful and \
                    self._attempt_counter[task] < self._max_attempts
                )
        ]
        if len(tasks) == 0:
            return None

        # Get tasks with least successes
        tasks = sorted(tasks, key=lambda task: self._success_counter[task])
        tasks = [task for task in tasks if self._success_counter[task] == self._success_counter[tasks[0]]]

        # Get tasks with least attempts
        tasks = sorted(tasks, key=lambda task: self._attempt_counter[task])
        tasks = [task for task in tasks if self._attempt_counter[task] == self._attempt_counter[tasks[0]]]

        # Get tasks with least in-flight
        tasks = sorted(tasks, key=lambda task: self._inflight_counter[task])
        tasks = [task for task in tasks if self._inflight_counter[task] == self._inflight_counter[tasks[0]]]

        task = tasks[0]
        self._attempt_counter[task] += 1
        self._inflight_counter[task] += 1
        return task

    def on_rollout_finish(self, result: RolloutResult):
        """
        Treat rollout as successful if there was no error.
        Decrement the in-flight counter for the task.
        """
        self._inflight_counter[result.task_spec] -= 1
        if self._is_rollout_success(result):
            self._success_counter[result.task_spec] += 1


def run_rollouts(
    strategy: RolloutStrategy,
    rollout_worker: RolloutWorker,
    max_parallel: int,
):
    """
    Run rollout strategy with ProcessPoolExecutor
    """
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
            initializer=rollout_worker.init_worker,
            initargs=(log_queue,)
        ) as executor:
            futures = {}
            next_rollout_id = 1

            # Submit initial batch of rollouts
            for _ in range(max_parallel):
                if (next_task_spec := strategy.next_task()) is not None:
                    future = executor.submit(
                        rollout_worker.run,
                        next_rollout_id,
                        next_task_spec,
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

                    # Notify strategy
                    strategy.on_rollout_finish(result)

                    # Submit next rollout if available
                    if (next_task_spec := strategy.next_task()) is not None:
                        new_future = executor.submit(
                            rollout_worker.run,
                            next_rollout_id,
                            next_task_spec,
                        )
                        futures[new_future] = next_rollout_id
                        next_rollout_id += 1

                    break  # Only process one completion at a time, as futures is modified

        # Summary
        logger.info("="*60)
        logger.info(f"Batch rollout generation complete!")

    finally:
        queue_listener.stop()
