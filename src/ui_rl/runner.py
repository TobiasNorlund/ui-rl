import logging
import asyncio
from typing import Callable, Coroutine
from abc import ABC, abstractmethod
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


async def run_rollout(
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
        await run_cua_rollout(
            task_spec=task_spec,
            rollout=rollout,
            runtime=runtime,
            max_steps=max_steps,
        )
        logger.info(f"Rollout {rollout_id} completed")
        return RolloutResult(rollout_id, task_spec, rollout, None)
    except asyncio.CancelledError:
        logger.info(f"Rollout {rollout_id} cancelled")
        raise
    except Exception as e:
        logger.error(f"Rollout {rollout_id} was stopped due to an error: {e}")
        return RolloutResult(rollout_id, task_spec, rollout, e)


async def run_rollouts(
    strategy: RolloutStrategy, 
    rollout_factory: Callable[[TaskSpec], UITARS15_Rollout],
    runtime: CUASessionRuntime, 
    max_parallel: int, 
    max_steps: int,
    on_rollout_finish: Coroutine
):
    """
    Run and save rollouts using a rollout strategy
    """
    # Create asyncio tasks for each rollout
    asyncio_tasks = set()
    for next_rollout_id in range(1, max_parallel+1):
        if (next_task_spec := strategy.next_task()) is not None:
            rollout = rollout_factory(next_task_spec)
            asyncio_tasks.add(
                asyncio.create_task(
                    run_rollout(next_rollout_id, rollout, next_task_spec, runtime, max_steps)
                )
            )
        else:
            break

    try:
        while True:
            # Wait for the next task to complete
            done, asyncio_tasks = await asyncio.wait(asyncio_tasks, return_when=asyncio.FIRST_COMPLETED)
            for asyncio_task in done:
                result: RolloutResult = await asyncio_task
                
                # Call callbacks
                await on_rollout_finish(result)
                strategy.on_rollout_finish(result)
                
                # Start next rollout
                if (next_task_spec := strategy.next_task()) is not None:
                    next_rollout_id += 1
                    rollout = rollout_factory(next_task_spec)
                    asyncio_tasks.add(
                        asyncio.create_task(
                            run_rollout(next_rollout_id, rollout, next_task_spec, runtime, max_steps)
                        )
                    )
                else:
                    break
            else:
                continue
            if len(asyncio_tasks) == 0:
                break # Only hits this if inner loop breaks out and there are no more remaining tasks

    except asyncio.CancelledError:
        logger.info("Received cancellation signal, cancelling all running tasks...")
        # Cancel all remaining tasks
        for asyncio_task in asyncio_tasks:
            if not asyncio_task.done():
                asyncio_task.cancel()
        # Wait for all tasks to finish cancelling
        await asyncio.gather(*asyncio_tasks, return_exceptions=True)
        raise

    # Summary
    logger.info("="*60)
    logger.info(f"Batch rollout generation complete!")
