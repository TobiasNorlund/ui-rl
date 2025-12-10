import logging
import asyncio
from typing import Callable
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
    `next_task()` returns the same task until `min_successful` rollouts have completed without error
    """
    
    def __init__(self, tasks: list[TaskSpec], min_successful: int, max_attempts: int = 100):
        self._tasks = tasks
        self._min_successful = min_successful
        self._max_attempts = max_attempts
        self._success_counter = {
            task: 0
            for task in tasks
        }
        self._attempt_counter = {
            task: 0
            for task in tasks
        }

    def next_task(self):
        for task in self._tasks:
            if self._success_counter[task] < self._min_successful and self._attempt_counter[task] < self._max_attempts:
                self._attempt_counter[task] += 1
                return task
        else:
            return None

    def on_rollout_finish(self, result: RolloutResult):
        """
        Treat rollout as successful if there was no error
        """
        if result.error is None:
            self._success_counter[result.task_spec] += 1


async def run_rollout(
    rollout_id: int,
    task_spec: TaskSpec,
    runtime: CUASessionRuntime,
    model_host: str,
    model_name: str,
    max_steps: int,
    httpx_client: httpx.AsyncClient
) -> RolloutResult:
    """
    Run a single rollout and return the result
    """
    logger.info(f"Starting UITARS rollout for task: {task_spec}")
    rollout = UITARS15_Rollout(
        task_instruction=task_spec,
        model_host=model_host,
        model_name=model_name,
        httpx_client=httpx_client,
        temperature=0.1
    )
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
    vllm_host: str, 
    model_name: str, 
    strategy: RolloutStrategy, 
    runtime: CUASessionRuntime, 
    max_parallel: int, 
    max_steps: int,
    httpx_client: httpx.AsyncClient,
    on_rollout_finish: Callable[[RolloutResult], None]
):
    """
    Run and save rollouts using a rollout strategy
    """
    # Create asyncio tasks for each rollout
    asyncio_tasks = set()
    for next_rollout_id in range(1, max_parallel+1):
        if (next_task_spec := strategy.next_task()) is not None:
            asyncio_tasks.add(
                asyncio.create_task(
                    run_rollout(next_rollout_id, next_task_spec, runtime, vllm_host, model_name, max_steps, httpx_client)
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
                on_rollout_finish(result)
                strategy.on_rollout_finish(result)
                
                # Start next rollout
                if (next_task_spec := strategy.next_task()) is not None:
                    next_rollout_id += 1
                    asyncio_tasks.add(
                        asyncio.create_task(
                            run_rollout(next_rollout_id, next_task_spec, runtime, vllm_host, model_name, max_steps, httpx_client)
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
