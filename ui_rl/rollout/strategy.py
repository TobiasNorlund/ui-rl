
from abc import ABC, abstractmethod
import math


class Strategy(ABC):

    @abstractmethod
    def next_task(self):
        pass

    def on_success(self, task):
        pass


class FixedStrategy(Strategy):
    """
    Generate rollouts for a fixed set of tasks, irregardless of how many succeeds
    """

    def __init__(self, tasks):
        self._tasks = tasks
        self._next_task_index = 0

    def next_task(self):
        if self._next_task_index < len(self._tasks):
            self._next_task_index += 1
            return self._tasks[self._next_task_index-1]
        else:
            return None


class NSuccessfulStrategy(Strategy):
    """
    Generate rollouts until all tasks have at least `min_successful` succeeded rollouts
    """
    
    def __init__(self, tasks, min_successful: int, max_attempts: int = math.inf):
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

    def on_success(self, task):
        self._success_counter[task] += 1
