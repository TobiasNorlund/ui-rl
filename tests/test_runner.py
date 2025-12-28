from ui_rl.runner import NSuccessfulStrategy, RolloutResult
from pytest import fixture

from ui_rl.runtime import CUASessionRuntime
from ui_rl.task import TaskSpec


class ExampleTaskSpec(TaskSpec):
    def __init__(self, id: int):
        self.id = id
    def create_session(self, runtime: CUASessionRuntime):
        pass
    def get_task_instruction(self) -> str:
        pass


class TestNSuccessfulStrategy:

    def test_min_success_criteria(self):
        strategy = NSuccessfulStrategy(
            tasks=[ExampleTaskSpec(id=i) for i in range(2)],
            is_rollout_success=lambda r: True,
            min_successful=2,
        )

        next_task = strategy.next_task()
        assert next_task.id == 0
        strategy.on_rollout_finish(RolloutResult(next_task.id, next_task, None, None))

        next_task = strategy.next_task()
        assert next_task.id == 1, "Should prioritize the task with least successes"
        strategy.on_rollout_finish(RolloutResult(next_task.id, next_task, None, None))

        # Attempt each task again
        next_task = strategy.next_task()
        strategy.on_rollout_finish(RolloutResult(next_task.id, next_task, None, None))
        next_task = strategy.next_task()
        strategy.on_rollout_finish(RolloutResult(next_task.id, next_task, None, None))
        
        next_task = strategy.next_task()
        assert next_task is None

    def test_min_attempts(self):
        strategy = NSuccessfulStrategy(
            tasks=[ExampleTaskSpec(id=i) for i in range(1)],
            is_rollout_success=lambda r: True,
            min_successful=1,
            min_attempts=2
        )

        next_task = strategy.next_task()
        assert next_task.id == 0
        strategy.on_rollout_finish(RolloutResult(next_task.id, next_task, None, None))

        next_task = strategy.next_task()
        assert next_task.id == 0, "Should attempt twice despite min_successful is met"
        strategy.on_rollout_finish(RolloutResult(next_task.id, next_task, None, None))

        next_task = strategy.next_task()
        assert next_task is None

    def test_max_attempts(self):
        strategy = NSuccessfulStrategy(
            tasks=[ExampleTaskSpec(id=i) for i in range(1)],
            is_rollout_success=lambda r: False,
            min_successful=1,
            max_attempts=1
        )

        next_task = strategy.next_task()
        assert next_task.id == 0
        strategy.on_rollout_finish(RolloutResult(next_task.id, next_task, None, None))

        next_task = strategy.next_task()
        assert next_task is None, "Should only attempt once"
