from abc import ABC, abstractmethod
from typing import Any

from .runtime import CUASessionRuntime


class TaskSpec(ABC):

    @abstractmethod
    def get_task_instruction(self) -> str:
        """
        Returns an instruction string, to be used for prompting a CUA LLM to perform this task
        """
        pass

    @abstractmethod
    def create_session(self, runtime: CUASessionRuntime):
        """
        Creates a session using `runtime.create_session(...)` for performing this task, and returns the session id string
        """
        pass

    def as_dict(self) -> Any:
        return vars(self)