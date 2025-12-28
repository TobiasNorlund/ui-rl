from abc import ABC, abstractmethod
import json
import hashlib

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

    def as_dict(self) -> dict:
        return vars(self)

    def __eq__(self, other: "TaskSpec"):
        return self.as_dict() == other.as_dict()

    def __hash__(self):
        # 1. Convert to a sorted JSON string to handle nested structures
        # 2. Use sort_keys=True so order doesn't change the hash
        encoded_data = json.dumps(self.as_dict(), sort_keys=True).encode('utf-8')
        
        # 3. Create a deterministic hash using MD5 or SHA256
        # 4. Convert the hex digest to an integer (required for __hash__)
        hash_hex = hashlib.md5(encoded_data).hexdigest()
        return int(hash_hex, 16)