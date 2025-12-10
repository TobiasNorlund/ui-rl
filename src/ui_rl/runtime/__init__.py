from abc import ABC, abstractmethod
from ..cua import Action, State


class CUASessionRuntime(ABC):

    @abstractmethod
    def create_session(self, *args, **kwargs) -> str:
        pass

    @abstractmethod
    def teardown_session(self, session_id: str):
        pass

    @abstractmethod
    async def session_ready(self, session_id: str):
        pass

    @abstractmethod
    async def session_act(self, session_id: str, action: Action) -> State:
        pass

    @abstractmethod
    async def get_session_progress(self, session_id: str) -> dict:
        pass
