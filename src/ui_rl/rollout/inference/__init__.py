from abc import ABC, abstractmethod


class InferenceEngine(ABC):
    """
    An inference engine for completing an LLM prompt
    """

    @abstractmethod
    def generate(self, messages: list):
        pass
