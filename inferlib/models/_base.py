from abc import ABC, abstractmethod
from inferlib.engine.sequence import Sequence
from inferlib.engine.page import PageManager


class Model(ABC):
    @abstractmethod
    def prefill(self, *, sequences: list[Sequence], cache: PageManager) -> None:
        raise NotImplementedError()

    @abstractmethod
    def decode(self, *, sequences: list[Sequence], cache: PageManager) -> list[int]:
        raise NotImplementedError()
