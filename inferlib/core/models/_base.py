from abc import ABC, abstractmethod
from inferlib.core.engine.sequence import Sequence
from inferlib.core.engine.page import PageManager


class Model(ABC):
    @abstractmethod
    def prefill(
        self, *, sequences: list[Sequence], page_manager: PageManager
    ) -> list[int]:
        raise NotImplementedError()

    @abstractmethod
    def decode(
        self, *, sequences: list[Sequence], page_manager: PageManager
    ) -> list[int]:
        raise NotImplementedError()
