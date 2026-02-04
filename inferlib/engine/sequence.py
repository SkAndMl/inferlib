from dataclasses import dataclass
from enum import Enum, auto


class SequenceState(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    s_id: int
    state: SequenceState
    prompt_tokens: list[int]
    completion_tokens: list[int]
    last_token_id: int = -1
    temperature: float = 0.1
    max_tokens: int = 200
    eos_token_id: int

    @property
    def sequence_length(self) -> int:
        return len(self.prompt_tokens) + len(self.completion_tokens)

    @property
    def is_finished(self) -> bool:
        return (
            self.last_token_id == self.eos_token_id
            or len(self.completion_tokens) == self.max_tokens
        )
