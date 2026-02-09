from dataclasses import dataclass
from enum import Enum, auto


class SequenceState(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    s_id: int
    prompt_tokens: list[int]
    completion_tokens: list[int]
    eos_token_id: int
    state: SequenceState = SequenceState.WAITING
    last_token_id: int = -1
    temperature: float = 0.1
    max_tokens: int = 200

    @property
    def sequence_length(self) -> int:
        return len(self.prompt_tokens) + len(self.completion_tokens)

    @property
    def is_finished(self) -> bool:
        return (
            self.last_token_id == self.eos_token_id
            or len(self.completion_tokens) == self.max_tokens
        )
