from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal


class SequenceState(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    s_id: str
    prompt_tokens: list[int]
    completion_tokens: list[int]
    eos_token_id: int
    last_text: str = ""
    state: SequenceState = SequenceState.WAITING
    last_token_id: int = -1
    temperature: float = 0.1
    max_tokens: int = 200

    def __len__(self) -> int:
        return len(self.prompt_tokens) + len(self.completion_tokens)

    @property
    def sequence_length(self) -> int:
        return len(self.prompt_tokens) + len(self.completion_tokens)

    @property
    def is_finished(self) -> bool:
        return (
            self.last_token_id == self.eos_token_id
            or len(self.completion_tokens) == self.max_tokens
        )

    @property
    def finish_reason(self) -> Literal["stop", "length"] | None:
        if self.last_token_id == self.eos_token_id:
            return "stop"
        if len(self.completion_tokens) == self.max_tokens:
            return "length"
        return None
