from dataclasses import dataclass
from typing import Literal


@dataclass
class Sequence:
    s_id: str
    prompt_tokens: list[int]
    completion_tokens: list[int]
    eos_token_id: int
    last_text: str = ""
    last_token_id: int = -1
    temperature: float = 0.1
    max_tokens: int = 200
    tokens_evicted: int = 0
    cached_pages: int = 0

    # prefix caching
    # set when a prefill plan is committed to the page pool. this is a per-prefill
    # value (how many tokens we can skip THIS time), as opposed to
    # cached_pages which is a running bookkeeping counter for hash eviction.
    prefix_cached_tokens: int = 0

    def __len__(self) -> int:
        return len(self.prompt_tokens) + len(self.completion_tokens)

    @property
    def is_finished(self) -> bool:
        return self.last_token_id == self.eos_token_id or len(self.completion_tokens) == self.max_tokens

    @property
    def finish_reason(self) -> Literal["stop", "length"] | None:
        if self.last_token_id == self.eos_token_id:
            return "stop"
        if len(self.completion_tokens) == self.max_tokens:
            return "length"
        return None

    @property
    def tokens(self) -> list[int]:
        return self.prompt_tokens + self.completion_tokens
