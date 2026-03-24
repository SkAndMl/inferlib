from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FinishReason = Literal["stop", "length"]


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 16
    ignore_eos: bool = False
    thinking: bool = False


@dataclass(slots=True)
class GenerationOutput:
    text: str
    token_ids: list[int]
    finish_reason: FinishReason
    prompt_tokens: int
    completion_tokens: int
