from __future__ import annotations

import tiktoken

from inferlib.schema.payload import Payload
from inferlib.schema.sequence_state import SequenceState


class TokenProcessor:
    def __init__(self):
        self._tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, payload: Payload) -> tuple[list[int]]:
        payload.input_token_ids = self._tokenizer.encode(payload.message)
        sequence_state = SequenceState(seq_id=payload.chat_id)

        return sequence_state

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids)
