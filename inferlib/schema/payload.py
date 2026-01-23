from pydantic import BaseModel
from typing import Optional


class Payload(BaseModel):
    chat_id: str
    message: str
    input_token_ids: Optional[list[int]]
    # TODO: integrate these
    # max_tokens: int = 10
    # temperature: float = 1.0
