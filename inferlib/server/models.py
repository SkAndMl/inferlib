from pydantic import BaseModel
from typing import Literal


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = True
    temperature: float = 1.0
    max_tokens: int = 4096
    thinking: bool = False


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Literal["stop", "length"]


class UsageStats(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str  # e.g. f"chatcmpl-{uuid4()}"
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageStats


class Delta(BaseModel):
    role: Literal["assistant"] | None = None  # only set on the first chunk
    content: str | None = None  # the decoded token string; None on last chunk


class StreamingChoice(BaseModel):
    index: int
    delta: Delta
    finish_reason: Literal["stop", "length"] | None  # None on all chunks except last


class ChatCompletionChunk(BaseModel):
    id: str  # same id held for all chunks in a response
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamingChoice]
