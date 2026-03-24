from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    stream: bool = True
    temperature: float = Field(default=1.0, ge=0.0)
    max_tokens: int = Field(default=4096, gt=0)
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
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageStats


class Delta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class StreamingChoice(BaseModel):
    index: int
    delta: Delta
    finish_reason: Literal["stop", "length"] | None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamingChoice]


class ErrorBody(BaseModel):
    type: str
    code: str
    message: str


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model: str


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: Literal["inferlib"] = "inferlib"


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class ChatRecord(BaseModel):
    chat_id: str
    title: str
    created_at: int
    updated_at: int
    preview: str | None = None


class MessageRecord(BaseModel):
    message_id: str
    chat_id: str
    role: Literal["system", "user", "assistant"]
    content: str
    created_at: int


class SaveMessageRequest(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    message_id: str | None = None


class SaveMessageResponse(BaseModel):
    saved: Literal[True] = True
    chat_id: str
    message_id: str


class UpdateChatTitleRequest(BaseModel):
    title: str = Field(min_length=1)


class UpdateChatTitleResponse(BaseModel):
    updated: Literal[True] = True
    chat_id: str
    title: str


class DeleteChatResponse(BaseModel):
    deleted: Literal[True] = True
    chat_id: str


class ListChatsResponse(BaseModel):
    chats: list[ChatRecord]


class GetChatResponse(ChatRecord):
    pass


class GetChatMessagesResponse(BaseModel):
    chat: ChatRecord
    messages: list[MessageRecord]
