from __future__ import annotations

import asyncio
import time
from uuid import uuid4

from contextlib import asynccontextmanager
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from typing import AsyncIterator

from inferlib import InferlibEngine
from server.log import logger
from server.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Delta,
    Message,
    StreamingChoice,
)

_engine: InferlibEngine | None = None


@asynccontextmanager
async def lifespan(app: APIRouter):
    global _engine
    _engine = InferlibEngine()
    await _engine.start()
    yield
    await _engine.stop()


router = APIRouter(lifespan=lifespan)


async def _content_generator(
    chat_id: str, model: str, q: asyncio.Queue
) -> AsyncIterator[ChatCompletionChunk | str]:
    parts: list[str] = []
    first_chunk: bool = True

    def _delta_generator(content: str | None) -> Delta:
        nonlocal first_chunk
        delta = Delta(content=content)
        if first_chunk:
            delta.role = "assistant"
            first_chunk = False
        return delta

    while True:
        # TODO: update q to return finish reason
        content, finish_reason = await q.get()
        delta = _delta_generator(content)
        yield ChatCompletionChunk(
            id=chat_id,
            model=model,
            created=int(time.time()),
            choices=[StreamingChoice(index=0, delta=delta, finish_reason=None)],
        )
        if finish_reason is not None:
            yield ChatCompletionChunk(
                id=chat_id,
                model=model,
                created=int(time.time()),
                choices=[
                    StreamingChoice(index=0, delta=Delta(), finish_reason=finish_reason)
                ],
            )
            yield "[DONE]"
            break

        parts.append(content)


async def _sse_wrapper(generator) -> AsyncIterator[str]:
    async for chunk in generator:
        chunk_str = chunk
        if isinstance(chunk, ChatCompletionChunk):
            chunk_str = chunk.model_dump_json(exclude_none=True)

        yield f"data: {chunk_str}\n\n"


@router.post("/v1/chat/completions")
async def chat(payload: ChatCompletionRequest):
    logger.info(f"{payload=}")
    chat_id = f"chatcmpl-{str(uuid4())}"
    chat_history = [m.model_dump() for m in payload.messages]
    q = _engine.add_request(chat_history=chat_history, chat_id=chat_id)

    if payload.stream:
        return StreamingResponse(
            _sse_wrapper(_content_generator(model=payload.model, chat_id=chat_id, q=q)),
            media_type="text/event-stream",
        )

    streaming_choices = [
        chunk.choices[0]
        async for chunk in _content_generator(model=payload.model, chat_id=chat_id, q=q)
        if isinstance(chunk, ChatCompletionChunk)
    ]
    finish_reason = streaming_choices[-1].finish_reason
    content = "".join(
        [choice.delta.content for choice in streaming_choices if choice.delta.content]
    )

    chat_completion_response = ChatCompletionResponse(
        id=chat_id,
        created=int(time.time()),
        model=payload.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
    )
    return chat_completion_response
