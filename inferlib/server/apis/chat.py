from __future__ import annotations

import asyncio
import time

from typing import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from inferlib.core import InferlibEngine
from inferlib.types import SamplingParams
from inferlib.server.errors import invalid_request
from inferlib.server.log import logger
from inferlib.server.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Delta,
    Message,
    StreamingChoice,
    UsageStats,
)


router = APIRouter()


def _request_validation_code(message: str) -> str:
    if "max_model_len" in message or "pages per request" in message:
        return "context_length_exceeded"
    if "prompt" in message:
        return "invalid_prompt"
    return "invalid_request"


async def _sse_wrapper(generator: AsyncIterator[ChatCompletionChunk | str]) -> AsyncIterator[str]:
    async for chunk in generator:
        chunk_str = chunk
        if isinstance(chunk, ChatCompletionChunk):
            chunk_str = chunk.model_dump_json(exclude_none=True)

        yield f"data: {chunk_str}\n\n"


def _build_prompt_token_ids(request: Request, payload: ChatCompletionRequest) -> list[int]:
    engine = request.app.state.engine
    raw_prompt_tokens = engine.tokenizer.apply_chat_template(
        conversation=[message.model_dump() for message in payload.messages],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=payload.thinking,
    )
    return InferlibEngine._normalize_token_ids(raw_prompt_tokens)


def _build_sampling_params(payload: ChatCompletionRequest) -> SamplingParams:
    return SamplingParams(
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        thinking=payload.thinking,
    )


def _log_request_completed(
    *,
    request_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str,
    started_at: float,
    first_token_at: float | None,
) -> None:
    duration_seconds = max(time.perf_counter() - started_at, 1e-9)
    ttft_ms = None if first_token_at is None else round((first_token_at - started_at) * 1000, 2)
    logger.info(
        "chat request completed",
        extra={
            "request_id": request_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "duration_ms": round(duration_seconds * 1000, 2),
            "ttft_ms": ttft_ms,
            "tokens_per_second": round(completion_tokens / duration_seconds, 2),
        },
    )


async def _content_generator(
    *,
    request_id: str,
    model: str,
    prompt_tokens: int,
    queue: asyncio.Queue,
    started_at: float,
) -> AsyncIterator[ChatCompletionChunk | str]:
    first_chunk = True
    completion_tokens = 0
    first_token_at: float | None = None

    while True:
        event = await queue.get()
        if event.error is not None:
            raise RuntimeError(event.error)

        if first_token_at is None and event.token_id is not None:
            first_token_at = time.perf_counter()
        if event.token_id is not None:
            completion_tokens += 1

        delta = Delta(content=event.text or None)
        if first_chunk:
            delta.role = "assistant"
            first_chunk = False

        yield ChatCompletionChunk(
            id=request_id,
            model=model,
            created=int(time.time()),
            choices=[StreamingChoice(index=0, delta=delta, finish_reason=None)],
        )

        if event.finish_reason is not None:
            yield ChatCompletionChunk(
                id=request_id,
                model=model,
                created=int(time.time()),
                choices=[StreamingChoice(index=0, delta=Delta(), finish_reason=event.finish_reason)],
            )
            _log_request_completed(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=event.finish_reason,
                started_at=started_at,
                first_token_at=first_token_at,
            )
            yield "[DONE]"
            return


async def _build_chat_completion_response(
    *,
    request_id: str,
    model: str,
    prompt_tokens: int,
    queue: asyncio.Queue,
    started_at: float,
) -> ChatCompletionResponse:
    completion_token_ids: list[int] = []
    text_parts: list[str] = []
    finish_reason: str | None = None
    first_token_at: float | None = None

    while finish_reason is None:
        event = await queue.get()
        if event.error is not None:
            raise RuntimeError(event.error)
        if first_token_at is None and event.token_id is not None:
            first_token_at = time.perf_counter()
        if event.token_id is not None:
            completion_token_ids.append(event.token_id)
        if event.text:
            text_parts.append(event.text)
        finish_reason = event.finish_reason

    _log_request_completed(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=len(completion_token_ids),
        finish_reason=finish_reason,
        started_at=started_at,
        first_token_at=first_token_at,
    )

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content="".join(text_parts)),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=len(completion_token_ids),
            total_tokens=prompt_tokens + len(completion_token_ids),
        ),
    )


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat(request: Request, payload: ChatCompletionRequest):
    engine = request.app.state.engine
    request_id = f"chatcmpl-{uuid4()}"
    started_at = time.perf_counter()

    if payload.model != engine.model_class:
        raise invalid_request(
            code="model_not_loaded",
            message=f"Requested model {payload.model!r} is not loaded; loaded model is {engine.model_class!r}.",
        )

    prompt_token_ids = _build_prompt_token_ids(request, payload)
    sampling_params = _build_sampling_params(payload)
    validation_error = engine.validate_request(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    if validation_error is not None:
        logger.warning(
            "chat request rejected",
            extra={
                "request_id": request_id,
                "model": payload.model,
                "prompt_tokens": len(prompt_token_ids),
                "max_tokens": payload.max_tokens,
                "reason": validation_error,
            },
        )
        raise invalid_request(
            code=_request_validation_code(validation_error),
            message=validation_error,
        )

    logger.info(
        "chat request accepted",
        extra={
            "request_id": request_id,
            "model": payload.model,
            "prompt_tokens": len(prompt_token_ids),
            "max_tokens": payload.max_tokens,
            "stream": payload.stream,
        },
    )
    queue = engine.add_request(
        prompt_token_ids=prompt_token_ids,
        request_id=request_id,
        sampling_params=sampling_params,
    )

    if payload.stream:
        return StreamingResponse(
            _sse_wrapper(
                _content_generator(
                    request_id=request_id,
                    model=engine.model_class,
                    prompt_tokens=len(prompt_token_ids),
                    queue=queue,
                    started_at=started_at,
                )
            ),
            media_type="text/event-stream",
        )

    return await _build_chat_completion_response(
        request_id=request_id,
        model=engine.model_class,
        prompt_tokens=len(prompt_token_ids),
        queue=queue,
        started_at=started_at,
    )
