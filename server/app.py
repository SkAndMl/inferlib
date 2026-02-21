from __future__ import annotations

import json
from uuid import uuid4

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from inferlib import InferlibEngine
from server.db_client import DBClient, get_db_client
from server.log import logger
from server.models import Payload

_engine: InferlibEngine | None = None
_db_client: DBClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _db_client
    _engine = InferlibEngine()
    _db_client = await get_db_client()

    await _engine.start()
    await _db_client.initialize()

    yield

    await _engine.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat(payload: Payload):
    logger.info(f"{payload=}")
    if payload.role != "user":
        raise HTTPException(status_code=400, detail="Only user messages are accepted")

    await _db_client.add_message(payload)

    messages = await _db_client.get_messages(payload.chat_id)
    chat_history = [
        {"role": message["role"], "content": message["content"]} for message in messages
    ]
    q = _engine.add_request(chat_history=chat_history, chat_id=payload.chat_id)

    async def _content_generator():
        parts: list[str] = []
        try:
            while True:
                chunk = await q.get()
                if chunk is None:
                    break
                parts.append(chunk)
                yield chunk
        finally:
            content = "".join(parts)
            if content:
                await _db_client.add_message(
                    Payload(
                        chat_id=payload.chat_id,
                        message_id=str(uuid4()),
                        role="assistant",
                        content=content,
                        stream=payload.stream,
                    )
                )

    async def _sse_generator():
        async for chunk in _content_generator():
            data = json.dumps({"message": chunk})
            yield f"data: {data}\n\n"

    if payload.stream:
        return StreamingResponse(_sse_generator(), media_type="text/event-stream")

    content = "".join([chunk async for chunk in _content_generator()])
    return {"message": content}


@app.get("/v1/chats")
async def list_chats(limit: int = 50, offset: int = 0):
    chats = await _db_client.list_chats(limit=limit, offset=offset)
    return {"chats": chats}


@app.get("/v1/chats/{chat_id}")
async def get_chat(chat_id: str):
    chat = await _db_client.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.get("/v1/chats/{chat_id}/messages")
async def get_chat_messages(chat_id: str, limit: int | None = None, offset: int = 0):
    chat = await _db_client.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    messages = await _db_client.get_messages(chat_id=chat_id, limit=limit, offset=offset)
    return {"chat": chat, "messages": messages}


@app.delete("/v1/chats/{chat_id}")
async def delete_chat(chat_id: str):
    deleted = await _db_client.delete_chat(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"deleted": True, "chat_id": chat_id}


@app.get("/")
async def root():
    return FileResponse("server/index.html")


@app.get("/health")
async def health_check():
    return {"status": "200"}
