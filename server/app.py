from __future__ import annotations

import json

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from inferlib.engine.engine import InferlibEngine
from inferlib.log import logger

_engine: InferlibEngine | None = None


class Payload(BaseModel):
    id: str
    session_id: str
    chat_history: list[dict[str, str]]
    stream: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = InferlibEngine()
    await _engine.start()
    yield
    await _engine.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat(payload: Payload):
    logger.info(f"{payload=}")
    q = _engine.add_request(payload.model_dump())

    async def _generator():
        while True:
            chunk = await q.get()
            if chunk is None:
                break
            chunk = json.dumps({"message": chunk})
            yield f"data: {chunk}\n\n"

    if payload.stream:
        return StreamingResponse(_generator(), media_type="text/event-stream")

    content = "".join(
        [
            json.loads(chunk[len("data: ") : -2])["message"]
            async for chunk in _generator()
        ]
    )
    return {"message": content}


@app.get("/")
async def root():
    return FileResponse("server/index.html")


@app.get("/health")
async def health_check():
    return {"status": "200"}
