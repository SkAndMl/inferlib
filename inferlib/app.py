from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI

from inferlib.core.engine import InferlibEngine
from inferlib.schema.payload import Payload

_engine: InferlibEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = InferlibEngine(num_pages=128, page_size=16)
    await _engine.start()
    yield
    await _engine.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/chat")
async def chat(payload: Payload):
    sequence = await _engine(payload)
    return {"message": sequence}
