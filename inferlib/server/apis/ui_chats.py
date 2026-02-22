from __future__ import annotations

from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from inferlib.server.db_client import get_db_client

router = APIRouter()


class SaveMessageRequest(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    message_id: str | None = None


class UpdateChatTitleRequest(BaseModel):
    title: str


@router.get("/v1/chats")
async def list_chats(limit: int = 50, offset: int = 0):
    db_client = await get_db_client()
    chats = await db_client.list_chats(limit=limit, offset=offset)
    return {"chats": chats}


@router.get("/v1/chats/{chat_id}")
async def get_chat(chat_id: str):
    db_client = await get_db_client()
    chat = await db_client.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.post("/v1/chats/{chat_id}/messages")
async def save_message(chat_id: str, payload: SaveMessageRequest):
    db_client = await get_db_client()
    message_id = payload.message_id or str(uuid4())
    await db_client.add_message(
        chat_id=chat_id,
        message_id=message_id,
        role=payload.role,
        content=payload.content,
    )
    return {"saved": True, "chat_id": chat_id, "message_id": message_id}


@router.get("/v1/chats/{chat_id}/messages")
async def get_chat_messages(chat_id: str, limit: int | None = None, offset: int = 0):
    db_client = await get_db_client()
    chat = await db_client.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    messages = await db_client.get_messages(chat_id=chat_id, limit=limit, offset=offset)
    return {"chat": chat, "messages": messages}


@router.patch("/v1/chats/{chat_id}")
async def update_chat_title(chat_id: str, payload: UpdateChatTitleRequest):
    db_client = await get_db_client()
    updated = await db_client.update_chat_title(chat_id=chat_id, title=payload.title)
    if not updated:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"updated": True, "chat_id": chat_id, "title": payload.title}


@router.delete("/v1/chats/{chat_id}")
async def delete_chat(chat_id: str):
    db_client = await get_db_client()
    deleted = await db_client.delete_chat(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"deleted": True, "chat_id": chat_id}
