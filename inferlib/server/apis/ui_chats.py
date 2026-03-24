from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Request

from inferlib.server.errors import not_found
from inferlib.server.models import (
    DeleteChatResponse,
    GetChatMessagesResponse,
    GetChatResponse,
    ListChatsResponse,
    SaveMessageRequest,
    SaveMessageResponse,
    UpdateChatTitleRequest,
    UpdateChatTitleResponse,
)


router = APIRouter()


@router.get("/v1/chats", response_model=ListChatsResponse)
async def list_chats(request: Request, limit: int = 50, offset: int = 0) -> ListChatsResponse:
    chats = await request.app.state.db_client.list_chats(limit=limit, offset=offset)
    return ListChatsResponse(chats=chats)


@router.get("/v1/chats/{chat_id}", response_model=GetChatResponse)
async def get_chat(request: Request, chat_id: str) -> GetChatResponse:
    chat = await request.app.state.db_client.get_chat(chat_id)
    if chat is None:
        raise not_found(code="chat_not_found", message="Chat not found")
    return GetChatResponse(**chat)


@router.post("/v1/chats/{chat_id}/messages", response_model=SaveMessageResponse)
async def save_message(request: Request, chat_id: str, payload: SaveMessageRequest) -> SaveMessageResponse:
    message_id = payload.message_id or str(uuid4())
    await request.app.state.db_client.add_message(
        chat_id=chat_id,
        message_id=message_id,
        role=payload.role,
        content=payload.content,
    )
    return SaveMessageResponse(chat_id=chat_id, message_id=message_id)


@router.get("/v1/chats/{chat_id}/messages", response_model=GetChatMessagesResponse)
async def get_chat_messages(
    request: Request,
    chat_id: str,
    limit: int | None = None,
    offset: int = 0,
) -> GetChatMessagesResponse:
    chat = await request.app.state.db_client.get_chat(chat_id)
    if chat is None:
        raise not_found(code="chat_not_found", message="Chat not found")

    messages = await request.app.state.db_client.get_messages(chat_id=chat_id, limit=limit, offset=offset)
    return GetChatMessagesResponse(chat=chat, messages=messages)


@router.patch("/v1/chats/{chat_id}", response_model=UpdateChatTitleResponse)
async def update_chat_title(
    request: Request,
    chat_id: str,
    payload: UpdateChatTitleRequest,
) -> UpdateChatTitleResponse:
    updated = await request.app.state.db_client.update_chat_title(chat_id=chat_id, title=payload.title)
    if not updated:
        raise not_found(code="chat_not_found", message="Chat not found")
    return UpdateChatTitleResponse(chat_id=chat_id, title=payload.title)


@router.delete("/v1/chats/{chat_id}", response_model=DeleteChatResponse)
async def delete_chat(request: Request, chat_id: str) -> DeleteChatResponse:
    deleted = await request.app.state.db_client.delete_chat(chat_id)
    if not deleted:
        raise not_found(code="chat_not_found", message="Chat not found")
    return DeleteChatResponse(chat_id=chat_id)
