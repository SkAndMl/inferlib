import aiosqlite
import asyncio
import time

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from inferlib.server.log import logger

CREATE_CHAT_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS chats (
    chat_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
"""

CREATE_MESSAGE_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
);
"""


class DBClient:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        _conn = await aiosqlite.connect(self.db_path)
        try:
            await _conn.execute("PRAGMA foreign_keys = ON;")
            _conn.row_factory = aiosqlite.Row
            yield _conn
            await _conn.commit()
        except Exception as e:
            logger.error(e)
            await _conn.rollback()
            raise
        finally:
            await _conn.close()

    async def initialize(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with self.get_connection() as conn:
            await conn.execute(CREATE_CHAT_TABLE_QUERY)
            await conn.execute(CREATE_MESSAGE_TABLE_QUERY)
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chats_updated_at
                ON chats(updated_at DESC)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_chat_created_at
                ON messages(chat_id, created_at ASC)
                """
            )

    async def add_message(self, chat_id: str, message_id: str, role: str, content: str):
        title = " ".join(content.split()[:4])
        now = int(time.time())

        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT OR IGNORE INTO chats (chat_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, title, now, now),
            )

            await conn.execute(
                "UPDATE chats SET updated_at = ? WHERE chat_id = ?",
                (now, chat_id),
            )

            await conn.execute(
                """
                INSERT INTO messages (message_id, chat_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (message_id, chat_id, role, content, now),
            )

    async def get_chat(self, chat_id: str) -> dict[str, Any] | None:
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT chat_id, title, created_at, updated_at
                FROM chats
                WHERE chat_id = ?
                """,
                (chat_id,),
            )
            row = await cursor.fetchone()
            return None if row is None else dict(row)

    async def list_chats(
        self, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT
                    chats.chat_id,
                    chats.title,
                    chats.created_at,
                    chats.updated_at,
                    (
                        SELECT messages.content
                        FROM messages
                        WHERE messages.chat_id = chats.chat_id
                        ORDER BY messages.created_at DESC
                        LIMIT 1
                    ) AS preview
                FROM chats
                ORDER BY chats.updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_messages(
        self, chat_id: str, limit: int | None = None, offset: int = 0
    ) -> list[dict[str, Any]]:
        async with self.get_connection() as conn:
            if limit is None:
                cursor = await conn.execute(
                    """
                    SELECT message_id, chat_id, role, content, created_at
                    FROM messages
                    WHERE chat_id = ?
                    ORDER BY created_at ASC
                    """,
                    (chat_id,),
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT message_id, chat_id, role, content, created_at
                    FROM messages
                    WHERE chat_id = ?
                    ORDER BY created_at ASC
                    LIMIT ? OFFSET ?
                    """,
                    (chat_id, limit, offset),
                )

            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def update_chat_title(self, chat_id: str, title: str) -> bool:
        now = int(time.time())
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                """
                UPDATE chats
                SET title = ?, updated_at = ?
                WHERE chat_id = ?
                """,
                (title, now, chat_id),
            )
            return cursor.rowcount > 0

    async def delete_chat(self, chat_id: str) -> bool:
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM chats WHERE chat_id = ?",
                (chat_id,),
            )
            return cursor.rowcount > 0


_db_client: DBClient | None = None
_db_lock = asyncio.Lock()


async def get_db_client() -> DBClient:
    from inferlib.server.config import DB_PATH

    global _db_client
    if _db_client is not None:
        return _db_client

    async with _db_lock:
        db_path = Path(DB_PATH).expanduser().resolve()
        if db_path.is_dir():
            db_path = db_path / "chats.sqlite3"
        _db_client = DBClient(db_path)

    return _db_client


__all__ = ["DBClient", "get_db_client"]
