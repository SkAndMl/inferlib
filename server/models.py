from pydantic import BaseModel


class Payload(BaseModel):
    chat_id: str
    message_id: str
    role: str
    content: str
    stream: bool
