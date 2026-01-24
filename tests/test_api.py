import asyncio
import httpx

CHAT_URL = "http://localhost:8000/chat"


async def test_chat_endpoint(payload: dict):
    async with httpx.AsyncClient() as client:
        response = await client.post(url=CHAT_URL, json=payload)
        response.raise_for_status()
        print(response.json()["message"])


async def main():
    payloads = [
        {"chat_id": str(i), "message": m, "input_token_ids": None}
        for i, m in enumerate(["hello", "i want to", "my name is", "how are"])
    ]
    await asyncio.gather(*[test_chat_endpoint(payload) for payload in payloads])


asyncio.run(main())
