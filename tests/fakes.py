from __future__ import annotations

import asyncio

from types import SimpleNamespace


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def apply_chat_template(
        self,
        *,
        conversation: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        del tokenize, add_generation_prompt
        rendered = "\n".join(f"{message['role']}:{message['content']}" for message in conversation)
        if enable_thinking:
            rendered = f"{rendered}<thinking>"
        return [ord(char) for char in rendered]


class FakeServerEngine:
    def __init__(
        self,
        model_class: str,
        *,
        page_size: int = 32,
        batch_size: int = 4,
        max_active_sequences: int = 8,
        memory_limit: int = 4 * 1024 * 1024 * 1024,
    ) -> None:
        del page_size, batch_size, max_active_sequences, memory_limit
        self.model_class = model_class
        self.tokenizer = FakeTokenizer()
        self.max_model_len = 64

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def validate_request(self, *, prompt_token_ids: list[int], sampling_params) -> str | None:
        total_tokens = len(prompt_token_ids) + sampling_params.max_tokens
        if not prompt_token_ids:
            return "prompt must produce at least one token"
        if total_tokens > self.max_model_len:
            return (
                f"prompt tokens + max_tokens exceed max_model_len "
                f"({total_tokens} > {self.max_model_len})"
            )
        return None

    def add_request(self, *, prompt_token_ids: list[int], request_id: str, sampling_params):
        del prompt_token_ids, request_id, sampling_params
        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait(SimpleNamespace(text="hello", token_id=101, finish_reason=None, error=None))
        queue.put_nowait(SimpleNamespace(text=" world", token_id=102, finish_reason="length", error=None))
        return queue


class FakePublicEngine:
    instances: list["FakePublicEngine"] = []

    def __init__(self, model_class: str, **kwargs) -> None:
        self.model_class = model_class
        self.kwargs = kwargs
        self.tokenizer = FakeTokenizer()
        self.received_prompts: list[list[int]] = []
        self.received_sampling_params = []
        FakePublicEngine.instances.append(self)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def generate(self, *, prompt_token_batches, sampling_params, use_tqdm: bool = False):
        del use_tqdm
        self.received_prompts = prompt_token_batches
        self.received_sampling_params = sampling_params
        from inferlib.types import GenerationOutput

        return [
            GenerationOutput(
                text=f"out-{index}",
                token_ids=[index],
                finish_reason="length",
                prompt_tokens=len(prompt),
                completion_tokens=1,
            )
            for index, prompt in enumerate(prompt_token_batches)
        ]
