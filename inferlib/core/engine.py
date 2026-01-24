import torch

from functools import partial

from inferlib.cache import PagedKVCache, PagePool
from inferlib.models import GPT2
from inferlib.scheduler import FCFSScheduler
from inferlib.schema.payload import Payload

from .token_processor import TokenProcessor


class InferlibEngine:
    def __init__(
        self,
        num_pages: int,
        page_size: int,
    ):
        self._token_processor = TokenProcessor()
        self._llm = GPT2.from_pretrained("small")
        self._page_pool = PagePool(
            num_pages=num_pages,
            page_size=page_size,
            num_layers=self._llm.config.n_layer,
            num_heads=self._llm.config.n_head,
            dtype=torch.float32,
            head_dim=self._llm.config.n_embd // self._llm.config.n_head,
            device="cpu",
        )
        self._cache = PagedKVCache(
            page_pool=self._page_pool,
            num_layers=self._llm.config.n_layer,
            page_size=page_size,
            device="cpu",
        )
        self._scheduler = FCFSScheduler(
            fn_to_call=partial(self._llm.generate, cache=self._cache)
        )

    async def start(self):
        await self._scheduler.start()

    async def stop(self):
        await self._scheduler.stop()

    async def __call__(self, payload: Payload) -> str:
        sequence_state = self._token_processor.encode(payload)
        output_tokens = await self._scheduler.submit(
            payload=payload, sequence_state=sequence_state
        )
        return self._token_processor.decode(output_tokens)
