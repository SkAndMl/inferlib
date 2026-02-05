import asyncio
import tiktoken
import torch

from asyncio import Future
from typing import Any

from inferlib.models import GPT2

from inferlib.engine.runner import Runner
from inferlib.engine.scheduler import Scheduler
from inferlib.engine.sequence import Sequence
from inferlib.engine.page import PageManager
from inferlib.log import logger


class InferlibEngine:
    def __init__(self):
        self.llm = GPT2.from_pretrained("large")
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.model_config = self.llm.config
        self.page_manager = PageManager(
            num_pages=128,
            num_layers=self.model_config.n_layer,
            num_heads=self.model_config.n_head,
            page_size=16,
            head_dim=self.model_config.n_embd // self.model_config.n_head,
            dtype=torch.float16,
            device=torch.device("cpu"),
        )
        self.scheduler = Scheduler(page_manager=self.page_manager, batch_size=1)
        self.runner = Runner(llm=self.llm, page_manager=self.page_manager)

        self._sequence_to_future: dict[str, Future] = {}
        self._task = None
        self._lock = asyncio.Lock()

    def add_request(self, payload: dict[str, Any]) -> Future:
        if self._task is None:
            raise RuntimeError("Engine not started")

        assert "message" in payload
        assert "id" in payload
        prompt_tokens: list[int] = self.tokenizer.encode(payload["message"])
        sequence = Sequence(
            s_id=payload["id"],
            prompt_tokens=prompt_tokens,
            completion_tokens=[],
            eos_token_id=self.tokenizer.eot_token,
            max_tokens=10,
        )
        self.scheduler.add_request(sequence)
        future = Future()
        self._sequence_to_future[payload["id"]] = future
        return future

    async def start(self):
        async with self._lock:
            if self._task is None:
                self._task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        async with self._lock:
            if self._task is None:
                return

            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

            self._task = None

    async def _worker_loop(self):
        while True:
            try:
                batch = self.scheduler.schedule()
                if not batch:
                    await asyncio.sleep(0.1)
                    continue

                await asyncio.to_thread(
                    self.runner.run, batch
                )  # updates sequence inplace
                self.scheduler.update(batch)

                finished_sequences = self.scheduler.get_finished_sequences()
                if finished_sequences:
                    for sequence in finished_sequences:
                        response = self.tokenizer.decode(sequence.completion_tokens)
                        self._sequence_to_future[sequence.s_id].set_result(response)
                        del self._sequence_to_future[sequence.s_id]

            except Exception as e:
                logger.error(e)
