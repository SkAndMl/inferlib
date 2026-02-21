import asyncio
import torch

from asyncio import Queue
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Any

from inferlib.models import Qwen3

from inferlib.engine.runner import Runner
from inferlib.engine.scheduler import Scheduler
from inferlib.engine.sequence import Sequence
from inferlib.engine.page import PageManager
from inferlib.log import logger


class InferlibEngine:
    def __init__(self):
        self.llm, self.model_config = Qwen3.from_pretrained("Qwen/Qwen3-0.6B")
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B"
        )
        self.page_manager = PageManager(
            num_pages=128,
            num_layers=self.model_config.num_hidden_layers,
            num_heads=self.model_config.num_key_value_heads,
            page_size=16,
            head_dim=self.model_config.head_dim,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.scheduler = Scheduler(page_manager=self.page_manager, batch_size=4)
        self.runner = Runner(llm=self.llm, page_manager=self.page_manager)

        self._sequence_to_queue: dict[str, Queue] = {}
        self._task = None
        self._lock = asyncio.Lock()

        self._eot_token = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def add_request(self, payload: dict[str, Any]) -> Queue:
        if self._task is None:
            raise RuntimeError("Engine not started")

        assert "chat_history" in payload
        assert "id" in payload
        prompt_tokens: list[int] = self.tokenizer.apply_chat_template(
            conversation=payload["chat_history"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        sequence = Sequence(
            s_id=payload["id"],
            prompt_tokens=prompt_tokens,
            completion_tokens=[],
            eos_token_id=self._eot_token,
            max_tokens=4096,
        )
        self.scheduler.add_request(sequence)
        q = Queue()
        self._sequence_to_queue[payload["id"]] = q
        return q

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
                batch = await self.scheduler.schedule()
                if not batch:
                    await asyncio.sleep(0.1)
                    continue

                await asyncio.to_thread(
                    self.runner.run, batch
                )  # updates sequence inplace
                self.scheduler.update(batch)

                for sequence in batch:
                    if sequence.is_finished:
                        await self._sequence_to_queue[sequence.s_id].put(
                            self.tokenizer.decode(
                                [sequence.last_token_id], skip_special_tokens=True
                            )
                        )
                        await self._sequence_to_queue[sequence.s_id].put(None)
                        del self._sequence_to_queue[sequence.s_id]
                    else:
                        await self._sequence_to_queue[sequence.s_id].put(
                            self.tokenizer.decode(
                                [sequence.last_token_id], skip_special_tokens=True
                            )
                        )

            except Exception as e:
                logger.error(e)
