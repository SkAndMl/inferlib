import asyncio
import torch

from asyncio import Queue
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from inferlib.core.models import Qwen3

from inferlib.core.engine.runner import Runner
from inferlib.core.engine.scheduler import Scheduler
from inferlib.core.engine.sequence import Sequence
from inferlib.core.engine.page import PageManager
from inferlib.core.log import logger


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

    @staticmethod
    def _normalize_token_ids(tokens: object) -> list[int]:
        # Handle common tokenizer return types:
        # - list[int]
        # - tokenizers.Encoding (has `.ids`)
        # - list[tokenizers.Encoding]
        # - BatchEncoding-like objects (has `.input_ids`)
        if isinstance(tokens, list):
            if not tokens:
                return []

            if all(isinstance(token, int) for token in tokens):
                return [int(token) for token in tokens]

            if all(hasattr(token, "ids") for token in tokens):
                return [int(token_id) for token in tokens for token_id in token.ids]

            if all(isinstance(token, list) for token in tokens):
                return [int(token_id) for token_list in tokens for token_id in token_list]

        if hasattr(tokens, "ids"):
            return [int(token_id) for token_id in tokens.ids]

        if hasattr(tokens, "input_ids"):
            input_ids = tokens.input_ids
            if isinstance(input_ids, list):
                if not input_ids:
                    return []
                if all(isinstance(token, int) for token in input_ids):
                    return [int(token) for token in input_ids]
                if all(isinstance(token, list) for token in input_ids):
                    return [
                        int(token_id) for token_list in input_ids for token_id in token_list
                    ]

        raise TypeError(f"Unsupported tokenizer output type: {type(tokens)!r}")

    def add_request(
        self,
        chat_history: list[dict[str, str]],
        chat_id: str,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> Queue:
        if self._task is None:
            raise RuntimeError("Engine not started")

        raw_prompt_tokens = self.tokenizer.apply_chat_template(
            conversation=chat_history,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = self._normalize_token_ids(raw_prompt_tokens)
        sequence = Sequence(
            s_id=chat_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=[],
            eos_token_id=self._eot_token,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.scheduler.add_request(sequence)
        q = Queue()
        self._sequence_to_queue[chat_id] = q
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
                    new_text = self.tokenizer.decode(
                        sequence.completion_tokens, skip_special_tokens=True
                    )
                    if sequence.is_finished:
                        await self._sequence_to_queue[sequence.s_id].put(
                            (
                                new_text[len(sequence.last_text) :],
                                sequence.finish_reason,
                            )
                        )
                        del self._sequence_to_queue[sequence.s_id]
                    else:
                        await self._sequence_to_queue[sequence.s_id].put(
                            (
                                new_text[len(sequence.last_text) :],
                                sequence.finish_reason,
                            )
                        )
                        sequence.last_text = new_text

            except Exception as e:
                logger.error(e)
