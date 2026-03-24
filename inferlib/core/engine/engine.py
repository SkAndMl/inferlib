import asyncio
import math
import torch

from asyncio import Queue
from dataclasses import dataclass
from uuid import uuid4
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from inferlib.core.models import Qwen3

from inferlib.core.engine.runner import Runner
from inferlib.core.engine.scheduler import Scheduler
from inferlib.core.engine.sequence import Sequence
from inferlib.core.engine.page import PageManager
from inferlib.core.log import logger
from inferlib.types import GenerationOutput, SamplingParams


@dataclass(slots=True)
class _GenerationEvent:
    text: str = ""
    token_id: int | None = None
    finish_reason: str | None = None
    error: str | None = None


class InferlibEngine:
    def __init__(
        self,
        model_class: str,
        page_size: int = 32,
        batch_size: int = 4,
        max_active_sequences: int = 8,
        memory_limit: int = 4 * 1024 * 1024 * 1024,  # 4 gb
        max_model_len: int | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.model_class = model_class
        self.llm, self.model_config = Qwen3.from_pretrained(model_class)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_class)

        self.num_pages = self.calc_num_pages(
            dtype,
            num_layers=self.model_config.num_hidden_layers,
            num_heads=self.model_config.num_key_value_heads,
            head_dim=self.model_config.head_dim,
            page_size=page_size,
            memory_limit=memory_limit,
        )
        memory_bounded_pages = max(1, self.num_pages // max_active_sequences)
        max_pages_by_model_len = math.ceil(
            (max_model_len or self.num_pages * page_size) / page_size
        )
        self.max_model_len = min(self.num_pages * page_size, max_model_len or self.num_pages * page_size)
        self.max_pages_per_sequence = max(1, min(memory_bounded_pages, max_pages_by_model_len))
        self.page_manager = PageManager(
            num_pages=self.num_pages,
            num_layers=self.model_config.num_hidden_layers,
            num_heads=self.model_config.num_key_value_heads,
            page_size=page_size,
            head_dim=self.model_config.head_dim,
            max_pages_per_sequence=self.max_pages_per_sequence,
            dtype=dtype,
            device=torch.device("cpu"),
        )
        self.request_event = asyncio.Event()
        self.scheduler = Scheduler(
            page_manager=self.page_manager,
            request_event=self.request_event,
            max_pages_per_sequence=self.max_pages_per_sequence,
            batch_size=batch_size,
            max_active_sequences=max_active_sequences,
        )
        self.runner = Runner(llm=self.llm, page_manager=self.page_manager)

        self._sequence_to_queue: dict[str, Queue] = {}
        self._task = None
        self._lock = asyncio.Lock()

        self._eot_token = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        logger.info(
            "engine initialized",
            extra={
                "model_class": self.model_class,
                "page_size": page_size,
                "batch_size": batch_size,
                "max_active_sequences": max_active_sequences,
                "memory_limit_bytes": memory_limit,
                "num_pages": self.num_pages,
                "max_model_len": self.max_model_len,
                "max_pages_per_sequence": self.max_pages_per_sequence,
            },
        )

    @staticmethod
    def calc_num_pages(
        dtype: torch.dtype,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int,
        memory_limit: int,
    ) -> int:
        dtype_to_memory = {torch.float16: 2, torch.float32: 4}
        if dtype not in dtype_to_memory:
            raise ValueError(f"Unsupported dtype: {dtype}")
        per_page_bytes = (
            num_layers * num_heads * page_size * head_dim * dtype_to_memory[dtype] * 2
        )
        if per_page_bytes > memory_limit:
            raise ValueError("memory_limit is too small for a single page")
        return math.floor(memory_limit / per_page_bytes)

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
                return [
                    int(token_id) for token_list in tokens for token_id in token_list
                ]

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
                        int(token_id)
                        for token_list in input_ids
                        for token_id in token_list
                    ]

        raise TypeError(f"Unsupported tokenizer output type: {type(tokens)!r}")

    def validate_request(
        self,
        *,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> str | None:
        if not prompt_token_ids:
            return "prompt must produce at least one token"
        total_requested_tokens = len(prompt_token_ids) + sampling_params.max_tokens
        if total_requested_tokens > self.max_model_len:
            return (
                f"prompt tokens + max_tokens exceed max_model_len "
                f"({total_requested_tokens} > {self.max_model_len})"
            )

        prompt_pages = math.ceil(len(prompt_token_ids) / self.page_manager.page_size)
        if prompt_pages > self.max_pages_per_sequence:
            return (
                f"prompt requires {prompt_pages} pages but this server allows "
                f"at most {self.max_pages_per_sequence} pages per request"
            )
        return None

    def add_request(
        self,
        *,
        prompt_token_ids: list[int],
        request_id: str,
        sampling_params: SamplingParams,
    ) -> Queue:
        if self._task is None or self._task.done():
            raise RuntimeError("Engine not started")

        validation_error = self.validate_request(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )
        if validation_error is not None:
            raise ValueError(validation_error)

        sequence = Sequence(
            s_id=request_id,
            prompt_tokens=prompt_token_ids,
            completion_tokens=[],
            eos_token_id=None if sampling_params.ignore_eos else self._eot_token,
            max_tokens=sampling_params.max_tokens,
            max_total_tokens=self.max_model_len,
            temperature=sampling_params.temperature,
        )
        added: bool = self.scheduler.add_request(sequence)
        if not added:
            raise ValueError("request exceeds current sequence page budget")
        q = Queue()
        self._sequence_to_queue[request_id] = q
        return q

    async def start(self):
        async with self._lock:
            if self._task is None or self._task.done():
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

    async def generate(
        self,
        *,
        prompt_token_batches: list[list[int]],
        sampling_params: list[SamplingParams],
        use_tqdm: bool = False,
    ) -> list[GenerationOutput]:
        if not prompt_token_batches:
            return []

        if len(prompt_token_batches) != len(sampling_params):
            raise ValueError("sampling_params must match prompt batch length")

        await self.start()

        for prompt_tokens, params in zip(prompt_token_batches, sampling_params):
            validation_error = self.validate_request(
                prompt_token_ids=prompt_tokens,
                sampling_params=params,
            )
            if validation_error is not None:
                raise ValueError(validation_error)

        requests: list[tuple[int, int, Queue]] = []
        for index, (prompt_tokens, params) in enumerate(zip(prompt_token_batches, sampling_params)):
            request_id = f"gen-{uuid4()}"
            queue = self.add_request(
                prompt_token_ids=prompt_tokens,
                request_id=request_id,
                sampling_params=params,
            )
            requests.append((index, len(prompt_tokens), queue))

        if not use_tqdm:
            outputs = await asyncio.gather(
                *[
                    self._collect_output(index=index, prompt_tokens=prompt_tokens, queue=queue)
                    for index, prompt_tokens, queue in requests
                ]
            )
            return [output for _, output in sorted(outputs, key=lambda item: item[0])]

        try:
            from tqdm import tqdm
        except ImportError:
            outputs = await asyncio.gather(
                *[
                    self._collect_output(index=index, prompt_tokens=prompt_tokens, queue=queue)
                    for index, prompt_tokens, queue in requests
                ]
            )
            return [output for _, output in sorted(outputs, key=lambda item: item[0])]

        tasks = [
            asyncio.create_task(self._collect_output(index=index, prompt_tokens=prompt_tokens, queue=queue))
            for index, prompt_tokens, queue in requests
        ]
        gathered: list[tuple[int, GenerationOutput]] = []
        with tqdm(total=len(tasks)) as progress:
            for task in asyncio.as_completed(tasks):
                gathered.append(await task)
                progress.update(1)
        return [output for _, output in sorted(gathered, key=lambda item: item[0])]

    async def _collect_output(
        self,
        *,
        index: int,
        prompt_tokens: int,
        queue: Queue,
    ) -> tuple[int, GenerationOutput]:
        text_parts: list[str] = []
        token_ids: list[int] = []
        finish_reason: str | None = None

        while finish_reason is None:
            event: _GenerationEvent = await queue.get()
            if event.error is not None:
                raise RuntimeError(event.error)
            if event.text:
                text_parts.append(event.text)
            if event.token_id is not None:
                token_ids.append(event.token_id)
            finish_reason = event.finish_reason

        return (
            index,
            GenerationOutput(
                text="".join(text_parts),
                token_ids=token_ids,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(token_ids),
            ),
        )

    async def _worker_loop(self):
        while True:
            try:
                batch = await self.scheduler.schedule()
                if not batch:
                    if self.scheduler.prefill_empty and self.scheduler.decode_empty:
                        self.request_event.clear()
                        await self.request_event.wait()
                    else:
                        await asyncio.sleep(0.05)  # no pages available, retry shortly
                    continue

                await asyncio.to_thread(
                    self.runner.run, batch
                )  # updates sequence inplace
                self.scheduler.update(batch)

                for sequence in batch:
                    new_text = self.tokenizer.decode(
                        sequence.completion_tokens, skip_special_tokens=True
                    )
                    token_id = sequence.completion_tokens[-1]
                    delta = new_text[len(sequence.last_text) :]
                    event = _GenerationEvent(
                        text=delta,
                        token_id=token_id,
                        finish_reason=sequence.finish_reason,
                    )
                    if sequence.is_finished:
                        await self._sequence_to_queue[sequence.s_id].put(event)
                        del self._sequence_to_queue[sequence.s_id]
                    else:
                        await self._sequence_to_queue[sequence.s_id].put(event)
                        sequence.last_text = new_text

            except Exception as error:
                logger.exception("engine worker failed")
                for queue in self._sequence_to_queue.values():
                    await queue.put(_GenerationEvent(error=str(error)))
                self._sequence_to_queue.clear()
                raise
