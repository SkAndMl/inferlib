from __future__ import annotations

import asyncio

import torch

from inferlib.core.engine.page import PageManager
from inferlib.core.engine.runner import Runner
from inferlib.core.engine.scheduler import Scheduler
from inferlib.core.engine.sequence import Sequence
from inferlib.core.engine.engine import InferlibEngine


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _BatchEncoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _StubModel:
    def __init__(self, *, prefill_tokens: list[int] | None = None, decode_tokens: list[int] | None = None) -> None:
        self.prefill_tokens = prefill_tokens or []
        self.decode_tokens = decode_tokens or []
        self.prefill_calls: list[list[str]] = []
        self.decode_calls: list[list[str]] = []

    def prefill(self, *, sequences, page_manager):
        del page_manager
        self.prefill_calls.append([sequence.s_id for sequence in sequences])
        return self.prefill_tokens

    def decode(self, *, sequences, page_manager):
        del page_manager
        self.decode_calls.append([sequence.s_id for sequence in sequences])
        return self.decode_tokens


def _make_page_manager(*, num_pages: int = 8, page_size: int = 4, max_pages_per_sequence: int = 2) -> PageManager:
    return PageManager(
        num_pages=num_pages,
        num_layers=1,
        num_heads=1,
        page_size=page_size,
        head_dim=2,
        max_pages_per_sequence=max_pages_per_sequence,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_normalize_token_ids_supports_common_shapes() -> None:
    assert InferlibEngine._normalize_token_ids([1, 2, 3]) == [1, 2, 3]
    assert InferlibEngine._normalize_token_ids(_Encoding([4, 5])) == [4, 5]
    assert InferlibEngine._normalize_token_ids([_Encoding([6]), _Encoding([7, 8])]) == [6, 7, 8]
    assert InferlibEngine._normalize_token_ids(_BatchEncoding([[9, 10], [11]])) == [9, 10, 11]


def test_calc_num_pages_uses_dtype_size() -> None:
    pages = InferlibEngine.calc_num_pages(
        torch.float16,
        num_layers=2,
        num_heads=2,
        head_dim=8,
        page_size=4,
        memory_limit=1024,
    )

    assert pages == 2


def test_runner_prefill_appends_sampled_token() -> None:
    model = _StubModel(prefill_tokens=[41])
    runner = Runner(llm=model, page_manager=object())
    sequence = Sequence(
        s_id="prefill",
        prompt_tokens=[1, 2, 3, 4],
        completion_tokens=[],
        eos_token_id=127,
        temperature=1.0,
        max_tokens=4,
    )

    runner.run([sequence])

    assert model.prefill_calls == [["prefill"]]
    assert sequence.completion_tokens == [41]
    assert sequence.last_token_id == 41


def test_runner_decode_appends_token_to_active_sequence() -> None:
    model = _StubModel(decode_tokens=[52])
    runner = Runner(llm=model, page_manager=object())
    sequence = Sequence(
        s_id="decode",
        prompt_tokens=[1, 2, 3, 4],
        completion_tokens=[41],
        eos_token_id=127,
        last_token_id=41,
        temperature=1.0,
        max_tokens=4,
    )

    runner.run([sequence])

    assert model.decode_calls == [["decode"]]
    assert sequence.completion_tokens == [41, 52]
    assert sequence.last_token_id == 52


def test_scheduler_requeues_unfinished_sequence_for_continuous_decode() -> None:
    page_manager = _make_page_manager()
    scheduler = Scheduler(
        page_manager=page_manager,
        request_event=asyncio.Event(),
        max_pages_per_sequence=2,
        batch_size=1,
        max_active_sequences=1,
    )
    sequence = Sequence(
        s_id="seq",
        prompt_tokens=[1, 2, 3, 4],
        completion_tokens=[],
        eos_token_id=127,
        temperature=1.0,
        max_tokens=4,
    )

    assert scheduler.add_request(sequence)
    first_batch = asyncio.run(scheduler.schedule())
    assert [seq.s_id for seq in first_batch] == ["seq"]

    sequence.completion_tokens.append(9)
    sequence.last_token_id = 9
    scheduler.update(first_batch)

    second_batch = asyncio.run(scheduler.schedule())

    assert [seq.s_id for seq in second_batch] == ["seq"]


def test_scheduler_evicts_oldest_page_when_decode_needs_new_page_at_limit() -> None:
    page_manager = _make_page_manager(num_pages=6, page_size=4, max_pages_per_sequence=2)
    scheduler = Scheduler(
        page_manager=page_manager,
        request_event=asyncio.Event(),
        max_pages_per_sequence=2,
        batch_size=1,
        max_active_sequences=1,
    )
    sequence = Sequence(
        s_id="seq",
        prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8],
        completion_tokens=[],
        eos_token_id=127,
        temperature=1.0,
        max_tokens=8,
    )

    assert scheduler.add_request(sequence)
    prefill_batch = asyncio.run(scheduler.schedule())
    assert page_manager.get_num_pages(sequence) == 2

    sequence.completion_tokens.append(9)
    sequence.last_token_id = 9
    scheduler.update(prefill_batch)

    decode_batch = asyncio.run(scheduler.schedule())

    assert [seq.s_id for seq in decode_batch] == ["seq"]
    assert sequence.tokens_evicted == 4
    assert page_manager.get_num_pages(sequence) == 2
