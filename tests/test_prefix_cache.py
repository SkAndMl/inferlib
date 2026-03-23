import asyncio
import math

import torch

from inferlib.core.engine.page import PageManager
from inferlib.core.engine.runner import Runner
from inferlib.core.engine.scheduler import Scheduler
from inferlib.core.engine.sequence import Sequence
from inferlib.core.models.qwen3 import Qwen3, Qwen3Config


def _make_cfg() -> Qwen3Config:
    return Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        head_dim=8,
        vocab_size=128,
        max_position_embeddings=128,
        page_size=4,
        tie_word_embeddings=False,
    )


def _make_page_manager(cfg: Qwen3Config) -> PageManager:
    return PageManager(
        num_pages=128,
        num_layers=cfg.num_hidden_layers,
        num_heads=cfg.num_key_value_heads,
        page_size=cfg.page_size,
        head_dim=cfg.head_dim,
        max_pages_per_sequence=32,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _run_steps(model: Qwen3, page_manager: PageManager, sequence: Sequence, steps: int) -> list[int]:
    event = asyncio.Event()
    scheduler = Scheduler(
        page_manager=page_manager,
        request_event=event,
        max_pages_per_sequence=32,
        batch_size=1,
        max_active_sequences=1,
    )
    runner = Runner(llm=model, page_manager=page_manager)
    assert scheduler.add_request(sequence)

    pages_after_step: list[int] = []
    for _ in range(steps):
        batch = asyncio.run(scheduler.schedule())
        assert batch
        runner.run(batch)
        scheduler.update(batch)
        pages_after_step.append(page_manager.get_num_pages(sequence))

    return pages_after_step


def test_decode_reuses_partially_filled_last_page():
    cfg = _make_cfg()
    torch.manual_seed(0)
    model = Qwen3(cfg).eval()
    page_manager = _make_page_manager(cfg)
    sequence = Sequence(
        s_id="seq",
        prompt_tokens=[10, 11, 12, 13, 14, 15, 16],
        completion_tokens=[],
        eos_token_id=127,
        temperature=1.0,
        max_tokens=8,
    )

    torch.manual_seed(123)
    pages_after_step = _run_steps(model, page_manager, sequence, steps=2)

    assert pages_after_step == [2, 2]


def test_decode_allocates_new_page_at_next_boundary():
    cfg = _make_cfg()
    torch.manual_seed(0)
    model = Qwen3(cfg).eval()
    page_manager = _make_page_manager(cfg)
    sequence = Sequence(
        s_id="seq",
        prompt_tokens=[10, 11, 12, 13, 14, 15, 16],
        completion_tokens=[],
        eos_token_id=127,
        temperature=1.0,
        max_tokens=8,
    )

    torch.manual_seed(123)
    pages_after_step = _run_steps(model, page_manager, sequence, steps=3)

    assert pages_after_step == [2, 2, 3]


def test_plan_prefill_is_side_effect_free_until_allocate():
    cfg = _make_cfg()
    torch.manual_seed(0)
    model = Qwen3(cfg).eval()
    prompt = [10, 11, 12, 13, 14, 15, 16]
    page_manager = _make_page_manager(cfg)

    warmup = Sequence(
        s_id="warmup",
        prompt_tokens=prompt,
        completion_tokens=[],
        eos_token_id=127,
        temperature=1.0,
        max_tokens=8,
    )
    torch.manual_seed(123)
    _run_steps(model, page_manager, warmup, steps=1)

    candidate = Sequence(
        s_id="candidate",
        prompt_tokens=prompt,
        completion_tokens=[],
        eos_token_id=127,
        temperature=1.0,
        max_tokens=8,
    )
    pages_needed = math.ceil(len(candidate.prompt_tokens) / cfg.page_size)
    free_pages_before = page_manager.num_free_pages
    refcounts_before = [page.refcount for page in page_manager._page_info]

    plan = page_manager.plan_prefill(candidate, pages_needed)

    assert plan is not None
    assert plan.cached_page_ids == page_manager.get_cached_page_ids(warmup)
    assert plan.prefix_cached_tokens == cfg.page_size
    assert candidate.prefix_cached_tokens == 0
    assert page_manager.get_num_pages(candidate) == 0
    assert page_manager.num_free_pages == free_pages_before
    assert [page.refcount for page in page_manager._page_info] == refcounts_before

    page_manager.allocate_prefill(candidate, plan)

    assert candidate.prefix_cached_tokens == cfg.page_size
    assert page_manager.get_num_pages(candidate) == pages_needed
    assert page_manager.num_free_pages == free_pages_before - plan.fresh_pages_needed
    assert page_manager._page_info[plan.cached_page_ids[0]].refcount == refcounts_before[plan.cached_page_ids[0]] + 1


def test_scheduler_requeues_prefill_with_mismatched_cached_prefix():
    cfg = _make_cfg()
    torch.manual_seed(0)
    model = Qwen3(cfg).eval()
    shared_page_manager = _make_page_manager(cfg)

    warmup = Sequence(
        s_id="warmup",
        prompt_tokens=[10, 11, 12, 13, 14, 15, 16],
        completion_tokens=[],
        eos_token_id=127,
        temperature=1.0,
        max_tokens=8,
    )
    torch.manual_seed(123)
    _run_steps(model, shared_page_manager, warmup, steps=1)

    cached = Sequence(
        s_id="cached",
        prompt_tokens=[10, 11, 12, 13, 14, 15, 16],
        completion_tokens=[],
        eos_token_id=127,
        temperature=1.0,
        max_tokens=8,
    )
    uncached = Sequence(
        s_id="uncached",
        prompt_tokens=[20, 21, 22, 23, 24, 25, 26],
        completion_tokens=[],
        eos_token_id=127,
        temperature=1.0,
        max_tokens=8,
    )

    event = asyncio.Event()
    scheduler = Scheduler(
        page_manager=shared_page_manager,
        request_event=event,
        max_pages_per_sequence=32,
        batch_size=2,
        max_active_sequences=2,
    )
    runner = Runner(llm=model, page_manager=shared_page_manager)
    assert scheduler.add_request(cached)
    assert scheduler.add_request(uncached)

    torch.manual_seed(123)
    first_batch = asyncio.run(scheduler.schedule())
    assert [sequence.s_id for sequence in first_batch] == ["cached"]
    assert cached.prefix_cached_tokens == cfg.page_size
    assert shared_page_manager.get_num_pages(uncached) == 0

    runner.run(first_batch)
    scheduler.update(first_batch)

    torch.manual_seed(123)
    second_batch = asyncio.run(scheduler.schedule())
    assert [sequence.s_id for sequence in second_batch] == ["uncached"]
    assert uncached.prefix_cached_tokens == 0


def test_prefix_cached_generation_matches_fresh_generation():
    cfg = _make_cfg()
    torch.manual_seed(0)
    model = Qwen3(cfg).eval()
    prompt = [10, 11, 12, 13, 14, 15, 16]

    shared_page_manager = _make_page_manager(cfg)
    warmup = Sequence(
        s_id="warmup",
        prompt_tokens=prompt,
        completion_tokens=[],
        eos_token_id=127,
        temperature=1.0,
        max_tokens=8,
    )
    torch.manual_seed(123)
    _run_steps(model, shared_page_manager, warmup, steps=5)

    cached = Sequence(
        s_id="cached",
        prompt_tokens=prompt,
        completion_tokens=[],
        eos_token_id=127,
        temperature=1.0,
        max_tokens=8,
    )
    torch.manual_seed(123)
    _run_steps(model, shared_page_manager, cached, steps=5)

    fresh_page_manager = _make_page_manager(cfg)
    fresh = Sequence(
        s_id="fresh",
        prompt_tokens=prompt,
        completion_tokens=[],
        eos_token_id=127,
        temperature=1.0,
        max_tokens=8,
    )
    torch.manual_seed(123)
    _run_steps(model, fresh_page_manager, fresh, steps=5)

    assert cached.prefix_cached_tokens > 0
    assert cached.completion_tokens == fresh.completion_tokens
