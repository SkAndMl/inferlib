from __future__ import annotations

import pytest

import inferlib.api

from inferlib import LLM, SamplingParams
from tests.fakes import FakePublicEngine


def test_llm_generate_encodes_text_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    FakePublicEngine.instances.clear()
    monkeypatch.setattr(inferlib.api, "InferlibEngine", FakePublicEngine)

    llm = LLM("fake-model")
    outputs = llm.generate(["hi", "ok"], SamplingParams(max_tokens=4))
    engine = FakePublicEngine.instances[-1]
    llm.close()

    assert engine.received_prompts == [[104, 105], [111, 107]]
    assert len(engine.received_sampling_params) == 2
    assert outputs[0].text == "out-0"
    assert outputs[1].completion_tokens == 1


def test_llm_generate_accepts_token_id_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    FakePublicEngine.instances.clear()
    monkeypatch.setattr(inferlib.api, "InferlibEngine", FakePublicEngine)

    llm = LLM("fake-model")
    llm.generate([[1, 2], [3, 4, 5]], [SamplingParams(), SamplingParams(max_tokens=2)])
    engine = FakePublicEngine.instances[-1]
    llm.close()

    assert engine.received_prompts == [[1, 2], [3, 4, 5]]
    assert engine.received_sampling_params[1].max_tokens == 2


def test_llm_generate_rejects_sampling_param_length_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    FakePublicEngine.instances.clear()
    monkeypatch.setattr(inferlib.api, "InferlibEngine", FakePublicEngine)

    llm = LLM("fake-model")
    with pytest.raises(ValueError):
        llm.generate([[1, 2], [3, 4]], [SamplingParams()])
    llm.close()
