from __future__ import annotations

import asyncio

from collections.abc import Sequence

import torch

from inferlib.core.engine.engine import InferlibEngine
from inferlib.types import GenerationOutput, SamplingParams


class LLM:
    def __init__(
        self,
        model: str,
        *,
        enforce_eager: bool = False,
        max_model_len: int | None = None,
        page_size: int = 32,
        batch_size: int = 4,
        max_active_sequences: int = 8,
        memory_limit_bytes: int = 4 * 1024 * 1024 * 1024,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        del enforce_eager
        self.model = model
        self._runner = asyncio.Runner()
        self._closed = False
        self._engine = InferlibEngine(
            model_class=model,
            page_size=page_size,
            batch_size=batch_size,
            max_active_sequences=max_active_sequences,
            memory_limit=memory_limit_bytes,
            max_model_len=max_model_len,
            dtype=dtype,
        )
        self._runner.run(self._engine.start())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._runner.run(self._engine.stop())
        finally:
            self._runner.close()

    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | Sequence[SamplingParams],
        use_tqdm: bool = False,
    ) -> list[GenerationOutput]:
        prompt_token_batches = self._normalize_prompts(prompts)
        params = self._normalize_sampling_params(len(prompt_token_batches), sampling_params)
        return self._runner.run(
            self._engine.generate(
                prompt_token_batches=prompt_token_batches,
                sampling_params=params,
                use_tqdm=use_tqdm,
            )
        )

    def _normalize_prompts(self, prompts: list[str] | list[list[int]]) -> list[list[int]]:
        if not prompts:
            return []

        if all(isinstance(prompt, str) for prompt in prompts):
            return [
                self._engine.tokenizer.encode(prompt, add_special_tokens=False)
                for prompt in prompts
            ]

        if all(isinstance(prompt, list) for prompt in prompts):
            normalized: list[list[int]] = []
            for prompt in prompts:
                if not all(isinstance(token_id, int) for token_id in prompt):
                    raise TypeError("prompt token ids must be integers")
                normalized.append([int(token_id) for token_id in prompt])
            return normalized

        raise TypeError("prompts must be all strings or all token-id lists")

    @staticmethod
    def _normalize_sampling_params(
        num_prompts: int,
        sampling_params: SamplingParams | Sequence[SamplingParams],
    ) -> list[SamplingParams]:
        if isinstance(sampling_params, SamplingParams):
            return [sampling_params for _ in range(num_prompts)]

        params = list(sampling_params)
        if len(params) != num_prompts:
            raise ValueError("sampling_params length must match prompts length")
        if not all(isinstance(param, SamplingParams) for param in params):
            raise TypeError("sampling_params must contain SamplingParams objects")
        return params

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            try:
                self.close()
            except Exception:
                pass
