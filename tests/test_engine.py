from __future__ import annotations

import torch

from inferlib.core.engine.engine import InferlibEngine


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _BatchEncoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


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
