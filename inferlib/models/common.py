import torch

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class ModelConfig:
    attn_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-05
    n_ctx: int = 1024
    n_embd: int = 1024
    n_head: int = 16
    n_layer: int = 24
    n_positions: int = 1024
    resid_pdrop: int = 0.1
    vocab_size: int = 50257
    use_kv_cache: bool = True
    page_size: int = 256  # assume it to be 1/4th of n_ctx
    bsz: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PagedKVCache:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.head_dim = config.n_embd // config.n_head
        self.max_page_idx = 0
        self._seq_len = 0
        self._k_pages: Dict[str, Any] = {
            0: {
                "page": torch.empty(
                    size=(config.bsz, config.n_head, config.page_size, self.head_dim)
                ),
                "num_filled": 0,
            }
        }
        self._v_pages: Dict[str, Any] = {
            0: {
                "page": torch.empty(
                    size=(config.bsz, config.n_head, config.page_size, self.head_dim)
                ),
                "num_filled": 0,
            }
        }

    @property
    def num_pages(self) -> int:
        return len(self._k_pages)

    @property
    def seq_len(self) -> int:
        return self._seq_len

    # should this be a generator?
    def get_kv_page(self, page_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert page_idx < self.num_pages
        k, v = self._k_pages[page_idx], self._v_pages[page_idx]
        return k["page"][:, :, : k["num_filled"], :], v["page"][
            :, :, : v["num_filled"], :
        ]

    def put_kv_page(self, kv: Tuple[torch.Tensor, torch.Tensor]):
        k, v = kv
        assert (
            k.shape
            == v.shape
            == (self.config.bsz, self.config.n_head, 1, self.head_dim)
        )
        if self._k_pages[self.max_page_idx]["num_filled"] == self.config.page_size:
            self.max_page_idx += 1
            self._k_pages[self.max_page_idx] = {
                "page": torch.empty(
                    size=(
                        self.config.bsz,
                        self.config.n_head,
                        self.config.page_size,
                        self.head_dim,
                    )
                ),
                "num_filled": 0,
            }
            self._v_pages[self.max_page_idx] = {
                "page": torch.empty(
                    size=(
                        self.config.bsz,
                        self.config.n_head,
                        self.config.page_size,
                        self.head_dim,
                    )
                ),
                "num_filled": 0,
            }

        num_filled = self._k_pages[self.max_page_idx]["num_filled"]
        self._k_pages[self.max_page_idx]["page"][
            :, :, num_filled : num_filled + 1, :
        ] = k
        self._v_pages[self.max_page_idx]["page"][
            :, :, num_filled : num_filled + 1, :
        ] = v

        self._k_pages[self.max_page_idx]["num_filled"] += 1
        self._v_pages[self.max_page_idx]["num_filled"] += 1
        self._seq_len += 1

    def reset(self):
        self.max_page_idx = 0
        self._seq_len = 0
        shape = (
            self.config.bsz,
            self.config.n_head,
            self.config.page_size,
            self.head_dim,
        )
        self._k_pages: Dict[str, Any] = {
            0: {
                "page": torch.empty(size=shape),
                "num_filled": 0,
            }
        }
        self._v_pages: Dict[str, Any] = {
            0: {
                "page": torch.empty(size=shape),
                "num_filled": 0,
            }
        }
