import torch

from dataclasses import dataclass


@dataclass
class Page:
    pid: int
    num_filled: int


@dataclass
class PagePool:
    num_pages: int
    num_layers: int
    tokens_per_page: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype

    def __post_init__(self):
        size = (
            self.num_pages,
            self.num_layers,
            self.num_kv_heads,
            self.tokens_per_page,
            self.head_dim,
        )

        self.key_pool = torch.empty(size=size, dtype=self.dtype)
        self.value_pool = torch.empty(size=size, dtype=self.dtype)
        self._num_free = self.num_pages

    @property
    def num_free(self) -> int:
        return self._num_free

    def alloc(self) -> Page:
        pass

    def free(self, page: Page):
        pass
