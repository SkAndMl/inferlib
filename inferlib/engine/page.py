import torch

from dataclasses import dataclass
from torch import Tensor


@dataclass
class PagePool:
    num_pages: int
    num_layers: int
    num_heads: int
    page_size: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self):
        size = (
            self.num_pages,
            self.num_layers,
            self.num_heads,
            self.page_size,
            self.head_dim,
        )
        self._key_pool = torch.empty(size=size, dtype=self.dtype, device=self.device)
        self._value_pool = torch.empty(size=size, dtype=self.dtype, device=self.device)

    def write(
        self, page_id: int, layer_id: int, offset: int, kv: tuple[Tensor, Tensor]
    ):
        assert layer_id < self.num_layers
        assert offset < self.page_size

        k, v = kv

        assert k.shape == v.shape == (1, self.num_heads, 1, self.head_dim)

        self._key_pool[page_id, layer_id, :, offset : offset + 1, :] = k[0]
        self._value_pool[page_id, layer_id, :, offset : offset + 1, :] = v[0]

    def write_many(
        self,
        page_ids: Tensor,
        layer_id: int,
        offsets: Tensor,
        kv: tuple[Tensor, Tensor],
    ):
        k, v = kv
        assert (
            k.shape == v.shape == (page_ids.shape[0], self.num_heads, 1, self.head_dim)
        )

        self._key_pool[page_ids, layer_id, :, offsets, :] = k.squeeze(2)
        self._value_pool[page_ids, layer_id, :, offsets, :] = v.squeeze(2)

    def read(
        self, page_id: int, layer_id: int, length: int | None = None
    ) -> tuple[Tensor, Tensor]:
        assert layer_id < self.num_layers
        if length is None:
            length = self.page_size

        k = self._key_pool[page_id, layer_id, :, :length, :]
        v = self._value_pool[page_id, layer_id, :, :length, :]
        return k, v

    def read_many(
        self, page_ids: Tensor, layer_id: int, length: int | None = None
    ) -> tuple[Tensor, Tensor]:
        if length is None:
            length = self.page_size

        k = self._key_pool.index_select(0, page_ids)[:, layer_id, :, :length, :]
        v = self._value_pool.index_select(0, page_ids)[:, layer_id, :, :length, :]
        return k, v


@dataclass
class PageManager:
    _page_pool: PagePool
