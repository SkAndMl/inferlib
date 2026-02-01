import torch

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from torch import Tensor

from inferlib.engine.sequence import Sequence


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

    def write_page(self, page_id: int, layer_id: int, kv: tuple[Tensor, Tensor]):
        length = kv[0].shape[1]
        self._key_pool[page_id, layer_id, :, :length, :] = kv[0]
        self._value_pool[page_id, layer_id, :, :length, :] = kv[1]

    def read(
        self, page_id: int, layer_id: int, length: int | None = None
    ) -> tuple[Tensor, Tensor]:
        assert layer_id < self.num_layers
        if length is None:
            length = self.page_size

        k = self._key_pool[page_id, layer_id, :, :length, :]
        v = self._value_pool[page_id, layer_id, :, :length, :]
        return k, v


class PageManager:
    page_pool: PagePool
    page_table: dict[int, list[int]]

    def __post_init__(self):
        self.free_pages = deque(range(self.page_pool.num_pages))
        self.page_size = self.page_pool.page_size

    def can_allocate(self, s_id: int, num_pages: int) -> bool:
        """
        scheduler calls this function before scheduling a sequence
        for decoding.
        if there are enough pages available the manager, reserves
        those pages for the sequence by moving the page_ids
        from `free_pages` to `allocated_pages`
        """
        if len(self.free_pages) < num_pages:
            return False

        if s_id not in self.page_table:
            self.page_table[s_id] = []

        page_ids = [self.free_pages.popleft() for _ in range(num_pages)]
        self.page_table[s_id] = page_ids
        return True

    def free(self, s_id: int):
        self.free_pages.extend(self.page_table.pop(s_id))

    def write(
        self,
        sequence: Sequence,
        layer_id: int,
        kv: tuple[Tensor, Tensor],
    ):
        self.page_pool.write(
            self.page_table[sequence.s_id][-1],
            layer_id,
            sequence.sequence_length % self.page_size,
            kv,
        )

    def read_pages(
        self, sequence: Sequence, layer_id: int
    ) -> Iterator[tuple[Tensor, Tensor]]:
        for page_id in sequence.page_ids:
            length = self.page_size
            if page_id == sequence.page_ids[-1]:
                rem = sequence.sequence_length % self.page_size
                length = self.page_size if rem == 0 else rem

            yield self.page_pool.read(page_id, layer_id, length)

    def _prefill_one(
        self, sequence: Sequence, layer_id: int, kv: tuple[Tensor, Tensor]
    ):
        page_ids = self.page_table[sequence.s_id]
        k, v = kv
        for i, page_id in enumerate(page_ids):
            length = self.page_size
            if page_id == page_ids[-1]:
                rem = sequence.sequence_length % self.page_size
                length = self.page_size if rem == 0 else rem

            _k = k[:, i * self.page_size : (i + 1) * self.page_size, :]
            _v = v[:, i * self.page_size : (i + 1) * self.page_size, :]
            _k, _v = _k[:, :length, :], _v[:, :length, :]
            self.page_pool.write_page(page_id, layer_id, (_k, _v))

    def prefill(
        self, sequences: list[Sequence], layer_id: int, kv: tuple[Tensor, Tensor]
    ):
        k, v = kv
        for i, sequence in enumerate(sequences):
            self._prefill_one(sequence, layer_id, (k[i, ...], v[i, ...]))
