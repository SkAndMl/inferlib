import torch

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from torch import Tensor

from inferlib.core.engine.sequence import Sequence


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
        self,
        page_ids: Tensor,
        layer_id: int,
        offsets: Tensor,
        kv: tuple[Tensor, Tensor],
    ):
        assert layer_id < self.num_layers
        assert (offsets < self.page_size).all()

        k, v = kv

        assert k.shape == v.shape == (len(page_ids), self.num_heads, 1, self.head_dim)

        self._key_pool[page_ids, layer_id, :, offsets, :] = k.squeeze(dim=2)
        self._value_pool[page_ids, layer_id, :, offsets, :] = v.squeeze(dim=2)

    def write_page(self, page_ids: Tensor, layer_id: int, kv: tuple[Tensor, Tensor]):
        length = kv[0].shape[2]
        self._key_pool[page_ids, layer_id, :, :length, :] = kv[0]
        self._value_pool[page_ids, layer_id, :, :length, :] = kv[1]

    def read(self, page_id: Tensor, layer_id: int) -> tuple[Tensor, Tensor]:
        assert layer_id < self.num_layers
        k = self._key_pool[page_id, layer_id, ...]
        v = self._value_pool[page_id, layer_id, ...]
        return k, v


@dataclass
class PageManager:
    num_pages: int
    num_layers: int
    num_heads: int
    page_size: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self):
        self._page_pool = PagePool(
            num_pages=self.num_pages,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            page_size=self.page_size,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self._page_table: dict[int, list[int]] = {}
        self._free_pages = deque(range(self._page_pool.num_pages))

    def can_allocate(self, s_id: int, num_pages: int) -> bool:
        """
        scheduler calls this function before scheduling a sequence
        for decoding.
        if there are enough pages available the manager, reserves
        those pages for the sequence by moving the page_ids
        from `free_pages` to `allocated_pages`
        """
        if len(self._free_pages) < num_pages:
            return False

        if s_id not in self._page_table:
            self._page_table[s_id] = []

        page_ids = [self._free_pages.popleft() for _ in range(num_pages)]
        self._page_table[s_id].extend(page_ids)
        return True

    def free(self, s_id: int):
        self._free_pages.extend(self._page_table.pop(s_id))

    def write(
        self,
        sequences: list[Sequence],
        layer_id: int,
        kv: tuple[Tensor, Tensor],
    ):
        page_ids = torch.tensor(
            [self._page_table[seq.s_id][-1] for seq in sequences],
            device=self.device,
            dtype=torch.long,
        )
        offsets = torch.tensor(
            data=[(len(seq) - 1) % self.page_size for seq in sequences],
            dtype=torch.long,
            device=self.device,
        )
        self._page_pool.write(
            page_ids=page_ids,
            layer_id=layer_id,
            offsets=offsets,
            kv=kv,
        )

    def read_pages(
        self, sequences: Sequence | list[Sequence], layer_id: int
    ) -> Iterator[tuple[Tensor, Tensor]]:
        if isinstance(sequences, Sequence):
            sequences = [sequences]

        num_pages = len(self._page_table[sequences[0].s_id])
        assert all(len(self._page_table[seq.s_id]) == num_pages for seq in sequences)

        for i in range(num_pages):
            page_ids = torch.tensor(
                [self._page_table[seq.s_id][i] for seq in sequences],
                device=self.device,
                dtype=torch.long,
            )

            yield self._page_pool.read(page_ids, layer_id)

    def prefill(
        self, sequences: list[Sequence], layer_id: int, kv: tuple[Tensor, Tensor]
    ):
        k, v = kv
        num_pages = len(self._page_table[sequences[0].s_id])
        assert all(len(self._page_table[seq.s_id]) == num_pages for seq in sequences)

        for i in range(num_pages):
            page_ids = torch.tensor(
                [self._page_table[seq.s_id][i] for seq in sequences],
                dtype=torch.long,
                device=self.device,
            )
            _k = k[:, :, i * self.page_size : (i + 1) * self.page_size, :]
            _v = v[:, :, i * self.page_size : (i + 1) * self.page_size, :]
            self._page_pool.write_page(
                page_ids=page_ids, layer_id=layer_id, kv=(_k, _v)
            )

    def get_num_pages(self, s_id: int) -> int:
        return len(self._page_table.get(s_id, []))


__all__ = ["PageManager"]
