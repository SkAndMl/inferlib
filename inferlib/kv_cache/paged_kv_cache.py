import torch

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SequenceState:
    seq_id: int
    seq_len: int
    page_ids: List[int] = field(default_factory=list)

    def update_page_ids(self, page_id):
        if len(self.page_ids) == 0:
            self.page_ids.append(page_id)
        elif self.page_ids[-1] != page_id:
            self.page_ids.append(page_id)

        self.seq_len += 1


@dataclass
class PagePool:
    num_pages: int
    num_layers: int
    tokens_per_page: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self):
        size = (
            self.num_pages,
            self.num_layers,
            self.num_kv_heads,
            self.tokens_per_page,
            self.head_dim,
        )

        self.key_pool = torch.empty(size=size, dtype=self.dtype, device=self.device)
        self.value_pool = torch.empty(size=size, dtype=self.dtype, device=self.device)
        self.free_pages = list(range(self.num_pages))

    @property
    def num_free(self) -> int:
        return len(self.free_pages)

    def _is_partially_filled(self, seq_len: int) -> bool:
        return seq_len % self.tokens_per_page != 0

    def alloc(self, seq: SequenceState) -> int:
        # seq.seq_len is the length of the tokens generated so far
        # it does not include the current kv token

        if not self._is_partially_filled(seq.seq_len) and len(self.free_pages) == 0:
            return -1  # return -1 if there are no free pages

        if len(seq.page_ids) == 0:
            page_id = self.free_pages.pop(0)
        else:
            if self._is_partially_filled(seq.seq_len):
                page_id = seq.page_ids[-1]
            else:
                page_id = self.free_pages.pop(0)

        return page_id

    def free(self, page_ids: List[int]):
        self.free_pages.extend(page_ids)

    def write_page(
        self,
        kv: Tuple[torch.Tensor, torch.Tensor],
        page_id: int,
        layer: int,
        token_position: int,
    ):
        # TODO: add seq id check before writing to a page

        k, v = kv
        assert (
            k.shape
            == v.shape
            == (1, self.num_layers, self.num_kv_heads, 1, self.head_dim)
        )
        self.key_pool[page_id, layer, :, token_position : token_position + 1, :] = k[0]
        self.value_pool[page_id, layer, :, token_position : token_position + 1, :] = v[
            0
        ]

    def read_page(
        self, page_id: int, layer: int, num_tokens: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert page_id not in self.free_pages

        num_tokens = num_tokens if num_tokens is not None else self.tokens_per_page
        k = self.key_pool[page_id, layer, :, :num_tokens, :]
        v = self.value_pool[page_id, layer, :, :num_tokens, :]

        # once batch is added '[None, ...]' can be discarded
        return k[None, ...], v[None, ...]


@dataclass
class CacheManager:
    seq: SequenceState
    num_layers: int
