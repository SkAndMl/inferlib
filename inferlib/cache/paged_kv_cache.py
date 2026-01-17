import math

from dataclasses import dataclass
from torch import Tensor
from typing import Iterator, Tuple

from .models import SequenceState
from .page_pool import PagePool


@dataclass
class PagedKVCache:
    page_pool: PagePool
    num_layers: int
    page_size: int

    def prefill(self, seq: SequenceState, layer_id: int, kv: Tuple[Tensor, Tensor]):
        k, v = kv  # 1, num_heads, some_len, head_dim
        num_tokens = k.shape[2]
        if len(seq.page_ids) == 0:
            assert layer_id == 0
            pages_needed = math.ceil(num_tokens / self.page_size)
            # TODO: check for -1, for now assume all are valid ids
            seq.page_ids = [self.page_pool.alloc() for _ in range(pages_needed)]

        for i, page_id in enumerate(seq.page_ids):
            length = self.page_size
            if page_id == seq.page_ids[-1]:
                if num_tokens % self.page_size != 0:
                    length = num_tokens % self.page_size
            for token_offset in range(length):
                token_position = i * self.page_size + token_offset
                _k = k[:, :, token_position : token_position + 1, :]
                _v = v[:, :, token_position : token_position + 1, :]
                self.page_pool.write(page_id, layer_id, token_offset, (_k, _v))

        if layer_id == self.num_layers - 1:
            seq.seq_len = num_tokens

    def _new_page_needed(self, seq: SequenceState) -> bool:
        if seq.seq_len == 0:
            return True
        if seq.seq_len % self.page_size == 0:
            return True
        return False

    def write(self, seq: SequenceState, layer_id: int, kv: Tuple[Tensor, Tensor]):
        if layer_id == 0 and self._new_page_needed(seq):
            seq.page_ids.append(self.page_pool.alloc())

        self.page_pool.write(
            seq.page_ids[-1], layer_id, seq.seq_len % self.page_size, kv
        )

        if layer_id == self.num_layers - 1:
            seq.seq_len += 1

    def read_pages(
        self, seq: SequenceState, layer_id: int
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        for page_id in seq.page_ids:
            length = None
            if page_id == seq.page_ids[-1]:
                if seq.seq_len % self.page_size == 0:
                    length = self.page_size
                else:
                    length = seq.seq_len % self.page_size
            yield self.page_pool.read(page_id, layer_id, length)
