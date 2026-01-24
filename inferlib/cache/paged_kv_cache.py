import math
import torch

from dataclasses import dataclass
from torch import Tensor
from typing import Iterator, List, Tuple

from inferlib.schema.sequence_state import SequenceState
from .page_pool import PagePool

# TODO: assumes alloc always returns a valid page id. should wait if returned page_id is -1


@dataclass
class PagedKVCache:
    page_pool: PagePool
    num_layers: int
    page_size: int
    device: torch.device

    def prefill(
        self,
        sequence_states: List[SequenceState],
        layer_id: int,
        kv: Tuple[Tensor, Tensor],
    ):
        assert kv[0].shape[0] == len(sequence_states)
        k, v = kv
        num_tokens = k.shape[2]

        if len(sequence_states[0].page_ids) == 0:
            assert layer_id == 0
            pages_needed = math.ceil(num_tokens / self.page_size)
            for seq in sequence_states:
                seq.page_ids = [self.page_pool.alloc() for _ in range(pages_needed)]

        num_pages = len(sequence_states[0].page_ids)
        for page_idx in range(num_pages):
            page_ids = torch.tensor(
                [seq.page_ids[page_idx] for seq in sequence_states],
                dtype=torch.long,
                device=self.page_pool.device,
            )
            length = self.page_size
            if page_idx == num_pages - 1 and num_tokens % self.page_size != 0:
                length = num_tokens % self.page_size

            _k = k[:, :, page_idx * self.page_size : (page_idx + 1) * self.page_size, :]
            _v = v[:, :, page_idx * self.page_size : (page_idx + 1) * self.page_size, :]
            _k, _v = _k[..., :length, :], _v[..., :length, :]
            self.page_pool.write_page(page_ids, layer_id, (_k, _v))

        if layer_id == self.num_layers - 1:
            for seq in sequence_states:
                seq.seq_len = num_tokens

    def _new_page_needed(self, seq: SequenceState) -> bool:
        if seq.seq_len == 0:
            return True
        if seq.seq_len % self.page_size == 0:
            return True
        return False

    def write(
        self,
        sequence_states: List[SequenceState],
        layer_id: int,
        kv: Tuple[Tensor, Tensor],
    ):
        if layer_id == 0:
            for seq in sequence_states:
                if self._new_page_needed(seq):
                    seq.page_ids.append(self.page_pool.alloc())

        page_ids = torch.tensor(
            [seq.page_ids[-1] for seq in sequence_states],
            dtype=torch.long,
            device=self.page_pool.device,
        )
        offsets = torch.tensor(
            [seq.seq_len % self.page_size for seq in sequence_states],
            dtype=torch.long,
            device=self.page_pool.device,
        )
        self.page_pool.write_many(page_ids, layer_id, offsets, kv)

        if layer_id == self.num_layers - 1:
            for seq in sequence_states:
                seq.seq_len += 1

    def read_pages(
        self, sequence_states: List[SequenceState], layer_id: int
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        for page_idx in range(len(sequence_states[0].page_ids)):
            length = None
            if page_idx == len(sequence_states[0].page_ids) - 1:
                if sequence_states[0].seq_len % self.page_size == 0:
                    length = self.page_size
                else:
                    length = sequence_states[0].seq_len % self.page_size
            page_ids = torch.tensor(
                [seq.page_ids[page_idx] for seq in sequence_states],
                dtype=torch.long,
                device=self.page_pool.device,
            )
            k, v = self.page_pool.read_many(page_ids, layer_id, length)
            yield (k, v)

    def free_pages(self, sequence_states: List[SequenceState]):
        for sequence_state in sequence_states:
            self.page_pool.free(sequence_state.page_ids)
