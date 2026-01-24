import math
import torch

from dataclasses import dataclass
from torch import Tensor
from typing import Iterator, List, Tuple

from inferlib.schema.sequence_state import SequenceState
from .page_pool import PagePool


@dataclass
class PagedKVCache:
    page_pool: PagePool
    num_layers: int
    page_size: int
    device: torch.device

    def _prefill_one(
        self, sequence_state: SequenceState, layer_id: int, kv: Tuple[Tensor, Tensor]
    ):
        k, v = kv  # 1, num_heads, some_len, head_dim
        num_tokens = k.shape[2]
        if len(sequence_state.page_ids) == 0:
            assert layer_id == 0
            pages_needed = math.ceil(num_tokens / self.page_size)
            # TODO: check for -1, for now assume all are valid ids
            sequence_state.page_ids = [
                self.page_pool.alloc() for _ in range(pages_needed)
            ]

        for i, page_id in enumerate(sequence_state.page_ids):
            length = self.page_size
            if page_id == sequence_state.page_ids[-1]:
                if num_tokens % self.page_size != 0:
                    length = num_tokens % self.page_size
            for token_offset in range(length):
                token_position = i * self.page_size + token_offset
                _k = k[:, :, token_position : token_position + 1, :]
                _v = v[:, :, token_position : token_position + 1, :]
                self.page_pool.write(page_id, layer_id, token_offset, (_k, _v))

        if layer_id == self.num_layers - 1:
            sequence_state.seq_len = num_tokens

    def prefill(
        self,
        sequence_states: List[SequenceState],
        layer_id: int,
        kv: Tuple[Tensor, Tensor],
    ):
        assert kv[0].shape[0] == len(sequence_states)
        for i in range(len(sequence_states)):
            self._prefill_one(
                sequence_states[i],
                layer_id,
                (kv[0][i : i + 1, ...], kv[1][i : i + 1, ...]),
            )

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
