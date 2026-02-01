import torch

from dataclasses import dataclass
from torch import Tensor
from typing import Iterator, List, Tuple

from inferlib.engine.sequence import Sequence
from inferlib.engine.page import PagePool


@dataclass
class PagedKVCache:
    page_pool: PagePool
    num_layers: int
    page_size: int
    device: torch.device

    # def prefill(
    #     self,
    #     sequences: List[Sequence],
    #     layer_id: int,
    #     kv: Tuple[Tensor, Tensor],
    # ):
    #     assert kv[0].shape[0] == len(sequences)
    #     k, v = kv
    #     num_tokens = k.shape[2]

    #     if len(sequences[0].page_ids) == 0:
    #         assert layer_id == 0
    #         pages_needed = math.ceil(num_tokens / self.page_size)
    #         for seq in sequences:
    #             seq.page_ids = [self.page_pool.alloc() for _ in range(pages_needed)]

    #     num_pages = len(sequences[0].page_ids)
    #     for page_idx in range(num_pages):
    #         page_ids = torch.tensor(
    #             [seq.page_ids[page_idx] for seq in sequences],
    #             dtype=torch.long,
    #             device=self.page_pool.device,
    #         )
    #         length = self.page_size
    #         if page_idx == num_pages - 1 and num_tokens % self.page_size != 0:
    #             length = num_tokens % self.page_size

    #         _k = k[:, :, page_idx * self.page_size : (page_idx + 1) * self.page_size, :]
    #         _v = v[:, :, page_idx * self.page_size : (page_idx + 1) * self.page_size, :]
    #         _k, _v = _k[..., :length, :], _v[..., :length, :]
    #         self.page_pool.write_page(page_ids, layer_id, (_k, _v))

    #     if layer_id == self.num_layers - 1:
    #         for seq in sequences:
    #             seq.seq_len = num_tokens

    def _new_page_needed(self, seq: Sequence) -> bool:
        if seq.sequence_length == 0:
            return True
        if seq.sequence_length % self.page_size == 0:
            return True
        return False

    def write(
        self,
        sequence: Sequence,
        layer_id: int,
        kv: Tuple[Tensor, Tensor],
    ):
        if layer_id == 0 and self._new_page_needed(sequence):
            sequence.page_ids.append(self.page_pool.alloc())

        self.page_pool.write(
            sequence.page_ids[-1],
            layer_id,
            sequence.sequence_length % self.page_size,
            kv,
        )

    def read_pages(
        self, sequence: Sequence, layer_id: int
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        for page_id in sequence.page_ids:
            length = self.page_size
            if page_id == sequence.page_ids[-1]:
                rem = sequence.sequence_length % self.page_size
                length = self.page_size if rem == 0 else rem

            yield self.page_pool.read(page_id, layer_id, length)

    def free_pages(self, sequences: List[Sequence]):
        for sequence_state in sequences:
            self.page_pool.free(sequence_state.page_ids)
