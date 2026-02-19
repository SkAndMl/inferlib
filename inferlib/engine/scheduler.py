import math
from collections import defaultdict, deque
from typing import Literal

from inferlib.engine.page import PageManager
from inferlib.engine.sequence import Sequence, SequenceState
from inferlib.log import logger


class Bucket:
    def __init__(self, page_size: int):
        self.page_size = page_size
        self._buckets: dict[int, deque[Sequence]] = defaultdict(deque)
        self._total_sequences: int = 0

    def add(
        self,
        sequences: Sequence | list[Sequence],
        append: Literal["left", "right"] = "right",
    ):
        if isinstance(sequences, Sequence):
            sequences = [sequences]
        for sequence in sequences:
            bucket_idx = (
                sequence.sequence_length + self.page_size - 1
            ) // self.page_size

            match append:
                case "right":
                    self._buckets[bucket_idx].append(sequence)
                case "left":
                    self._buckets[bucket_idx].appendleft(sequence)
            self._total_sequences += 1

    def get_batch(self, batch_size: int) -> list[Sequence]:
        if len(self) == 0:
            return []

        max_idx = max(self._buckets, key=lambda k: len(self._buckets[k]))
        batch = []
        while self._buckets[max_idx] and len(batch) < batch_size:
            batch.append(self._buckets[max_idx].popleft())
            self._total_sequences -= 1
        return batch

    def __len__(self):
        return self._total_sequences


class Scheduler:
    def __init__(self, page_manager: PageManager, batch_size: int = 4):
        self.page_manager = page_manager
        self._page_size = page_manager.page_size
        self.batch_size = batch_size
        self.prefill_bucket = Bucket(self._page_size)
        self.decode_sequences: deque[Sequence] = deque()
        self.finished_sequences: deque[Sequence] = deque()

    def add_request(self, sequence: Sequence):
        sequence.state = SequenceState.WAITING
        self.prefill_bucket.add(sequence)
        logger.debug(
            f"sequence: {sequence.s_id} added; # prefill: {len(self.prefill_bucket)}"
        )

    def get_finished_sequences(self) -> list[Sequence]:
        sequences = []
        while len(self.finished_sequences) > 0:
            sequences.append(self.finished_sequences.popleft())
        return sequences

    def schedule(self) -> list[Sequence]:
        logger.debug("scheduling...")
        batch = []
        if len(self.decode_sequences) > 0:
            while self.decode_sequences and len(batch) < self.batch_size:
                seq = self.decode_sequences.popleft()

                pages_needed = self._calculate_pages_needed(seq)
                if pages_needed:
                    if not self.page_manager.can_allocate(seq.s_id, pages_needed):
                        self.decode_sequences.appendleft(seq)
                        break

                seq.state = SequenceState.RUNNING
                batch.append(seq)
            logger.debug(f"scheduled {len(batch)} decode sequences")
            return batch

        prefill_batch = self.prefill_bucket.get_batch(self.batch_size)
        for i, sequence in enumerate(prefill_batch):
            pages_needed = self._calculate_pages_needed(sequence)
            if not self.page_manager.can_allocate(sequence.s_id, pages_needed):
                self.prefill_bucket.add(prefill_batch[i:][::-1], append="left")
                break

            sequence.state = SequenceState.RUNNING
            batch.append(sequence)

        logger.debug(f"scheduled {len(batch)} prefill sequences")
        return batch

    def update(self, sequences: list[Sequence]):
        assert all(sequence.last_token_id != -1 for sequence in sequences)
        while sequences:
            sequence = sequences.pop()
            if sequence.is_finished:
                sequence.state = SequenceState.FINISHED
                self.page_manager.free(sequence.s_id)
                self.finished_sequences.append(sequence)
                logger.info(f"{sequence.s_id} finished...")
                continue

            sequence.state = SequenceState.WAITING
            self.decode_sequences.append(sequence)

    def _calculate_pages_needed(self, sequence: Sequence) -> int:
        # not prefilled yet
        if sequence.last_token_id == -1:
            return math.ceil(len(sequence) / self._page_size)

        return int(not (len(sequence) - 1) % self._page_size)
