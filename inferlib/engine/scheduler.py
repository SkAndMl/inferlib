import asyncio
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

    def get(self, bucket_idx: int) -> Sequence | None:
        if len(self) == 0:
            return None

        if len(self._buckets[bucket_idx]) > 0:
            self._total_sequences -= 1
            return self._buckets[bucket_idx].popleft()

        return None

    def __len__(self):
        return self._total_sequences

    @property
    def max_freq_bucket(self) -> int | None:
        if len(self) == 0:
            return None
        return max(self._buckets, key=lambda k: len(self._buckets[k]))


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

    async def schedule(self) -> list[Sequence]:
        if self.decode_sequences:
            return await self._get_decode_batch()

        return await self._get_prefill_batch()

    async def _get_decode_batch(self) -> list[Sequence]:
        batch: list[Sequence] = []
        awaited = False
        while len(batch) < self.batch_size:
            if not self.decode_sequences:
                if awaited:
                    break

                awaited = True
                await asyncio.sleep(0.1)
                continue

            seq = self.decode_sequences.popleft()
            pages_needed = self._calculate_pages_needed(sequence=seq)
            if pages_needed and not self.page_manager.can_allocate(
                s_id=seq.s_id, num_pages=pages_needed
            ):
                self.decode_sequences.appendleft(seq)
                break

            seq.state = SequenceState.RUNNING
            batch.append(seq)

        return batch

    async def _get_prefill_batch(self) -> list[Sequence]:
        batch: list[Sequence] = []
        bucket_idx = self.prefill_bucket.max_freq_bucket

        if bucket_idx is None:
            return batch

        awaited = False
        while len(batch) < self.batch_size:
            sequence = self.prefill_bucket.get(bucket_idx)
            if sequence is None:
                if awaited:
                    break

                awaited = True
                await asyncio.sleep(0.1)
                continue

            pages_needed = self._calculate_pages_needed(sequence=sequence)
            if not self.page_manager.can_allocate(sequence.s_id, pages_needed):
                self.prefill_bucket.add(sequences=sequence, append="left")
                break

            sequence.state = SequenceState.RUNNING
            batch.append(sequence)

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
