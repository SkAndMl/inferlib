import asyncio
import math
from collections import deque
from typing import Literal

from inferlib.core.engine.page import PageManager
from inferlib.core.engine.sequence import Sequence, SequenceState
from inferlib.core.log import logger


class _Bucket:
    def __init__(self, page_size: int):
        self.page_size = page_size
        self._buckets: dict[int, deque[Sequence]] = {}
        self._skip_counts: dict[int, int] = {}
        self._total_sequences: int = 0

    def add(
        self,
        sequences: Sequence | list[Sequence],
        append: Literal["left", "right"] = "right",
    ):
        if isinstance(sequences, Sequence):
            sequences = [sequences]
        for sequence in sequences:
            # subtract evicted tokens
            num_tokens = len(sequence) - sequence.tokens_evicted
            bucket_idx = (num_tokens + self.page_size - 1) // self.page_size

            if bucket_idx not in self._buckets:
                self._buckets[bucket_idx] = deque()
                self._skip_counts[bucket_idx] = 0

            match append:
                case "right":
                    self._buckets[bucket_idx].append(sequence)
                case "left":
                    self._buckets[bucket_idx].appendleft(sequence)
            self._total_sequences += 1

    def get(self, bucket_idx: int) -> Sequence | None:
        if len(self) == 0:
            return None

        bucket = self._buckets.get(bucket_idx)
        if bucket is None:
            return None

        if len(bucket) > 0:
            self._total_sequences -= 1
            sequence = bucket.popleft()
            if len(bucket) == 0:
                del self._skip_counts[bucket_idx]
                del self._buckets[bucket_idx]
            return sequence

        return None

    def __len__(self):
        return self._total_sequences

    def __bool__(self):
        return self._total_sequences > 0

    @property
    def max_freq_bucket(self) -> int | None:
        if len(self) == 0:
            return None

        _max_bucket = max(self._buckets, key=lambda k: len(self._buckets[k]) + self._skip_counts[k])
        self._skip_counts[_max_bucket] = 0
        for k in self._skip_counts:
            if k == _max_bucket:
                continue
            self._skip_counts[k] += 1

        return _max_bucket


class Scheduler:
    def __init__(
        self,
        page_manager: PageManager,
        request_event: asyncio.Event,
        max_pages_per_sequence: int,
        batch_size: int = 4,
        max_active_sequences: int = 8,
    ):
        self.page_manager = page_manager
        self._page_size = page_manager.page_size
        self.batch_size = batch_size
        self.request_event = request_event
        self.max_active_sequences = max_active_sequences
        self.max_pages_per_sequence = max_pages_per_sequence
        self._prefill_bucket = _Bucket(self._page_size)
        self._decode_bucket = _Bucket(self._page_size)

        self._active_sequences = set()
        self._prefill_before_decode: int = 0
        self._max_prefill_before_decode: int = 2

    def add_request(self, sequence: Sequence) -> bool:
        sequence.state = SequenceState.WAITING
        # reject prompts that exceed the max number of pages for a sequence
        if self._calculate_pages_needed(sequence) > self.max_pages_per_sequence:
            return False

        self._prefill_bucket.add(sequence)
        self.request_event.set()
        logger.debug(f"sequence: {sequence.s_id} added; # prefill: {len(self._prefill_bucket)}")
        return True

    async def schedule(self) -> list[Sequence]:
        batch = []
        if self._prefill_bucket and self._prefill_before_decode < self._max_prefill_before_decode:
            batch = await self._get_batch("prefill")
            # check if we are actually able to get prefill sequences, if not fall through to decode
            if len(batch) > 0:
                self._prefill_before_decode += 1
                return batch

        if self._decode_bucket:
            self._prefill_before_decode = 0
            return await self._get_batch("decode")

        return []

    async def _get_batch(self, bucket_type: Literal["prefill", "decode"]) -> list[Sequence]:
        batch: list[Sequence] = []
        bucket = self._prefill_bucket if bucket_type == "prefill" else self._decode_bucket
        bucket_idx = bucket.max_freq_bucket
        if bucket_idx is None:
            return batch

        while len(batch) < self.batch_size:
            # break if # active sequences == max active sequences
            # break only for "prefill"; "decode" contains only already counted active sequences
            if (
                bucket_type == "prefill"
                and len(self._active_sequences) >= self.max_active_sequences
            ):
                break

            sequence = bucket.get(bucket_idx)
            if sequence is None:
                break

            pages_needed = self._calculate_pages_needed(sequence=sequence)
            if (
                bucket_type == "decode"
                and self.page_manager.get_num_pages(sequence.s_id) == self.max_pages_per_sequence
                and pages_needed
            ):
                self.page_manager.evict(sequence.s_id)
                sequence.tokens_evicted += self._page_size

            if not self.page_manager.reserve(sequence.s_id, pages_needed):
                bucket.add(sequences=sequence, append="left")
                break

            self.page_manager.commit(sequence.s_id)
            sequence.state = SequenceState.RUNNING
            self._active_sequences.add(sequence.s_id)
            batch.append(sequence)

        return batch

    def update(self, sequences: list[Sequence]):
        assert all(sequence.last_token_id != -1 for sequence in sequences)
        for sequence in sequences:
            if sequence.is_finished:
                sequence.state = SequenceState.FINISHED
                self.page_manager.free(sequence.s_id)
                self._active_sequences.remove(sequence.s_id)
                logger.info(f"{sequence.s_id} finished...")
                continue

            sequence.state = SequenceState.WAITING
            self._decode_bucket.add(sequence)

    def _calculate_pages_needed(self, sequence: Sequence) -> int:
        num_tokens = len(sequence) - sequence.tokens_evicted
        # not prefilled yet
        if sequence.last_token_id == -1:
            return math.ceil(num_tokens / self._page_size)

        # subtract 1 from num_tokens, because the last token does not have a kv cache yet
        return int(not (num_tokens - 1) % self._page_size)

    @property
    def prefill_empty(self) -> bool:
        return len(self._prefill_bucket) == 0

    @property
    def decode_empty(self) -> bool:
        return len(self._decode_bucket) == 0
