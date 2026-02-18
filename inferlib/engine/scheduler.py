import math
from collections import deque

from inferlib.engine.page import PageManager
from inferlib.engine.sequence import Sequence, SequenceState
from inferlib.log import logger


class Scheduler:
    def __init__(self, page_manager: PageManager, batch_size: int = 4):
        self.page_manager = page_manager
        self._page_size = page_manager.page_size
        self.batch_size = batch_size
        self.prefill_sequences: deque[Sequence] = deque()
        self.decode_sequences: deque[Sequence] = deque()
        self.finished_sequences: deque[Sequence] = deque()

    def add_request(self, sequence: Sequence):
        sequence.state = SequenceState.WAITING
        self.prefill_sequences.append(sequence)
        logger.debug(
            f"sequence: {sequence.s_id} added; # prefill: {len(self.prefill_sequences)}"
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
        if len(self.prefill_sequences) > 0:
            while self.prefill_sequences and len(batch) < self.batch_size:
                seq = self.prefill_sequences.popleft()

                pages_needed = self._calculate_pages_needed(seq)
                if not self.page_manager.can_allocate(seq.s_id, pages_needed):
                    self.prefill_sequences.appendleft(seq)
                    break

                seq.state = SequenceState.RUNNING
                batch.append(seq)
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

        return int(not len(sequence) % self._page_size)
