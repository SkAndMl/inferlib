from collections import deque

from inferlib.engine.sequence import Sequence


class Scheduler:
    def __init__(self):
        self.waiting_sequences = deque()
        self.running_sequences = deque()
        self.finished_sequences = deque()

    def add_request(self, sequence: Sequence):
        self.waiting_sequences.append(sequence)

    def get_finished_sequences(self) -> list[Sequence]:
        sequences = []
        while len(self.finished_sequences) > 0:
            sequences.append(self.finished_sequences.popleft())
        return sequences

    def schedule(self):
        pass
