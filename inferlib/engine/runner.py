from inferlib.models import Model
from inferlib.engine.sequence import Sequence
from inferlib.engine.page import PageManager


class Runner:
    def __init__(self, llm: Model, page_manager: PageManager):
        self.llm = llm
        self.page_manager = page_manager

    def run(self, sequences: list[Sequence]):
        if sequences[0].last_token_id == -1:
            # prefill
            assert all(seq.last_token_id == -1 for seq in sequences)
            self.llm.prefill(sequences=sequences, page_manager=self.page_manager)
            for seq in sequences:
                seq.last_token_id = seq.prompt_tokens[-1]
            return sequences

        assert not any(seq.last_token_id == -1 for seq in sequences)
        tokens = self.llm.decode(sequences=sequences, page_manager=self.page_manager)
        for token, sequence in zip(tokens, sequences):
            sequence.completion_tokens.append(token)
            sequence.last_token_id = token
        return sequences
