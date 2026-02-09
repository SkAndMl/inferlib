from inferlib.models import Model
from inferlib.engine.sequence import Sequence
from inferlib.engine.page import PageManager
from inferlib.log import logger


class Runner:
    def __init__(self, llm: Model, page_manager: PageManager):
        self.llm = llm
        self.page_manager = page_manager

    def run(self, sequences: list[Sequence]):
        logger.info(f"started running {len(sequences)} sequences")
        if sequences[0].last_token_id == -1:
            # prefill
            assert all(seq.last_token_id == -1 for seq in sequences)
            next_tokens = self.llm.prefill(
                sequences=sequences, page_manager=self.page_manager
            )
            for seq, next_token in zip(sequences, next_tokens):
                seq.completion_tokens.append(next_token)
                seq.last_token_id = next_token
            logger.info(f"finished prefilling {len(sequences)} sequences")
            return

        assert not any(seq.last_token_id == -1 for seq in sequences)
        tokens = self.llm.decode(sequences=sequences, page_manager=self.page_manager)
        for token, sequence in zip(tokens, sequences):
            sequence.completion_tokens.append(token)
            sequence.last_token_id = token

        logger.info(f"finished running {len(sequences)} sequences")
