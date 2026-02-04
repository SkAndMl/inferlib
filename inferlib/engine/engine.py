import tiktoken
import torch
from typing import Any

from inferlib.models import GPT2

from inferlib.engine.runner import Runner
from inferlib.engine.scheduler import Scheduler
from inferlib.engine.sequence import Sequence
from inferlib.engine.page import PageManager


class InferlibEngine:
    def __init__(self):
        self.llm = GPT2.from_pretrained("small")
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.model_config = self.llm.config
        self.page_manager = PageManager(
            num_pages=128,
            num_layers=self.model_config.n_layer,
            num_heads=self.model_config.n_head,
            page_size=16,
            head_dim=self.model_config.n_head,
            dtype=torch.float16,
            device=torch.device("cpu"),
        )
        self.scheduler = Scheduler(page_manager=self.page_manager)
        self.runner = Runner(llm=self.llm, page_manager=self.page_manager)

    def add_request(self, payload: dict[str, Any]):
        assert "message" in payload
        assert "id" in payload
        prompt_tokens: list[int] = self.tokenizer.encode(payload["message"])
        sequence = Sequence(
            s_id=payload["id"],
            prompt_tokens=prompt_tokens,
            completion_tokens=[],
            eos_token_id=self.tokenizer.eot_token,
        )
        self.scheduler.add_request(sequence)
