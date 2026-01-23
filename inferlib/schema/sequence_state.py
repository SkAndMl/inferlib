from dataclasses import dataclass, field
from typing import List


@dataclass
class SequenceState:
    seq_id: int
    seq_len: int = 0
    page_ids: List[int] = field(default_factory=list)
