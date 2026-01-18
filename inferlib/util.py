import torch
from typing import List, Tuple


def generated_padded_batch(
    sequences: List[List[int]], pad_token_id: int
) -> Tuple[torch.Tensor, List[int]]:
    max_len = max(len(_) for _ in sequences)
    bsz = len(sequences)
    starting_positions = []
    padded_tokens = torch.full(size=(bsz, max_len), fill_value=pad_token_id)
    for i, sequence in enumerate(sequences):
        padded_tokens[i, -len(sequence) :] = torch.tensor(sequence)
        starting_positions.append(max_len - len(sequence))
    return padded_tokens, starting_positions
