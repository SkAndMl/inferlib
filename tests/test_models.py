import sys
import torch

from pathlib import Path
from tiktoken import get_encoding
from transformers import GPT2Model

ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(ROOT))

from inferlib.models import GPT2  # noqa: E402


def hf_logits_from_gpt2model(gpt2_model: GPT2Model, input_ids: torch.Tensor, **kwargs):
    out = gpt2_model(input_ids=input_ids, **kwargs)
    hidden = out.last_hidden_state  # (B, T, D), already includes ln_f in HF GPT2Model
    logits = hidden @ gpt2_model.wte.weight.T  # (B, T, V)
    return logits, out.past_key_values


def test_logits_equivalence_full_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_encoding("gpt2")

    gpt = GPT2.from_pretrained("small").to(device).eval()
    gpt_hf = GPT2Model.from_pretrained("gpt2").to(device).eval()

    text = "Hello"
    x = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)

    with torch.no_grad():
        our_logits, _ = gpt(x, use_kv_cache=False)
        hf_logits, _ = hf_logits_from_gpt2model(gpt_hf, x, use_cache=False)

    torch.testing.assert_close(our_logits, hf_logits, rtol=1e-3, atol=1e-3)


def test_logits_equivalence_with_kv_cache_incremental():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_encoding("gpt2")

    gpt = GPT2.from_pretrained("small").to(device).eval()
    gpt_hf = GPT2Model.from_pretrained("gpt2").to(device).eval()

    text = "Hello, how are"
    x = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)

    x_prefill = x[:, :-1]
    x_last = x[:, -1:]

    with torch.no_grad():
        our_logits_prefill, our_cache = gpt(x_prefill, use_kv_cache=True, kv_caches={})

        our_logits_step, our_cache2 = gpt(
            x_last, use_kv_cache=True, kv_caches=our_cache
        )

        hf_logits_prefill, hf_past = hf_logits_from_gpt2model(
            gpt_hf, x_prefill, use_cache=True
        )
        hf_logits_step, hf_past2 = hf_logits_from_gpt2model(
            gpt_hf, x_last, use_cache=True, past_key_values=hf_past
        )

    torch.testing.assert_close(
        our_logits_step[:, -1, :], hf_logits_step[:, -1, :], rtol=1e-3, atol=1e-3
    )

    for layer_idx in range(gpt.config.n_layer):
        k_ours, v_ours = our_cache2[layer_idx]
        k_hf, v_hf = hf_past2[layer_idx]

        torch.testing.assert_close(k_ours, k_hf, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(v_ours, v_hf, rtol=1e-3, atol=1e-3)
