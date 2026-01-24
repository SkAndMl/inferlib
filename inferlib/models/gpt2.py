if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    # Allow running this file directly (e.g., `python inferlib/models/gpt2.py`)
    # by ensuring the repo root is on sys.path.
    import os
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import torch

from dataclasses import dataclass
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Dict, List, Literal, Tuple

from inferlib.cache import PagedKVCache
from inferlib.schema.sequence_state import SequenceState
from inferlib.util import generated_padded_batch


@dataclass
class ModelConfig:
    attn_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-05
    n_ctx: int = 1024
    n_embd: int = 1024
    n_head: int = 16
    n_layer: int = 24
    n_positions: int = 1024
    resid_pdrop: int = 0.1
    vocab_size: int = 50257
    use_kv_cache: bool = True
    page_size: int = 16
    bsz: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class LayerNorm(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_eps = config.layer_norm_epsilon
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd))

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(dim=-1, keepdim=True)  # B, T, 1
        var = x.var(dim=-1, keepdim=True, correction=0)  # B, T, 1

        x_norm = (x - mu) * (var + self.ln_eps).rsqrt()  # B, T, D
        return self.weight * x_norm + self.bias


class MHA(nn.Module):
    def __init__(self, config: ModelConfig, _layer_id: int):
        super().__init__()

        assert config.n_embd % config.n_head == 0
        self.cfg = config
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)

        self.c_proj.SCALE_INIT = True  # for residual scaling
        self.paged_kv_cache = None

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(size=(1, 1, config.n_ctx, config.n_ctx)) * float("-inf"),
                diagonal=1,
            ),
        )

        self._layer_id = _layer_id

    def _online_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cache: PagedKVCache,
        sequence_states: List[SequenceState],
    ) -> Tensor:
        assert q.shape[2] == k.shape[2] == v.shape[2] == 1

        running_denom = torch.zeros(
            size=(q.shape[0], self.cfg.n_head, 1, 1),
            device=q.device,
            dtype=torch.float32,
        )
        running_max = torch.ones(
            size=(q.shape[0], self.cfg.n_head, 1, 1),
            device=q.device,
            dtype=torch.float32,
        ) * float("-inf")
        y = torch.zeros_like(q, dtype=torch.float32)

        qf = q.to(torch.float32)
        kf_cur = k.to(torch.float32)
        vf_cur = v.to(torch.float32)

        for k_i, v_i in cache.read_pages(sequence_states, self._layer_id):
            kf, vf = k_i.to(torch.float32), v_i.to(torch.float32)

            S_i: Tensor = (qf @ kf.transpose(-2, -1)) / (
                self.head_dim**0.5
            )  # bsz, head_dim, 1, pg_size
            max_i: Tensor = torch.max(
                S_i, dim=-1, keepdim=True
            ).values  # bsz, head_dim, 1, 1

            max_new = torch.maximum(running_max, max_i)  # bsz, head_dim, 1, 1
            P_i = (S_i - max_new).exp()
            denom_i = P_i.sum(dim=-1, keepdim=True)  # bsz, head_dim, 1, 1

            alpha = (running_max - max_new).exp()
            running_denom = alpha * running_denom + denom_i
            y = y * alpha + (P_i @ vf)

            # update max
            running_max = max_new

        S_i: Tensor = (qf @ kf_cur.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # bsz, head_dim, 1, pg_size
        max_i: Tensor = torch.max(
            S_i, dim=-1, keepdim=True
        ).values  # bsz, head_dim, 1, 1

        max_new = torch.maximum(running_max, max_i)  # bsz, head_dim, 1, 1
        P_i = (S_i - max_new).exp()
        denom_i = P_i.sum(dim=-1, keepdim=True)  # bsz, head_dim, 1, 1

        alpha = (running_max - max_new).exp()
        running_denom = alpha * running_denom + denom_i
        y = y * alpha + (P_i @ vf_cur)
        y /= running_denom

        cache.write(sequence_states, self._layer_id, (k, v))
        return y.to(q.dtype)

    def forward(
        self,
        x: Tensor,
        cache: PagedKVCache,
        sequence_states: List[SequenceState],
        prefill: bool = False,
    ) -> Tensor:
        B, T, D = x.shape
        q_len, k_len = T, T
        n_head = self.cfg.n_head
        q, k, v = self.c_attn(x).split(D, dim=-1)
        q: Tensor = q.reshape(B, T, n_head, self.head_dim).transpose(1, 2)
        k: Tensor = k.reshape(B, T, n_head, self.head_dim).transpose(1, 2)
        v: Tensor = v.reshape(B, T, n_head, self.head_dim).transpose(1, 2)

        if prefill and self.cfg.use_kv_cache:
            cache.prefill(sequence_states, self._layer_id, (k, v))

        if self.cfg.use_kv_cache and not prefill:
            y = self._online_attention(
                q[..., -1:, :],
                k[..., k_len - 1 : k_len, :],
                v[..., k_len - 1 : k_len, :],
                cache=cache,
                sequence_states=sequence_states,
            )
        else:
            attn_wts = q @ k.transpose(-2, -1) / (self.head_dim**0.5)  # B, H, N, N
            attn_wts = attn_wts + self.mask[:, :, k_len - q_len : k_len, :k_len]
            attn_wts = F.softmax(attn_wts, dim=-1)
            attn_wts = self.attn_drop(attn_wts)
            y = attn_wts @ v  # B, H, N, head_dim

        _seq = 1 if self.cfg.use_kv_cache and not prefill else T
        y = y.transpose(1, 2).contiguous().view(B, _seq, D)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.SCALE_INIT = True

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, config: ModelConfig, _layer_id: int):
        super().__init__()
        self.attn = MHA(config, _layer_id)
        self.mlp = MLP(config)
        self.ln_1 = LayerNorm(config)
        self.ln_2 = LayerNorm(config)
        self.resid_drop = nn.Dropout(p=config.resid_pdrop)

        self._layer_id = _layer_id

    def forward(
        self,
        x: Tensor,
        cache: PagedKVCache,
        sequence_states: List[SequenceState],
        prefill: bool = False,
    ) -> Tensor:
        x = x + self.resid_drop(
            self.attn(self.ln_1(x), cache, sequence_states, prefill)
        )
        x = x + self.resid_drop(self.mlp(self.ln_2(x)))
        return x


class GPT2(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)
        self.emb_drop = nn.Dropout(p=config.embd_pdrop)

        self.h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.ln_f = LayerNorm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def forward(
        self,
        x: Tensor,
        cache: PagedKVCache,
        sequence_states: List[SequenceState],
        cur_pos: int = 0,
        prefill: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tuple[Tensor, Tensor]] | None]:
        _, T = x.shape
        x = self.emb_drop(
            self.wte(x) + self.wpe(torch.arange(cur_pos, cur_pos + T).to(x.device))
        )
        for block in self.h:
            x = block(x, cache, sequence_states, prefill)

        return self.lm_head(self.ln_f(x))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = self.config.initializer_range
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )

    @classmethod
    def from_pretrained(
        cls, model_class: Literal["small", "medium", "large", "xl"]
    ) -> "GPT2":
        from transformers import GPT2Config, GPT2Model

        model_name = "gpt2" + ("" if model_class == "small" else f"-{model_class}")
        config = GPT2Config.from_pretrained(model_name)
        model = GPT2Model.from_pretrained(model_name)

        sd = model.state_dict()
        transposed = {
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        }

        for k in sd:
            if any(k.endswith(_) for _ in transposed):
                sd[k] = sd[k].t()

        model_cfg = ModelConfig.from_dict(config.to_dict())
        model_instance = GPT2(model_cfg)
        model_instance.load_state_dict(sd, strict=False)
        return model_instance

    def _prefill(
        self, *, x: Tensor, cache: PagedKVCache, sequence_states: List[SequenceState]
    ):
        x_recent = x[:, -self.config.n_ctx :].clone()
        logits = self(x_recent, cache, sequence_states, prefill=True)
        logits: Tensor = logits[:, -1:, :]
        next_tokens = logits.argmax(dim=-1)
        x = torch.cat((x, next_tokens), dim=-1).to(logits.device)
        return x

    def generate(
        self,
        *,
        sequences: list[list[int]],
        cache: PagedKVCache,
        sequence_states: List[SequenceState],
        max_tokens: int = 200,
        temperature: float = 0.1,
        pad_token_id: int = 50256,
    ):
        for sequence in sequences:
            assert len(sequence) + max_tokens <= self.config.n_ctx

        # assume length is only 1 for now
        self.eval()
        with torch.inference_mode():
            x, starting_positions = generated_padded_batch(sequences, pad_token_id)
            x = x.to(device=self.wte.weight.device)
            prompt_len = x.shape[1]
            x = self._prefill(x=x, cache=cache, sequence_states=sequence_states)
            for pos in range(max_tokens - 1):
                x_last = x[:, -1:].clone()
                logits: Tensor = self(
                    x_last, cache, sequence_states, cur_pos=prompt_len + pos
                )
                probs = F.softmax(logits / temperature, dim=-1)

                next_tokens = torch.multinomial(probs[:, -1, :], 1)  # b, 1
                x = torch.cat((x, next_tokens), dim=-1).to(logits.device)

        decoded_sequences = x.tolist()
        decoded_sequences = [
            d[s:] for d, s in zip(decoded_sequences, starting_positions)
        ]

        cache.free_pages(sequence_states)
        return decoded_sequences


if __name__ == "__main__":
    from tiktoken import get_encoding
    from inferlib.cache import PagePool

    tokenizer = get_encoding("gpt2")
    model = GPT2.from_pretrained("small")

    page_size = 16

    sequence_states = [SequenceState(seq_id="s1"), SequenceState(seq_id="s2")]
    page_pool = PagePool(
        num_pages=128,
        num_layers=model.config.n_layer,
        num_heads=model.config.n_head,
        page_size=page_size,
        head_dim=model.config.n_embd // model.config.n_head,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    cache = PagedKVCache(
        page_pool=page_pool,
        num_layers=model.config.n_layer,
        page_size=page_size,
        device="cpu",
    )

    text = "Hi, how"
    sequences = [tokenizer.encode(text), tokenizer.encode(text)]
    output_sequences = model.generate(
        sequences=sequences, max_tokens=10, cache=cache, sequence_states=sequence_states
    )

    for i in range(len(output_sequences)):
        print(f"Seq id: {sequence_states[i].seq_id}")
        print(f"Output: {tokenizer.decode(output_sequences[i])}")
        print("=" * 20)
