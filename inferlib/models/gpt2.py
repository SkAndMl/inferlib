if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    # Allow running this file directly (e.g., `python inferlib/models/gpt2.py`)
    # by ensuring the repo root is on sys.path.
    import os
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import torch

from torch import nn, Tensor
from torch.nn import functional as F
from typing import Dict, Literal, Tuple

from inferlib.models.common import ModelConfig, PagedKVCache


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
    def __init__(self, config: ModelConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0
        self.cfg = config
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)

        self.c_proj.SCALE_INIT = True  # for residual scaling
        self.paged_kv_cache = None
        if config.use_kv_cache:
            self.paged_kv_cache = PagedKVCache(config)

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(size=(1, 1, config.n_ctx, config.n_ctx)) * float("-inf"),
                diagonal=1,
            ),
        )

    def _reset_cache(self):
        assert self.cfg.use_kv_cache
        self.paged_kv_cache.reset()

    def _online_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        assert q.shape[2] == k.shape[2] == v.shape[2] == 1

        running_denom = torch.zeros(
            size=(q.shape[0], self.cfg.n_head, 1, 1), device=q.device
        )
        running_max = torch.ones(
            size=(q.shape[0], self.cfg.n_head, 1, 1), device=q.device
        ) * float("-inf")
        y = torch.zeros_like(q)

        for i in range(self.paged_kv_cache.num_pages):
            k_i, v_i = self.paged_kv_cache.get_kv_page(page_idx=i)
            S_i: Tensor = (q @ k_i.transpose(-2, -1)) / (
                self.head_dim**0.5
            )  # bsz, head_dim, 1, pg_size
            max_i: Tensor = torch.max(
                S_i, dim=-1, keepdim=True
            ).values  # bsz, head_dim, 1, 1
            P_i = (S_i - max_i).exp()  # bsz, head_dim, 1, pg_size

            max_new = torch.maximum(running_max, max_i)  # bsz, head_dim, 1, 1
            P_i = (max_i - max_new).exp() * P_i
            denom_i = P_i.sum(dim=-1, keepdim=True)  # bsz, head_dim, 1, 1
            denom_new = (running_max - max_new).exp() * running_denom + denom_i
            y = ((running_denom * y * (running_max - max_new).exp()) / denom_new) + (
                P_i @ v_i
            ) / denom_new

            # update running stats
            running_denom = denom_new.clone()
            running_max = max_new.clone()

        S_i: Tensor = (q @ k.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # bsz, head_dim, 1, pg_size
        max_i: Tensor = torch.max(
            S_i, dim=-1, keepdim=True
        ).values  # bsz, head_dim, 1, 1
        P_i = (S_i - max_i).exp()  # bsz, head_dim, 1, pg_size

        max_new = torch.maximum(running_max, max_i)  # bsz, head_dim, 1, 1
        P_i = (max_i - max_new).exp() * P_i
        denom_i = P_i.sum(dim=-1, keepdim=True)  # bsz, head_dim, 1, 1
        denom_new = (running_max - max_new).exp() * running_denom + denom_i
        y = ((running_denom * y * (running_max - max_new).exp()) / denom_new) + (
            P_i @ v
        ) / denom_new

        self.paged_kv_cache.put_kv_page((k, v))

        return y

    def forward(self, x: Tensor, prefill: bool = False) -> Tensor:
        B, T, D = x.shape
        q_len, k_len = T, T
        n_head = self.cfg.n_head
        q, k, v = self.c_attn(x).split(D, dim=-1)
        q: Tensor = q.reshape(B, T, n_head, self.head_dim).transpose(1, 2)
        k: Tensor = k.reshape(B, T, n_head, self.head_dim).transpose(1, 2)
        v: Tensor = v.reshape(B, T, n_head, self.head_dim).transpose(1, 2)

        if prefill and self.cfg.use_kv_cache:
            for i in range(k_len):
                self.paged_kv_cache.put_kv_page(
                    (k[..., i : i + 1, :], v[..., i : i + 1, :])
                )

        if self.cfg.use_kv_cache and not prefill:
            y = self._online_attention(
                q[..., -1:, :],
                k[..., k_len - 1 : k_len, :],
                v[..., k_len - 1 : k_len, :],
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
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = MHA(config)
        self.mlp = MLP(config)
        self.ln_1 = LayerNorm(config)
        self.ln_2 = LayerNorm(config)
        self.resid_drop = nn.Dropout(p=config.resid_pdrop)

    def forward(self, x: Tensor, prefill: bool = False) -> Tensor:
        x = x + self.resid_drop(self.attn(self.ln_1(x), prefill))
        x = x + self.resid_drop(self.mlp(self.ln_2(x)))
        return x


class GPT2(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)
        self.emb_drop = nn.Dropout(p=config.embd_pdrop)

        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def forward(
        self, x: Tensor, cur_pos: int = 0, prefill: bool = False
    ) -> Tuple[Tensor, Dict[str, Tuple[Tensor, Tensor]] | None]:
        _, T = x.shape
        x = self.emb_drop(
            self.wte(x) + self.wpe(torch.arange(cur_pos, cur_pos + T).to(x.device))
        )
        for i, block in enumerate(self.h):
            x = block(x, prefill)

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
        self,
        *,
        x: Tensor,
    ):
        x_recent = x[:, -self.config.n_ctx :].clone()
        logits = self(x_recent, prefill=True)
        logits: Tensor = logits[:, -1:, :]
        next_tokens = logits.argmax(dim=-1)
        x = torch.cat((x, next_tokens), dim=-1).to(logits.device)
        return x

    def generate(
        self,
        *,
        tokens: list[list[int]],
        max_tokens: int = 200,
    ):
        assert len(tokens[0]) + max_tokens <= self.config.n_ctx, (
            f"max_tokens ({max_tokens}) exceeded model's ctx length ({self.config.n_ctx})"
        )
        # assume length is only 1 for now
        self.eval()
        with torch.inference_mode():
            x = torch.tensor(tokens, device=self.wte.weight.device)
            prompt_len = x.shape[1]
            x = self._prefill(x=x)
            for pos in range(max_tokens - 1):
                x_last = x[:, -1:].clone()
                logits = self(x_last, cur_pos=prompt_len + pos)
                next_tokens = logits.argmax(dim=-1)  # b, 1
                x = torch.cat((x, next_tokens), dim=-1).to(logits.device)

            self._reset_cache()

        return x.tolist()

    def _reset_cache(self):
        for block in self.h:
            block.attn._reset_cache()


if __name__ == "__main__":
    from tiktoken import get_encoding

    tokenizer = get_encoding("gpt2")
    model = GPT2.from_pretrained("medium")
    text = "Hi, how"
    tokens = [tokenizer.encode(text)]
    output_tokens = model.generate(tokens=tokens, max_tokens=10)
    print(tokenizer.decode(output_tokens[0]))
