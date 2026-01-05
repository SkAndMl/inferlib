import torch

from dataclasses import dataclass
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Dict, Literal, Tuple


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
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cfg = config
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)

        self.c_proj.SCALE_INIT = True  # for residual scaling

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(size=(1, 1, config.n_ctx, config.n_ctx)) * float("-inf"),
                diagonal=1,
            ),
        )

    def forward(
        self,
        x: Tensor,
        use_kv_cache: bool = True,
        kv_cache: Tuple[Tensor, Tensor] = None,
    ) -> Tensor:
        B, T, D = x.shape
        q_len, k_len = T, T
        n_head, head_dim = self.cfg.n_head, D // self.cfg.n_head
        q, k, v = self.c_attn(x).split(D, dim=-1)
        q: Tensor = q.reshape(B, T, n_head, head_dim).transpose(1, 2)
        k: Tensor = k.reshape(B, T, n_head, head_dim).transpose(1, 2)
        v: Tensor = v.reshape(B, T, n_head, head_dim).transpose(1, 2)

        if use_kv_cache and kv_cache is not None:
            k_old, v_old = kv_cache
            k = torch.cat([k_old, k], dim=2).to(k.device)
            v = torch.cat([v_old, v], dim=2).to(v.device)

            k_len += k_old.shape[2]

        attn_wts = q @ k.transpose(-2, -1) / (head_dim**0.5)  # B, H, N, N
        attn_wts = attn_wts + self.mask[:, :, k_len - q_len : k_len, :k_len]
        attn_wts = F.softmax(attn_wts, dim=-1)
        attn_wts = self.attn_drop(attn_wts)

        y = attn_wts @ v  # B, H, N, head_dim
        y = y.transpose(1, 2).contiguous().view(B, T, D)

        if use_kv_cache:
            kv_cache = (k, v)

        return self.c_proj(y), kv_cache


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.SCALE_INIT = True

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = MHA(config)
        self.mlp = MLP(config)
        self.ln_1 = LayerNorm(config)
        self.ln_2 = LayerNorm(config)
        self.resid_drop = nn.Dropout(p=config.resid_pdrop)

    def forward(
        self,
        x: Tensor,
        use_kv_cache: bool = False,
        kv_cache: Tuple[Tensor, Tensor] = None,
    ) -> Tensor:
        attn_x, kv_cache = self.attn(self.ln_1(x), use_kv_cache, kv_cache)
        x = x + self.resid_drop(attn_x)
        x = x + self.resid_drop(self.mlp(self.ln_2(x)))
        return x, kv_cache


class GPT2(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)
        self.emb_drop = nn.Dropout(p=config.embd_pdrop)

        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def forward(
        self,
        x: Tensor,
        use_kv_cache: bool = True,
        kv_caches: Dict[int, Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Tuple[Tensor, Tensor]] | None]:
        if use_kv_cache and kv_caches is None:
            kv_caches = {}

        _, T = x.shape
        cur_pos = 0
        if use_kv_cache and kv_caches.get(0, None) is not None:
            cur_pos = kv_caches[0][0].shape[2]
        x = self.emb_drop(
            self.wte(x) + self.wpe(torch.arange(cur_pos, cur_pos + T).to(x.device))
        )
        for i, block in enumerate(self.h):
            x, kv_cache = block(x, use_kv_cache, kv_caches.get(i, None))
            kv_caches[i] = kv_cache

        return self.lm_head(self.ln_f(x)), kv_caches

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
        use_kv_cache: bool = True,
        kv_caches: Dict[int, Tuple[Tensor, Tensor]] = None,
    ):
        x_recent = x[:, -self.config.n_ctx :].clone()
        logits, kv_caches = self(x_recent, use_kv_cache, kv_caches)
        logits: Tensor = logits[:, -1:, :]
        next_tokens = logits.argmax(dim=-1)
        x = torch.cat((x, next_tokens), dim=-1).to(logits.device)
        return x, kv_caches

    def generate(
        self,
        *,
        tokens: list[list[int]],
        max_tokens: int = 200,
        use_kv_cache: bool = True,
        kv_caches: Dict[int, Tuple[Tensor, Tensor]] = None,
    ):
        # assume length is only 1 for now
        self.eval()
        with torch.inference_mode():
            x = torch.tensor(tokens, device=self.wte.weight.device)
            x, kv_caches = self._prefill(
                x=x, use_kv_cache=use_kv_cache, kv_caches=kv_caches
            )
            for _ in range(max_tokens - 1):
                x_last = x[:, -1:].clone()
                logits, kv_caches = self(x_last, use_kv_cache, kv_caches)
                next_tokens = logits.argmax(dim=-1)  # b, 1
                x = torch.cat((x, next_tokens), dim=-1).to(logits.device)

        return x.tolist()
