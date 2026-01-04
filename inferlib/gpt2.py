import torch

from dataclasses import dataclass
from torch import nn, Tensor
from torch.nn import functional as F


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


class LayerNorm(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_eps = config.layer_norm_epsilon
        self.gain = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd))

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(dim=-1, keepdim=True)  # B, T, 1
        var = x.var(dim=-1, keepdim=True, correction=0)  # B, T, 1

        x_norm = (x - mu) * (var + self.ln_eps).rsqrt()  # B, T, D
        return self.gain * x_norm + self.bias


class MHA(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cfg = config
        self.Wq = nn.Linear(config.n_embd, config.n_embd)
        self.Wk = nn.Linear(config.n_embd, config.n_embd)
        self.Wv = nn.Linear(config.n_embd, config.n_embd)
        self.Wo = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(size=(1, 1, config.n_ctx, config.n_ctx)) * float("-inf"),
                diagonal=1,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        n_head, head_dim = self.cfg.n_head, D // self.cfg.n_head
        q: Tensor = self.Wq(x).reshape(B, T, n_head, head_dim).transpose(1, 2)
        k: Tensor = self.Wk(x).reshape(B, T, n_head, head_dim).transpose(1, 2)
        v: Tensor = self.Wv(x).reshape(B, T, n_head, head_dim).transpose(1, 2)

        attn_wts = q @ k.transpose(-2, -1) / (head_dim**0.5)  # B, H, N, N
        attn_wts = attn_wts + self.mask[:, :, :T, :T]
        attn_wts = F.softmax(attn_wts, dim=-1)
        attn_wts = self.attn_drop(attn_wts)

        y = attn_wts @ v  # B, H, N, head_dim
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(y)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.GELU(),
            nn.Linear(config.n_embd * 4, config.n_embd),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mha = MHA(config)
        self.mlp = MLP(config)
        self.ln1 = LayerNorm(config)
        self.ln2 = LayerNorm(config)
        self.resid_drop = nn.Dropout(p=config.resid_pdrop)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.resid_drop(self.mha(self.ln1(x)))
        x = x + self.resid_drop(self.mlp(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.n_ctx, config.n_embd)
        self.emb_drop = nn.Dropout(p=config.embd_pdrop)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, x: Tensor) -> Tensor:
        _, T = x.shape
        x = self.emb_drop(
            self.tok_emb(x) + self.pos_emb(torch.arange(0, T).to(x.device))
        )
        for block in self.blocks:
            x = block(x)

        return self.lm_head(self.ln_f(x))
