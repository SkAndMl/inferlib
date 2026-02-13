import torch

from dataclasses import dataclass
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class Qwen3Config:
    attention_dropout: float = 0.0
    attention_bias: bool = False
    bos_token_id: int = 151643
    head_dim: int = 128
    hidden_size: int = 1024
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    max_position_embeddings: int = 40960
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    rope_scaling: float = None
    rope_theta: int = 1000000
    tie_word_embeddings: bool = True
    use_cache: bool = True
    vocab_size: int = 151936
    page_size: int = 16
    bsz: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "Qwen3Config":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def get_freqs_cis(cfg: Qwen3Config) -> Tensor:
    pos = torch.arange(0, cfg.max_position_embeddings)
    dim = cfg.head_dim
    thetas = cfg.rope_theta ** (-2 * torch.arange(0, dim // 2) / dim)

    freqs = torch.outer(pos, thetas)
    real = torch.cos(freqs)
    imag = torch.sin(freqs)
    return torch.complex(real, imag)


def apply_rot_emb(x: Tensor, freqs_cis: Tensor, start_pos: int = 0) -> Tensor:
    bsz, n_head, seq_len, head_dim = x.shape
    _x = x.view(bsz, n_head, seq_len, head_dim // 2, 2)
    _x = torch.view_as_complex(_x)  # bsz, n_head, seq_len, head_dim/2
    _freqs = freqs_cis[start_pos : start_pos + seq_len, :]  # seq_len, head_dim/2
    _x_rot = _x * _freqs[None, None, :, :]
    _x_rot = torch.view_as_real(_x_rot).view(bsz, n_head, seq_len, head_dim)
    return _x_rot


class SiLUActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, rms_norm_eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size=(hidden_size,)))
        self.eps = rms_norm_eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_norm = x / (rms + self.eps)
        return self.weight * x_norm


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg

        q_out = cfg.num_attention_heads * cfg.head_dim
        kv_out = cfg.num_key_value_heads * cfg.head_dim
        self.q_proj = nn.Linear(
            in_features=cfg.hidden_size,
            out_features=q_out,
            bias=cfg.attention_bias,
        )
        self.k_proj = nn.Linear(
            in_features=cfg.hidden_size,
            out_features=kv_out,
            bias=cfg.attention_bias,
        )
        self.v_proj = nn.Linear(
            in_features=cfg.hidden_size,
            out_features=kv_out,
            bias=cfg.attention_bias,
        )
        self.o_proj = nn.Linear(
            in_features=q_out,
            out_features=cfg.hidden_size,
            bias=cfg.attention_bias,
        )

        self.q_norm = Qwen3RMSNorm(
            hidden_size=cfg.head_dim, rms_norm_eps=cfg.rms_norm_eps
        )
        self.k_norm = Qwen3RMSNorm(
            hidden_size=cfg.head_dim, rms_norm_eps=cfg.rms_norm_eps
        )

        self.register_buffer("freqs_cis", get_freqs_cis(cfg), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q: Tensor = self.q_proj(x)
        k: Tensor = self.k_proj(x)
        v: Tensor = self.v_proj(x)

        q = q.view(B, T, self.cfg.num_attention_heads, self.cfg.head_dim).transpose(
            1, 2
        )
        k = k.view(B, T, self.cfg.num_key_value_heads, self.cfg.head_dim).transpose(
            1, 2
        )
        v = v.view(B, T, self.cfg.num_key_value_heads, self.cfg.head_dim).transpose(
            1, 2
        )

        q, k = self.q_norm(q), self.k_norm(k)

        k = k.repeat_interleave(
            self.cfg.num_attention_heads // self.cfg.num_key_value_heads, 1
        )
        v = v.repeat_interleave(
            self.cfg.num_attention_heads // self.cfg.num_key_value_heads, 1
        )

        q = apply_rot_emb(q, self.freqs_cis)
        k = apply_rot_emb(k, self.freqs_cis)

        mask: Tensor = torch.full(
            size=(T, T), fill_value=float("-inf"), device=x.device, dtype=x.dtype
        ).triu(1)

        attn: Tensor = q @ k.transpose(2, 3) / self.cfg.head_dim**0.5
        attn += mask
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        attn = F.dropout(attn, p=self.cfg.attention_dropout, training=self.training)

        y: Tensor = attn @ v  # B, N_Q_HEADS, T, HEAD_DIM
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(
            in_features=cfg.hidden_size, out_features=cfg.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            in_features=cfg.hidden_size, out_features=cfg.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            in_features=cfg.intermediate_size, out_features=cfg.hidden_size, bias=False
        )
        self.act_fn = SiLUActivation()

    def forward(self, x: Tensor):
        return self.down_proj(self.up_proj(x) * self.act_fn(self.gate_proj(x)))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(cfg)
        self.mlp = Qwen3MLP(cfg)
        self.input_layernorm = Qwen3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(
            hidden_size=cfg.hidden_size, rms_norm_eps=cfg.rms_norm_eps
        )
        self.lm_head = nn.Linear(
            in_features=cfg.hidden_size, out_features=cfg.vocab_size, bias=False
        )
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))
