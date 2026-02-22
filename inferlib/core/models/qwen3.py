import torch

from dataclasses import dataclass
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from inferlib.core.engine.page import PageManager
from inferlib.core.engine.sequence import Sequence
from inferlib.core.models._base import Model


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


def get_freqs_cis(cfg: Qwen3Config):
    pos = torch.arange(0, cfg.max_position_embeddings)
    dim = cfg.head_dim
    thetas = cfg.rope_theta ** (-2 * torch.arange(0, dim // 2) / dim)
    freqs = torch.outer(pos, thetas)
    return torch.cos(freqs), torch.sin(freqs)  # each: [max_pos, head_dim//2]


def apply_rot_emb(x: Tensor, cos: Tensor, sin: Tensor, start_positions: Tensor = None):
    bsz, _, seq_len, _ = x.shape
    t = torch.arange(seq_len, device=x.device, dtype=torch.long)[None, :]
    if start_positions is None:
        position_ids = torch.zeros((bsz, 1), device=x.device, dtype=torch.long) + t
    else:
        position_ids = start_positions[:, None] + t
    cos_bt = cos[position_ids][:, None, :, :]
    sin_bt = sin[position_ids][:, None, :, :]
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([x1 * cos_bt - x2 * sin_bt, x2 * cos_bt + x1 * sin_bt], dim=-1)


class SiLUActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, rms_norm_eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size=(hidden_size,)))
        self.eps = rms_norm_eps

    def forward(self, x: Tensor) -> Tensor:
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        x_norm = x / rms
        return self.weight * x_norm


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config, _layer_id: int = 0):
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

        cos, sin = get_freqs_cis(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self._layer_id = _layer_id

    def _online_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        sequences: list[Sequence],
        page_manager: PageManager,
    ) -> Tensor:
        bsz = q.shape[0]
        running_denom = torch.zeros(
            size=(bsz, self.cfg.num_attention_heads, 1, 1),
            device=q.device,
            dtype=torch.float32,
        )
        running_max = torch.ones(
            size=(bsz, self.cfg.num_attention_heads, 1, 1),
            device=q.device,
            dtype=torch.float32,
        ) * float("-inf")
        y = torch.zeros_like(q, dtype=torch.float32)

        qf = q.to(torch.float32)
        kf_cur = k.to(torch.float32).repeat_interleave(
            self.cfg.num_attention_heads // self.cfg.num_key_value_heads, 1
        )
        vf_cur = v.to(torch.float32).repeat_interleave(
            self.cfg.num_attention_heads // self.cfg.num_key_value_heads, 1
        )

        num_pages = page_manager.get_num_pages(sequences[0].s_id)

        for i, (k_i, v_i) in enumerate(
            page_manager.read_pages(sequences=sequences, layer_id=self._layer_id)
        ):
            if i == num_pages - 1:
                lens = [(len(seq) - 1) % page_manager.page_size for seq in sequences]
                lens = [_l if _l > 0 else page_manager.page_size for _l in lens]
                lens = torch.tensor(
                    lens,
                    dtype=torch.long,
                    device=q.device,
                )
                indices = torch.arange(
                    page_manager.page_size, dtype=torch.long, device=q.device
                ).unsqueeze(0)
                mask = indices < lens.unsqueeze(1)
            else:
                mask = torch.ones(
                    size=(q.shape[0], page_manager.page_size), device=q.device
                ).bool()
            mask = mask[:, None, None, :]  # bsz, 1, 1, pg_size

            kf, vf = k_i.to(torch.float32), v_i.to(torch.float32)
            kf = kf.repeat_interleave(
                self.cfg.num_attention_heads // self.cfg.num_key_value_heads, 1
            )
            vf = vf.repeat_interleave(
                self.cfg.num_attention_heads // self.cfg.num_key_value_heads, 1
            )

            S_i: Tensor = (qf @ kf.transpose(-2, -1)) / (
                self.cfg.head_dim**0.5
            )  # bsz, head_dim, 1, pg_size
            S_i.masked_fill_(~mask, float("-inf"))
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
            self.cfg.head_dim**0.5
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

        page_manager.write(sequences, self._layer_id, (k, v))
        return y.to(q.dtype)

    def forward(
        self,
        x: Tensor,
        sequences: list[Sequence],
        page_manager: PageManager,
        prefill: bool = False,
        start_positions: Tensor = None,
        mask: Tensor = None,
    ) -> Tensor:
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

        q = apply_rot_emb(
            q, cos=self.cos, sin=self.sin, start_positions=start_positions
        )
        k = apply_rot_emb(
            k, cos=self.cos, sin=self.sin, start_positions=start_positions
        )

        if prefill and self.cfg.use_cache:
            page_manager.prefill(
                sequences=sequences, layer_id=self._layer_id, kv=(k, v)
            )

        if self.cfg.use_cache and not prefill:
            y = self._online_attention(
                q[..., -1:, :],
                k[..., -1:, :],
                v[..., -1:, :],
                sequences=sequences,
                page_manager=page_manager,
            )

        else:
            k = k.repeat_interleave(
                self.cfg.num_attention_heads // self.cfg.num_key_value_heads, 1
            )
            v = v.repeat_interleave(
                self.cfg.num_attention_heads // self.cfg.num_key_value_heads, 1
            )

            if mask is None:
                mask: Tensor = torch.full(
                    size=(1, T, T),
                    fill_value=1,
                    device=x.device,
                    dtype=torch.bool,
                ).tril(0)
            else:
                assert mask.shape == (B, T, T)

            attn: Tensor = q @ k.transpose(2, 3) / self.cfg.head_dim**0.5
            attn.masked_fill_(~mask[:, None, ...], value=float("-inf"))
            attn = F.softmax(attn.float(), dim=-1).to(q.dtype).nan_to_num(nan=0.0)
            attn = F.dropout(attn, p=self.cfg.attention_dropout, training=self.training)

            y: Tensor = attn @ v  # B, N_Q_HEADS, T, HEAD_DIM

        _seq = 1 if self.cfg.use_cache and not prefill else T
        y = y.transpose(1, 2).contiguous().view(B, _seq, -1)
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
    def __init__(self, cfg: Qwen3Config, _layer_id: int = 0):
        super().__init__()
        self.self_attn = Qwen3Attention(cfg, _layer_id=_layer_id)
        self.mlp = Qwen3MLP(cfg)
        self.input_layernorm = Qwen3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        sequences: list[Sequence],
        page_manager: PageManager,
        prefill: bool = False,
        start_positions: Tensor = None,
        mask: Tensor = None,
    ) -> Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x),
            sequences=sequences,
            page_manager=page_manager,
            prefill=prefill,
            start_positions=start_positions,
            mask=mask,
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3(nn.Module, Model):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(
            hidden_size=cfg.hidden_size, rms_norm_eps=cfg.rms_norm_eps
        )
        self.lm_head = nn.Linear(
            in_features=cfg.hidden_size, out_features=cfg.vocab_size, bias=False
        )
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        x: Tensor,
        sequences: list[Sequence],
        page_manager: PageManager,
        prefill: bool = False,
        start_positions: Tensor = None,
        mask: Tensor = None,
    ) -> Tensor:
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(
                x,
                sequences=sequences,
                page_manager=page_manager,
                prefill=prefill,
                start_positions=start_positions,
                mask=mask,
            )
        return self.lm_head(self.norm(x))

    @classmethod
    def from_pretrained(
        cls, model_class="Qwen/Qwen3-0.6B"
    ) -> tuple["Qwen3", Qwen3Config]:
        hf_cfg = AutoConfig.from_pretrained(model_class).to_dict()
        cfg = Qwen3Config.from_dict(hf_cfg)
        hf_sd = AutoModelForCausalLM.from_pretrained(model_class).state_dict()
        keys = sorted(hf_sd.keys())
        for k in keys:
            if k.startswith("model."):
                hf_sd[k[len("model.") :]] = hf_sd.pop(k)

        model = Qwen3(cfg)
        model.load_state_dict(hf_sd)
        return model, cfg

    @torch.inference_mode()
    def prefill(
        self,
        *,
        sequences: list[Sequence],
        page_manager: PageManager,
        pad_token: int = 0,
    ) -> list[int]:
        B = len(sequences)
        device = self.embed_tokens.weight.device
        max_len = max(len(seq) for seq in sequences)
        batch = torch.full(
            size=(B, max_len),
            fill_value=pad_token,
            device=device,
            dtype=torch.long,
        )
        mask = torch.full(
            size=(B, max_len, max_len),
            fill_value=1,
            device=device,
            dtype=torch.bool,
        ).tril(0)
        seq_lens = torch.zeros(size=(len(sequences),), device=device, dtype=torch.long)
        for i, sequence in enumerate(sequences):
            seq_lens[i] = len(sequence)
            batch[i, : len(sequence)] = torch.tensor(
                sequence.prompt_tokens, device=device
            )
            mask[i, len(sequence) :, len(sequence) :] = 0

        logits: Tensor = self(
            batch,
            sequences=sequences,
            page_manager=page_manager,
            prefill=True,
            start_positions=None,
            mask=mask,
        )  # bsz, max_len, embed_dim

        temperatures = torch.tensor(
            [(seq.temperature) for seq in sequences], device=device
        )

        row_idx = torch.arange(B, device=device)
        last_idx = seq_lens - 1
        last_token_logits = logits[row_idx, last_idx, :]  # bsz, vocab
        next_token_probs = F.softmax(last_token_logits / temperatures[:, None], dim=-1)
        sampled_idx = torch.multinomial(next_token_probs, num_samples=1)  # bsz, 1
        return sampled_idx.squeeze(1).tolist()

    @torch.inference_mode()
    def decode(
        self, *, sequences: list[Sequence], page_manager: PageManager
    ) -> list[int]:
        batch = torch.tensor([[seq.last_token_id] for seq in sequences])
        batch = batch.to(device=self.embed_tokens.weight.device)
        start_positions = torch.tensor(
            [len(seq) - 1 for seq in sequences], device=self.embed_tokens.weight.device
        )
        logits: Tensor = self(
            batch,
            sequences=sequences,
            page_manager=page_manager,
            prefill=False,
            start_positions=start_positions,
        )
        next_tokens = logits.argmax(dim=-1)
        return (
            [next_tokens.squeeze().item()]
            if len(sequences) == 1
            else next_tokens.squeeze().tolist()
        )
