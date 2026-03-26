# Development

## Prerequisites

- Python 3.13+
- [`uv`](https://github.com/astral-sh/uv)
- Node / npm (for the frontend)
- Docker (optional, for container builds)

## Install

```bash
git clone https://github.com/skandml/inferlib
cd inferlib
uv sync --group dev
```

## Build the frontend

The frontend must be built explicitly — `inferlib serve` no longer builds it on startup.

```bash
inferlib build-frontend
```

This runs `npm ci && npm run build` inside `frontend/` and produces `frontend/dist/`.

## CI checks

These match what the GitHub Actions CI pipeline runs:

```bash
uv run ruff check .           # lint
uv run pytest -q              # tests
cd frontend && npm ci && npm run build   # frontend build
docker build .                # container smoke test
```

## Benchmark

`bench.py` measures engine throughput with synthetic token-ID workloads and optionally compares against vLLM.

```bash
uv run python bench.py \
  --backend both \
  --model ~/huggingface/Qwen3-0.6B/ \
  --num-seqs 256 \
  --max-input-len 1024 \
  --max-output-len 1024 \
  --seed 0 \
  --max-model-len 4096
```

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `both` | `inferlib`, `vllm`, or `both` |
| `--model` | `~/huggingface/Qwen3-0.6B/` | Model path |
| `--num-seqs` | `256` | Number of sequences |
| `--max-input-len` | `1024` | Max prompt length (tokens) |
| `--max-output-len` | `1024` | Max completion length (tokens) |
| `--seed` | `0` | RNG seed for reproducibility |
| `--max-model-len` | `4096` | KV cache context window |

If `vllm` is not installed, the script skips it and runs the inferlib benchmark only.

## Current limitations

- Qwen3 is the only supported model family.
- One model loaded per process — no hot-swapping.
- CPU execution only — no GPU/CUDA path.
- No authentication on the HTTP API — intended for local or private deployments.
- Benchmark measures engine throughput only, not end-to-end HTTP latency.
