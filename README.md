# inferlib

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SkAndMl/inferlib)

CPU-first local inference for LLMs — Python API and OpenAI-compatible server.

## Features

- **Paged KV cache** — fixed memory budget, no OOM surprises
- **Prefix caching** — shared prompt prefixes computed once, reused across requests
- **Batch scheduler** — interleaved prefill and continuous decode
- **OpenAI-compatible HTTP API** — drop-in for chat clients, with SSE streaming
- **Chat history** — persisted in SQLite, browsable in the built-in UI
- **Web UI** — React chat interface served alongside the API
- **CPU-first** — no CUDA required; runs anywhere Python and PyTorch run

## Architecture

```
              ┌──────────────┐   ┌──────────────────────┐
              │  Python API  │   │       Web UI          │
              │  LLM(model)  │   │   React / Vite / KaTeX│
              └──────┬───────┘   └──────────┬────────────┘
                     │                      │
              ┌──────▼──────────────────────▼────────────┐
              │              FastAPI Server               │
              │   /v1/chat/completions   /v1/chats/...    │
              │              SQLite (chats)               │
              └──────────────────────┬────────────────────┘
                                     │
              ┌──────────────────────▼────────────────────┐
              │              InferlibEngine                │
              │    async worker loop · per-request Queue   │
              └────────┬─────────────────────┬────────────┘
                       │                     │
             ┌─────────▼──────┐   ┌──────────▼─────────┐
             │   Scheduler    │   │       Runner        │
             │ prefill/decode │   │  prefill · decode   │
             └─────────┬──────┘   └──────────┬──────────┘
                       │                     │
             ┌─────────▼─────────────────────▼──────────┐
             │                PageManager                │
             │      paged KV pool · prefix cache         │
             └─────────────────────┬─────────────────────┘
                                   │
             ┌─────────────────────▼─────────────────────┐
             │               Qwen3 Model                  │
             │  GQA · RoPE · RMSNorm · SwiGLU             │
             │  online paged softmax (decode)             │
             └────────────────────────────────────────────┘
```

## Quickstart

### Docker

```bash
docker run -p 8000:8000 ghcr.io/skandml/inferlib:latest
```

Persist chat history across restarts:

```bash
docker run -p 8000:8000 -v ~/.inferlib:/root/.inferlib ghcr.io/skandml/inferlib:latest
```

### From source

Requires Python 3.13+, [`uv`](https://github.com/astral-sh/uv), and Node/npm for the UI.

```bash
git clone https://github.com/skandml/inferlib
cd inferlib
uv sync
inferlib build-frontend
inferlib serve
```

Open `http://localhost:8000` to use the chat UI.

## Usage

### Python API

```python
from inferlib import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-0.6B", max_model_len=4096)
outputs = llm.generate(
    ["Tell me something interesting."],
    SamplingParams(temperature=0.6, max_tokens=128),
)
print(outputs[0].text)
llm.close()
```

### HTTP API

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": false,
    "max_tokens": 128
  }'
```

Streaming uses SSE — set `"stream": true` and consume `data:` lines.

### Without the UI

```bash
inferlib serve --no-ui
```

## Benchmarks

Measured on CPU (Apple Silicon, macOS), batch of 4 sequences, ~128-token prompts, 64-token completions.

| Model | Throughput |
|-------|-----------|
| Qwen/Qwen3-0.6B | 12.23 tok/s |
| Qwen/Qwen3-1.7B | 5.85 tok/s |

## Learn more

- [Configuration](CONFIGURATION.md) — CLI flags and environment variables
- [API Reference](API.md) — HTTP endpoints and Python API details
- [Development](DEVELOPMENT.md) — contributing, tests, benchmarking

## License

MIT
