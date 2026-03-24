# inferlib

CPU-first local inference for Qwen3 models, with a small Python API and an OpenAI-compatible FastAPI server.

## What inferlib is

inferlib is a local serving engine built around:

- a paged KV cache
- a batch scheduler for prefill and decode work
- a single-process FastAPI server with chat persistence
- a small Python API for direct generation and benchmarking

Current production scope is intentionally narrow:

- Qwen3 only
- one loaded model per process
- CPU-first execution
- no distributed serving

## Install

### From source

Requires Python 3.13+, `uv`, Node/npm for the optional UI build, and Docker only if you want container packaging.

```bash
git clone https://github.com/skandml/inferlib
cd inferlib
uv sync --group dev
inferlib build-frontend
inferlib serve
```

If you only want the HTTP API and not the web UI:

```bash
inferlib serve --no-ui
```

### Docker

```bash
docker build -t inferlib .
docker run -p 8000:8000 inferlib
```

To persist chat history:

```bash
docker run -p 8000:8000 -v ~/.inferlib:/root/.inferlib inferlib
```

## Python API

```python
from inferlib import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-0.6B", max_model_len=4096)
outputs = llm.generate(
    ["Benchmark: hello"],
    SamplingParams(temperature=0.6, max_tokens=64),
)
print(outputs[0].text)
llm.close()
```

Raw token IDs are also supported:

```python
from inferlib import LLM, SamplingParams

llm = LLM("~/huggingface/Qwen3-0.6B/", max_model_len=4096)
outputs = llm.generate(
    [[1, 2, 3, 4]],
    [SamplingParams(ignore_eos=True, max_tokens=128)],
)
llm.close()
```

Public Python surface:

- `LLM`
- `SamplingParams`
- `GenerationOutput`

## CLI

### `inferlib serve`

```bash
inferlib serve \
  --model-class Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --db-path ~/.inferlib/chats.db \
  --page-size 32 \
  --batch-size 4 \
  --max-active-sequences 8 \
  --memory-limit-bytes 4294967296 \
  --log-level INFO \
  --log-format text
```

Available flags:

- `--model-class`
- `--host`
- `--port`
- `--db-path`
- `--page-size`
- `--batch-size`
- `--max-active-sequences`
- `--memory-limit-bytes`
- `--log-level`
- `--log-format text|json`
- `--no-ui`

### `inferlib build-frontend`

Builds the React/Vite bundle explicitly. `serve` no longer installs dependencies or builds the frontend on demand.

```bash
inferlib build-frontend
```

## HTTP API

### Health and models

- `GET /health`
- `GET /v1/models`

`/health` returns:

```json
{"status":"ok","model":"Qwen/Qwen3-0.6B"}
```

`/v1/models` returns only the model loaded in the current process.

### Chat completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": false,
    "max_tokens": 128,
    "temperature": 0.6
  }'
```

Streaming uses SSE with OpenAI-style chunk objects and a final `data: [DONE]`.

### Error envelope

All 4xx and 5xx responses use:

```json
{
  "error": {
    "type": "invalid_request_error",
    "code": "model_not_loaded",
    "message": "Requested model ... is not loaded ..."
  }
}
```

### Chat persistence routes

- `GET /v1/chats`
- `GET /v1/chats/{chat_id}`
- `GET /v1/chats/{chat_id}/messages`
- `POST /v1/chats/{chat_id}/messages`
- `PATCH /v1/chats/{chat_id}`
- `DELETE /v1/chats/{chat_id}`

## Configuration

CLI flags override environment variables.

| Variable | Default | Notes |
| --- | --- | --- |
| `INFERLIB_MODEL_CLASS` | `Qwen/Qwen3-0.6B` | Loaded model id or local path |
| `INFERLIB_HOST` | `0.0.0.0` | Server bind host |
| `INFERLIB_PORT` | `8000` | Server port |
| `INFERLIB_DB_PATH` | `~/.inferlib/chats.db` | SQLite path |
| `INFERLIB_PAGE_SIZE` | `32` | KV page size |
| `INFERLIB_BATCH_SIZE` | `4` | Scheduler batch size |
| `INFERLIB_MAX_ACTIVE_SEQUENCES` | `8` | Active sequence limit |
| `INFERLIB_MEMORY_LIMIT_BYTES` | `4294967296` | Engine memory budget |
| `INFERLIB_LOG_LEVEL` | `INFO` | Python log level |
| `INFERLIB_LOG_FORMAT` | `text` | `text` or `json` |
| `INFERLIB_UI` | `true` | Set false to disable the UI |

Legacy compatibility:

- `DB_PATH` is still accepted as a fallback for one release.

## Logging

inferlib now uses centralized structured logging for both inferlib and uvicorn logs.

Log records include:

- startup config
- request id
- prompt and completion token counts
- duration and TTFT
- tokens per second
- scheduler batch mode and batch size
- sequence completion and page pressure context

Use JSON logs with:

```bash
inferlib serve --log-format json
```

## Benchmark

`bench.py` benchmarks inferlib and optionally vLLM with synthetic token IDs, close to the nano-vllm workflow.

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

If `vllm` is not installed, the script prints a skip message and still runs the inferlib benchmark.

## Development

Checks used by CI:

```bash
uv run ruff check .
uv run pytest -q
cd frontend && npm ci && npm run build
docker build .
```

## Current limitations

- Qwen3 is the only supported model family.
- Only one model is loaded per process.
- The Python API exposes a small, benchmark-oriented subset of vLLM-style ergonomics.
- The benchmark is engine-level only; it does not measure end-to-end HTTP latency.

## License

MIT
