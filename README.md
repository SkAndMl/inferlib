# inferlib

A CPU-first serving engine for running multiple open-weights models locally.

## What is this?

inferlib is an LLM inference engine built from scratch with a focus on running open-weights models on CPU hardware. It implements a paged KV cache, a batch scheduler, and an OpenAI-compatible HTTP API — so you can point existing OpenAI clients at it with minimal changes.

## Quick Start (Docker)

The easiest way to run inferlib. No Python setup required.

```bash
docker run -p 8000:8000 skandml/inferlib:0.2.3
```

To use a different model:

```bash
docker run -p 8000:8000 \
  -e INFERLIB_MODEL_CLASS=Qwen/Qwen3-1.7B \
  skandml/inferlib:0.2.3
```

The `-v` flag persists your chat history between runs. Without it, the database resets every time the container stops. To persist data:

```bash
docker run -p 8000:8000 -v ~/.inferlib:/root/.inferlib skandml/inferlib:0.2.3
```

On first start, inferlib downloads the model weights from HuggingFace. This takes a few minutes depending on your connection. Subsequent starts are instant.

Open `http://localhost:8000` in your browser.

## Quick Start (from source)

Requires Python 3.13+, [uv](https://github.com/astral-sh/uv), and `npm`.

```bash
git clone https://github.com/skandml/inferlib
cd inferlib
uv sync
inferlib serve
```

`inferlib serve` automatically builds the frontend bundle on first run if needed, then serves the UI at `http://localhost:8000`.

With a specific model:

```bash
inferlib serve --model-class Qwen/Qwen3-1.7B
```

## Frontend Development

The production UI is served by FastAPI from a compiled React/Vite bundle.

For hot-reload frontend work (optional):

```bash
# terminal 1
inferlib serve

# terminal 2
cd frontend
npm install
npm run dev
```

Vite serves the React app on `http://localhost:5173` and proxies `/v1/*` and `/health` to the FastAPI server on `http://localhost:8000`.

## Usage

### CLI

```
inferlib serve [OPTIONS]

Options:
  --host TEXT         Host to bind to (default: 0.0.0.0)
  --port INT          Port to listen on (default: 8000)
  --model-class TEXT  Model to load (default: Qwen/Qwen3-0.6B)
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | Generate a response. Supports streaming via SSE. |
| `GET` | `/v1/models` | List loaded models. |
| `GET` | `/v1/chats` | List saved chat sessions. |
| `GET` | `/v1/chats/{chat_id}/messages` | Get messages for a chat. |
| `POST` | `/v1/chats/{chat_id}/messages` | Save a message to a chat. |
| `PATCH` | `/v1/chats/{chat_id}` | Update chat title. |
| `DELETE` | `/v1/chats/{chat_id}` | Delete a chat. |
| `GET` | `/health` | Health check. |

### Chat Completions

inferlib implements the OpenAI chat completions API. You can use `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": false
  }'
```

Or with streaming:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": true
  }'
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Model identifier |
| `messages` | array | required | Conversation history |
| `stream` | bool | `true` | Stream tokens via SSE |
| `temperature` | float | `1.0` | Sampling temperature |
| `max_tokens` | int | `4096` | Maximum tokens to generate |
| `thinking` | bool | `false` | Enable Qwen3 thinking mode |

## Supported Models

| Model | Size |
|-------|------|
| `Qwen/Qwen3-0.6B` | 0.6B parameters |
| `Qwen/Qwen3-1.7B` | 1.7B parameters |

## Architecture

inferlib is built in layers:

```
inferlib/
├── core/
│   ├── engine/
│   │   ├── engine.py      # worker loop, request lifecycle
│   │   ├── scheduler.py   # batch scheduling, prefill/decode priority
│   │   ├── runner.py      # prefill and decode execution
│   │   ├── page.py        # paged KV cache
│   │   └── sequence.py    # sequence state machine
│   └── models/
│       └── qwen3.py       # Qwen3 model implementation
└── server/
    ├── apis/
    │   ├── chat.py        # /v1/chat/completions
    │   └── ui_chats.py    # chat persistence endpoints
    ├── app.py             # FastAPI app
    ├── cli.py             # inferlib serve entrypoint
    ├── db_client.py       # SQLite via aiosqlite
    ├── models.py          # Pydantic request/response schemas
    └── static/            # compiled React frontend bundle
```

```
frontend/
├── src/
│   ├── components/        # chat UI building blocks
│   ├── App.tsx            # app state and orchestration
│   ├── api.ts             # browser API client
│   └── markdown.ts        # markdown, math, and thinking parsing
└── vite.config.ts         # frontend dev/build config
```

**Paged KV Cache** — attention key/value tensors are stored in fixed-size pages rather than contiguous buffers. This avoids memory fragmentation and allows multiple sequences to share the memory pool efficiently.

**Scheduler** — sequences are grouped into buckets by page count. The scheduler prioritises prefill (to minimise time-to-first-token) while interleaving decode steps to keep active sequences progressing. Bucket selection uses a skip-count mechanism to prevent starvation of minority buckets.

**Async engine** — the worker loop runs as an `asyncio` task. An `asyncio.Event` is used for wakeup signalling so the engine sleeps with zero CPU usage when idle, and wakes immediately when a new request arrives.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERLIB_MODEL_CLASS` | `Qwen/Qwen3-0.6B` | Model to load at startup |
| `DB_PATH` | `~/.inferlib/chats.db` | Path to SQLite database |

## Roadmap

**v1.0.0**
- Continuous batching
- Prefix caching
- ChatGPT-like UI improvements

**v1.5.0**
- Tool calling
- Web search

**v2.0.0**
- Additional model support

**v3.0.0**
- Rust rewrite of the inference path

## License

MIT
