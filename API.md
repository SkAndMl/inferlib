# API Reference

## Python API

### `LLM`

```python
from inferlib import LLM, SamplingParams

llm = LLM(
    model,                        # str — HuggingFace ID or local path
    max_model_len=None,           # int | None — max total tokens (prompt + completion)
    page_size=32,
    batch_size=4,
    max_active_sequences=8,
    memory_limit_bytes=4294967296,
)
```

`LLM` supports the context manager protocol:

```python
with LLM("Qwen/Qwen3-0.6B") as llm:
    outputs = llm.generate(["hello"], SamplingParams(max_tokens=64))
```

#### `llm.generate(prompts, sampling_params)`

```python
outputs: list[GenerationOutput] = llm.generate(
    prompts,          # list[str] | list[list[int]]
    sampling_params,  # SamplingParams | list[SamplingParams]
)
```

Prompts can be strings (tokenized internally) or raw token-ID lists:

```python
# String prompts
outputs = llm.generate(["hello", "world"], SamplingParams(max_tokens=64))

# Token-ID lists (one SamplingParams per prompt)
outputs = llm.generate(
    [[1, 2, 3, 4], [5, 6, 7]],
    [SamplingParams(max_tokens=32), SamplingParams(max_tokens=64)],
)
```

### `SamplingParams`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | `float` | `1.0` | Sampling temperature (applied to first token) |
| `max_tokens` | `int` | `16` | Max completion tokens |
| `ignore_eos` | `bool` | `False` | Do not stop at EOS token |
| `thinking` | `bool` | `False` | Enable Qwen3 thinking mode (chat template only) |

### `GenerationOutput`

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Decoded completion text |
| `token_ids` | `list[int]` | Completion token IDs |
| `finish_reason` | `"stop" \| "length"` | Why generation stopped |
| `prompt_tokens` | `int` | Number of prompt tokens |
| `completion_tokens` | `int` | Number of completion tokens |

---

## HTTP API

Base URL: `http://localhost:8000`

### Health

```
GET /health
```

```json
{"status": "ok", "model": "Qwen/Qwen3-0.6B"}
```

### Models

```
GET /v1/models
```

Returns the single model loaded in the current process.

```json
{
  "object": "list",
  "data": [{"id": "Qwen/Qwen3-0.6B", "object": "model", "owned_by": "inferlib"}]
}
```

### Chat completions

```
POST /v1/chat/completions
```

#### Request

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "hello"}
  ],
  "stream": false,
  "temperature": 0.6,
  "max_tokens": 128,
  "thinking": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `string` | required | Must match the loaded model |
| `messages` | `array` | required | At least one message |
| `stream` | `bool` | `true` | SSE streaming or wait for full response |
| `temperature` | `float` | `1.0` | Sampling temperature (≥ 0) |
| `max_tokens` | `int` | `4096` | Max completion tokens (> 0) |
| `thinking` | `bool` | `false` | Enable Qwen3 thinking mode |

#### Non-streaming response

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "Qwen/Qwen3-0.6B",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help?"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 20
  }
}
```

#### Streaming response (SSE)

Each line is a `data:` event with a chunk object, terminated by `data: [DONE]`:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Error envelope

All `4xx` and `5xx` responses use a consistent shape:

```json
{
  "error": {
    "type": "invalid_request_error",
    "code": "model_not_loaded",
    "message": "Requested model 'other' is not loaded; loaded model is 'Qwen/Qwen3-0.6B'."
  }
}
```

Common error codes:

| Code | Status | Meaning |
|------|--------|---------|
| `model_not_loaded` | 400 | Requested model is not the loaded one |
| `context_length_exceeded` | 400 | `prompt_tokens + max_tokens > max_model_len` |
| `invalid_prompt` | 400 | Empty prompt |
| `validation_error` | 400 | Malformed request body |
| `chat_not_found` | 404 | Chat ID does not exist |
| `internal_error` | 500 | Unhandled server exception |

---

## Chat persistence routes

These routes back the web UI and can also be called directly.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/chats` | List chats (query params: `limit=50`, `offset=0`) |
| `GET` | `/v1/chats/{chat_id}` | Get a single chat record |
| `GET` | `/v1/chats/{chat_id}/messages` | Get messages (query params: `limit`, `offset`) |
| `POST` | `/v1/chats/{chat_id}/messages` | Save a message to a chat |
| `PATCH` | `/v1/chats/{chat_id}` | Update chat title |
| `DELETE` | `/v1/chats/{chat_id}` | Delete a chat and its messages |

Chat records are created automatically when the first message is saved.
The chat title defaults to the first four words of the initial message content.
