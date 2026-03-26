# Configuration

CLI flags take precedence over environment variables, which take precedence over defaults.

## `inferlib serve` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-class` | `Qwen/Qwen3-0.6B` | Model ID (HuggingFace) or local path |
| `--host` | `0.0.0.0` | Bind host |
| `--port` | `8000` | Bind port |
| `--db-path` | `~/.inferlib/chats.db` | SQLite file path |
| `--page-size` | `32` | KV cache page size (tokens per page) |
| `--batch-size` | `4` | Max sequences per scheduler batch |
| `--max-active-sequences` | `8` | Max concurrently active sequences |
| `--memory-limit-bytes` | `4294967296` | KV cache memory budget (bytes) |
| `--log-level` | `INFO` | Python log level |
| `--log-format` | `text` | `text` or `json` |
| `--no-ui` | — | Disable the web UI |

## Environment variables

| Variable | Default | Notes |
|----------|---------|-------|
| `INFERLIB_MODEL_CLASS` | `Qwen/Qwen3-0.6B` | |
| `INFERLIB_HOST` | `0.0.0.0` | |
| `INFERLIB_PORT` | `8000` | |
| `INFERLIB_DB_PATH` | `~/.inferlib/chats.db` | |
| `INFERLIB_PAGE_SIZE` | `32` | |
| `INFERLIB_BATCH_SIZE` | `4` | |
| `INFERLIB_MAX_ACTIVE_SEQUENCES` | `8` | |
| `INFERLIB_MEMORY_LIMIT_BYTES` | `4294967296` | |
| `INFERLIB_LOG_LEVEL` | `INFO` | |
| `INFERLIB_LOG_FORMAT` | `text` | `text` or `json` |
| `INFERLIB_UI` | `true` | Set to `false` to disable the web UI |

### Legacy

`DB_PATH` is still accepted as a fallback for `INFERLIB_DB_PATH` for one release.

## Structured logging

Enable JSON logs for log-aggregation pipelines:

```bash
inferlib serve --log-format json
```

Log records include: startup config, request ID, prompt/completion token counts, duration, TTFT, tokens per second, scheduler batch mode and size, sequence finish reason.
