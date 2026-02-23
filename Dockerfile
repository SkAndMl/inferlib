FROM python:3.13-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY inferlib/ ./inferlib/
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["/app/.venv/bin/python", "-m", "inferlib.server.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
