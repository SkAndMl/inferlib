FROM node:23-slim AS frontend-build

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY inferlib/ ./inferlib/
COPY --from=frontend-build /frontend/dist ./inferlib/server/static/
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["/app/.venv/bin/python", "-m", "inferlib.server.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
