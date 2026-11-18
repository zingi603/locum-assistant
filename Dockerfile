# Two stages: build the virtualenv with uv, then copy only what runs into a slim image.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
# dependencies first, so code changes don't invalidate this cached layer
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
# bake the embedding model into the image, so a cold start doesn't download 130 MB
ENV FASTEMBED_CACHE_PATH=/app/models
RUN .venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

FROM python:3.12-slim
WORKDIR /app
RUN useradd --create-home --uid 10001 app
COPY --from=build --chown=app /app /app
ENV PATH="/app/.venv/bin:$PATH" FASTEMBED_CACHE_PATH=/app/models PORT=8080 PYTHONUNBUFFERED=1
USER app
EXPOSE 8080
CMD ["locum-api"]
