# syntax=docker/dockerfile:1
FROM python:3.12-slim

# uv binary, copied from the official image instead of pip-installing it
COPY --from=ghcr.io/astral-sh/uv:0.11.13 /uv /uvx /bin/

# Hugging Face Spaces runs containers as a non-root user with UID 1000. Matching that
# here means file ownership is the same locally and on the Space, so anything the app
# writes at runtime (the SQLite checkpointer) works in both places.
RUN useradd -m -u 1000 user
USER user
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Dependencies first, in their own layer: they change rarely, so editing source code
# does not force a re-install on every rebuild.
COPY --chown=user:user pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# Source code last, because it changes on every commit.
COPY --chown=user:user src/ ./src/
COPY --chown=user:user scripts/ ./scripts/
COPY --chown=user:user data/policies/ ./data/policies/
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Bind 0.0.0.0 (not 127.0.0.1) or the port mapping cannot reach the process from
# outside the container. The port itself has to be read from $PORT at runtime because
# hosts disagree: Render injects PORT=10000, HF Spaces expects 7860. Shell form is
# required here — exec form does not expand variables.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
