# Multi-stage: dependencies resolve once in the builder, so an application-code
# change rebuilds only the thin final layer.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Must match the runtime WORKDIR. The install below is editable, so the path
# recorded in the venv has to be one that still exists in the final image —
# building in /build left the finder pointing at /build/src and every `sf` call
# died with ModuleNotFoundError. The layout is load-bearing beyond imports too:
# settings.py derives REPO_ROOT from the package location, and prompts/, configs/
# and evals/ are resolved relative to it.
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependency metadata only, so the resolve layer caches independently of source.
COPY pyproject.toml README.md ./
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-cache \
      httpx pydantic pydantic-settings duckdb numpy typer rich selectolax \
      fastapi 'uvicorn[standard]' tenacity python-dotenv pyyaml

COPY src ./src
RUN VIRTUAL_ENV=/opt/venv uv pip install --no-cache --no-deps -e .


FROM python:3.12-slim AS runtime

# Non-root: the container runs an agent that executes model-authored Python, so
# it should not have write access to anything it does not need.
RUN groupadd --gid 10001 sf && \
    useradd --uid 10001 --gid sf --create-home --shell /usr/sbin/nologin sf

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SF_DATA_DIR=/data \
    SF_DB_PATH=/data/signalforge.duckdb

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=sf:sf src ./src
COPY --chown=sf:sf prompts ./prompts
COPY --chown=sf:sf configs ./configs
COPY --chown=sf:sf evals ./evals
COPY --chown=sf:sf pyproject.toml README.md ./

# The warehouse is a volume: a DuckDB file baked into an image would be lost on
# every deploy and shared by nobody.
RUN mkdir -p /data && chown sf:sf /data
VOLUME ["/data"]

USER sf
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# Ollama runs on the host, not in here — a 5GB model in the image would make it
# undeployable, and the provider is reached over HTTP anyway. Point
# SF_OLLAMA_HOST at http://host.docker.internal:11434 when running locally.
CMD ["uvicorn", "signalforge.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
