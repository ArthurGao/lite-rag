# lite-rag HTTP service (design §7). Build: docker build -t lite-rag .
# Needs at run time: LITERAG_DATABASE_URL, LITERAG_SERVER__KEYS (a JSON list from `literag key
# create`), and an embedding service (default: Ollama, LITERAG_EMBEDDING__BASE_URL).
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.10.8 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra server --extra pdf --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra server --extra pdf

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 literag
COPY --from=build /app /app
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER literag
WORKDIR /home/literag
EXPOSE 8700
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8700/health', timeout=3)"]
# Without keys `serve` refuses a non-loopback address, so this container needs keys to start.
CMD ["sh", "-c", "literag migrate && exec literag serve --host 0.0.0.0"]
