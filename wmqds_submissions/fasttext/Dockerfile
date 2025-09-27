FROM debian:bookworm-slim
RUN apt-get update && \
    apt-get install -y git && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/application

COPY ./app ./app
COPY uv.lock .
COPY pyproject.toml .

# Install UV
COPY --from=ghcr.io/astral-sh/uv:0.6.9 /uv /uvx /bin/
ENV UV_CACHE_DIR=/tmp/.uv_cache
ENV UV_TOOL_DIR=/tmp/.uv_tool
ENV UV_PYTHON_INSTALL_DIR=/opt/application/.uv_python
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

RUN uv python install 3.11
ENV UV_PYTHON_DOWNLOADS=0

RUN --mount=type=cache,target="${UV_CACHE_DIR}" uv sync
ENV PATH="/opt/application/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]