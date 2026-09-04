FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# uv + --locked installs exactly what CI tests against (uv.lock), not whatever
# pip's resolver picks fresh off PyPI today — see KEY_DECISIONS.md Decision 20.
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock alembic.ini ./
COPY alembic/ ./alembic/
COPY src/ ./src/
RUN uv sync --locked --no-editable

ENV PATH="/app/.venv/bin:${PATH}"

RUN useradd --create-home --shell /usr/sbin/nologin mlms && chown -R mlms:mlms /app
USER mlms

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD python -c "import os, socket; socket.create_connection(('127.0.0.1', int(os.environ.get('MCP_PORT', 8000))), timeout=2)"

CMD ["sh", "-c", "alembic upgrade head && python -m mlms.server"]
