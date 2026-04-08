FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=1
ENV MAX_CONCURRENT_ENVS=100

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md openenv.yaml requirements.txt /app/
COPY nl_reward_env /app/nl_reward_env
COPY server /app/server
COPY inference.py /app/inference.py
COPY tests /app/tests
COPY scripts /app/scripts

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host ${HOST} --port ${PORT} --workers ${WORKERS}"]
