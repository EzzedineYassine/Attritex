FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY churnsense ./churnsense

RUN uv pip install --system -e .

EXPOSE 8000

CMD ["churnsense", "serve", "--host", "0.0.0.0", "--port", "8000"]
