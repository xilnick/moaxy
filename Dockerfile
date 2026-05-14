# Stage 1 — Builder
FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Stage 2 — Runtime
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --system moaxy && useradd --system --gid moaxy --no-create-home moaxy
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN mkdir -p /app/config && chown -R moaxy:moaxy /app

USER moaxy
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["python", "-c", "import moaxy; print(f'moaxy v{moaxy.__version__} ready')"]
