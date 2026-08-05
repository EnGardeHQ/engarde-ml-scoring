# engarde-ml-scoring
# Rules (SKILLS.md §3.2 / docs/rules/02-backend.md): no EXPOSE, no HEALTHCHECK,
# CMD must consume the Railway-injected PORT and fail fast if it's absent in
# production. Never hardcode ports or connection strings.

FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 svcuser && chown -R svcuser:svcuser /app
USER svcuser

CMD ["/bin/sh", "-c", "if [ -z \"$PORT\" ] && [ \"$ENV\" = \"production\" ]; then echo 'FATAL: PORT not injected by Railway' >&2; exit 1; fi; uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
