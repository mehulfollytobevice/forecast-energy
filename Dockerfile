FROM node:22-bookworm-slim AS web
WORKDIR /repo
COPY app/web/package.json app/web/package-lock.json ./app/web/
RUN cd app/web && npm ci
COPY app/web ./app/web
RUN cd app/web && npm run build

FROM python:3.12-slim AS runtime
WORKDIR /repo
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY app ./app
COPY configs ./configs
RUN uv sync --frozen --no-dev --extra dashboard
COPY --from=web /repo/app/web/dist ./app/web/dist
ENV FORECAST_SNAPSHOT=/data/forecast_snapshot.json
ENV PYTHONPATH=/repo/src
EXPOSE 8000
CMD ["sh", "-c", "uv run uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
