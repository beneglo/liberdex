# liberdex: the search page, the HTTP API and the MCP endpoint in one container.
#
#   docker run -p 8080:8080 -e OPENROUTER_API_KEY=... ghcr.io/beneglo/liberdex
#
# The ranking models are downloaded at build time, so the container answers
# /health seconds after it starts rather than after a 130 MB download.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    LIBERDEX_HOST=0.0.0.0 \
    HOME=/home/liberdex

RUN useradd --create-home --uid 1000 liberdex \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --chown=liberdex:liberdex pyproject.toml README.md LICENSE ./
COPY --chown=liberdex:liberdex liberdex ./liberdex
COPY --chown=liberdex:liberdex skills ./skills

RUN pip install --no-cache-dir . \
 && rm -rf /app/liberdex /app/skills

USER liberdex
# Models into the image, under the user that will run.
RUN liberdex warmup

# Plan cache, site-search atlas, sitemaps: worth keeping between runs.
VOLUME ["/home/liberdex/.cache/liberdex"]
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s \
  CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["liberdex", "serve", "--host", "0.0.0.0", "--port", "8080"]
