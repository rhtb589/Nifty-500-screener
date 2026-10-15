# NIFTY 500 Screener — web app + in-container daily data refresh.
# Python matches the CI workflow (3.12). Data (stock_data/, *.csv) is baked
# in so the image works standalone; docker-compose bind-mounts the same paths
# so daily pipeline runs persist to the host.

FROM python:3.12-slim

ENV TZ=Asia/Kolkata \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_HOST=0.0.0.0 \
    FLASK_DEBUG=0 \
    PORT=5000 \
    ENABLE_DAILY_UPDATE=1 \
    UPDATE_TIME=17:05 \
    UPDATE_DAYS=mon-fri

# tzdata for TZ=Asia/Kolkata, curl for the container healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

# Application code + data
COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 5000

HEALTHCHECK --interval=60s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/tickers" >/dev/null || exit 1

ENTRYPOINT ["./entrypoint.sh"]
