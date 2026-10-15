#!/usr/bin/env bash
# Starts the daily-update scheduler (background) then the app under gunicorn.
# ENABLE_DAILY_UPDATE=0 disables the scheduler entirely.
#
# WEB_WORKERS default is 1 on purpose: each gunicorn worker holds its own
# in-memory CSV cache, so POST /api/reload only resets the worker that
# serves it. Threads provide concurrency without breaking that. Raise
# WEB_WORKERS only if you accept stale workers after a daily refresh.
set -e

if [ "${ENABLE_DAILY_UPDATE:-1}" = "1" ]; then
    echo "[entrypoint] daily update scheduler enabled (UPDATE_TIME=${UPDATE_TIME:-17:05}, UPDATE_DAYS=${UPDATE_DAYS:-mon-fri})"
    python scheduler.py &
else
    echo "[entrypoint] daily update scheduler disabled (ENABLE_DAILY_UPDATE=${ENABLE_DAILY_UPDATE})"
fi

exec gunicorn \
    --bind "0.0.0.0:${PORT:-5000}" \
    --worker-class gthread \
    --workers "${WEB_WORKERS:-1}" \
    --threads "${WEB_THREADS:-8}" \
    --timeout "${WEB_TIMEOUT:-120}" \
    --access-logfile - \
    --error-logfile - \
    "app:app"
