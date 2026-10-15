"""
In-container daily data refresh.

Runs the full pipeline after market close, in the same order as the CI
workflow (.github/workflows/daily-data.yml):

    refresh_fundamentals.py -> download_data.py -> compute_indicators.py
    -> build_snapshot.py

then POSTs to /api/reload so the running Flask app drops its CSV caches and
serves the fresh data without a restart.

Environment:
    UPDATE_TIME          "HH:MM" 24h in TZ (default 17:05)
    UPDATE_DAYS          "mon-fri" (default) or "daily"
    RUN_ON_BOOT          "1" runs the pipeline immediately on container start
    RELOAD_URL           default http://127.0.0.1:5000/api/reload
    RELOAD_TOKEN         optional; sent as X-Reload-Token when set
    TZ                   default Asia/Kolkata
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Same order (and scripts) as the daily CI workflow.
PIPELINE_STEPS = (
    "refresh_fundamentals.py",
    "download_data.py",
    "compute_indicators.py",
    "build_snapshot.py",
)


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def log(msg: str) -> None:
    tz = _env("TZ", "Asia/Kolkata")
    now = datetime.now(ZoneInfo(tz)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[scheduler {now}] {msg}", flush=True)


def next_run(now: datetime, time_str: str, days: str) -> datetime:
    """Next occurrence of HH:MM strictly after `now`, skipping weekends when
    days == "mon-fri"."""
    hour, minute = (int(p) for p in time_str.split(":"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    if days == "mon-fri":
        while target.weekday() >= 5:  # 5=Sat, 6=Sun
            target += timedelta(days=1)
    return target


def run_pipeline() -> bool:
    """Run pipeline steps in order; fail fast (mirrors CI step behaviour)."""
    for step in PIPELINE_STEPS:
        started = time.monotonic()
        log(f"step start: {step}")
        try:
            result = subprocess.run(
                [sys.executable, os.path.join(BASE_DIR, step)],
                cwd=BASE_DIR,
            )
        except Exception as exc:  # never let the scheduler die
            log(f"step ERROR: {step}: {exc}")
            return False
        if result.returncode != 0:
            log(f"step FAILED: {step} (exit {result.returncode}) - aborting pipeline")
            return False
        log(f"step done: {step} ({time.monotonic() - started:.1f}s)")
    return True


def notify_reload() -> None:
    """Tell the Flask app to drop its CSV caches so fresh data is served."""
    url = _env("RELOAD_URL", "http://127.0.0.1:5000/api/reload")
    token = os.environ.get("RELOAD_TOKEN", "")
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps({}).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            if token:
                request.add_header("X-Reload-Token", token)
            with urllib.request.urlopen(request, timeout=10) as response:
                log(f"cache reload notified (HTTP {response.status})")
                return
        except (urllib.error.URLError, OSError) as exc:
            log(f"cache reload attempt {attempt}/3 failed: {exc}")
            time.sleep(3)
    log("cache reload failed - app keeps old data until next restart")


def sleep_until(target: datetime) -> None:
    """Sleep in 60s slices so log lines show progress and shutdown stays fast."""
    while True:
        remaining = (target - datetime.now(target.tzinfo)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 60))


def main() -> None:
    tz_name = _env("TZ", "Asia/Kolkata")
    tz = ZoneInfo(tz_name)
    update_time = _env("UPDATE_TIME", "17:05")
    days = _env("UPDATE_DAYS", "mon-fri").lower()
    if days not in ("mon-fri", "daily"):
        days = "mon-fri"

    log(f"started (UPDATE_TIME={update_time}, UPDATE_DAYS={days}, TZ={tz_name})")

    if _env("RUN_ON_BOOT", "0") == "1":
        log("RUN_ON_BOOT=1 - running pipeline immediately")
        if run_pipeline():
            notify_reload()

    while True:
        now = datetime.now(tz)
        target = next_run(now, update_time, days)
        wait_min = (target - now).total_seconds() / 60
        log(
            f"next update: {target.strftime('%a %Y-%m-%d %H:%M %Z')} "
            f"(in {wait_min:.0f} min)"
        )
        sleep_until(target)

        log("update window reached - starting daily pipeline")
        if run_pipeline():
            notify_reload()
            log("daily update complete")
        else:
            log("daily update did NOT complete - previous data kept")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("stopped")
