"""
Import financial statements from yfinance for every NIFTY 500 ticker.

Six datasets, one CSV per ticker per metric:

    data/balance_sheet/{TICKER}.csv            annual balance sheet
    data/income_statement/{TICKER}.csv         annual income statement
    data/cashflow/{TICKER}.csv                 annual cash flow
    data/quarterly_results/balance_sheet/{TICKER}.csv
    data/quarterly_results/income_statement/{TICKER}.csv
    data/quarterly_results/cashflow/{TICKER}.csv

A file is only written when it is missing/empty OR the fetched data carries a
newer reporting period than the stored one (i.e. a new result was declared).
Restatements of existing periods are ignored. A failed fetch keeps the
previous file, so the job never destroys data.

Run:

    python refresh_financials.py
    python refresh_financials.py --tickers ACC.NS,SBIN.NS
    python refresh_financials.py --force
"""

import argparse
import os
import time

import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "nifty_500_data.csv")
DATA_DIR = os.path.join(BASE_DIR, "data")

# metric folder (relative to data/) -> yfinance Ticker attribute
METRICS = {
    "balance_sheet": "balance_sheet",
    "income_statement": "income_stmt",
    "cashflow": "cashflow",
    "quarterly_results/balance_sheet": "quarterly_balance_sheet",
    "quarterly_results/income_statement": "quarterly_income_stmt",
    "quarterly_results/cashflow": "quarterly_cashflow",
}

# yfinance rate limits — keep the whole 500-ticker / 6-statement run well behaved
DELAY_PER_STATEMENT = 0.2
DELAY_PER_TICKER = 0.4
PAUSE_EVERY = 25
PAUSE_SECONDS = 3.0
RETRIES = 3


def get_ticker_list():
    df = pd.read_csv(CSV_PATH, index_col=0)
    return sorted(df.index.tolist())


def fetch_statement(ticker_obj, attr):
    """Fetch one statement with retries. Returns a normalised DataFrame
    (columns = YYYY-MM-DD strings, newest first) or None if unavailable."""
    for attempt in range(RETRIES):
        try:
            df = getattr(ticker_obj, attr)
            break
        except Exception:
            df = None
            time.sleep(1.5 * (attempt + 1))
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    df = df.copy()
    df.columns = pd.to_datetime(df.columns, errors="coerce")
    df = df.drop(columns=[c for c in df.columns if pd.isna(c)])
    if df.empty:
        return None
    df = df[sorted(df.columns, reverse=True)]
    df.columns = [c.strftime("%Y-%m-%d") for c in df.columns]
    return df


def stored_max_period(path):
    """Latest reporting period stored in an existing file, or None."""
    try:
        old = pd.read_csv(path, index_col=0, nrows=1)
        periods = pd.to_datetime(old.columns, errors="coerce").dropna()
        return periods.max() if len(periods) else None
    except Exception:
        return None


def refresh(tickers, force=False):
    total = len(tickers)
    print(f"Importing financial statements for {total} tickers, "
          f"{len(METRICS)} metrics each...\n")

    created = updated = same = 0
    failed = []

    for i, ticker in enumerate(tickers, 1):
        status = {"created": 0, "updated": 0, "same": 0}
        ticker_failed = []
        try:
            ticker_obj = yf.Ticker(ticker)
        except Exception:
            ticker_obj = None

        if ticker_obj is None:
            failed.append(f"{ticker} (all)")
            print(f"[{i}/{total}] FAILED {ticker} (all metrics, keeping previous files)")
        else:
            for folder, attr in METRICS.items():
                path = os.path.join(DATA_DIR, folder, f"{ticker}.csv")
                df = fetch_statement(ticker_obj, attr)
                time.sleep(DELAY_PER_STATEMENT)

                if df is None:
                    ticker_failed.append(attr)
                    continue

                if os.path.exists(path) and os.path.getsize(path) > 0 and not force:
                    old_max = stored_max_period(path)
                    new_max = pd.to_datetime(df.columns).max()
                    if old_max is not None and new_max is not None and new_max <= old_max:
                        status["same"] += 1
                        continue

                os.makedirs(os.path.dirname(path), exist_ok=True)
                existed = os.path.exists(path) and os.path.getsize(path) > 0
                df.to_csv(path)
                if existed:
                    status["updated"] += 1
                    updated += 1
                else:
                    status["created"] += 1
                    created += 1

            same += status["same"]
            if ticker_failed:
                failed.append(f"{ticker} ({', '.join(ticker_failed)})")
            parts = [f"{v} {k}" for k, v in status.items() if v]
            note = f" | no data: {', '.join(ticker_failed)}" if ticker_failed else ""
            print(f"[{i}/{total}] {ticker}: {', '.join(parts) or 'nothing new'}{note}")

        if i % PAUSE_EVERY == 0:
            time.sleep(PAUSE_SECONDS)
        else:
            time.sleep(DELAY_PER_TICKER)

    print(f"\n{'=' * 50}")
    print(f"Financial statements import complete")
    print(f"  Created (file missing): {created}")
    print(f"  Updated (new period):   {updated}")
    print(f"  Up-to-date (skipped):   {same}")
    if failed:
        kind = "no data / fetch failed"
        print(f"  {kind} ({len(failed)}): {', '.join(failed[:20])}")
    print(f"  Written under: {DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", help="comma-separated ticker list (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="rewrite files even when no new period is present")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = get_ticker_list()
    refresh(tickers, force=args.force)


if __name__ == "__main__":
    main()
