"""
Refresh fundamentals in nifty_500_data.csv from yfinance.

Weekly CI job. Only existing columns are updated (names, order and units
are preserved); a ticker whose fetch fails keeps its previous values, so
the job never destroys data. Run:

    python refresh_fundamentals.py
"""

import os
import time

import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "nifty_500_data.csv")

# yfinance rate limits — keep the whole 500-ticker run well behaved
DELAY_PER_TICKER = 0.4
PAUSE_EVERY = 25
PAUSE_SECONDS = 3.0


def refresh() -> None:
    df = pd.read_csv(CSV_PATH, index_col=0)
    df.index.name = "Ticker"
    cols = list(df.columns)
    numeric_cols = {
        c for c in cols
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool
    }
    total = len(df)
    print(f"Refreshing fundamentals for {total} tickers, {len(cols)} columns...\n")

    updated = 0
    failed = []
    no_info = []

    for i, ticker in enumerate(df.index, 1):
        info = None
        for attempt in range(3):
            try:
                info = yf.Ticker(ticker).info or {}
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        if info is None:
            failed.append(ticker)
            print(f"[{i}/{total}] FAILED {ticker} (keeping previous values)")
        else:
            n = 0
            for c in cols:
                if c not in info:
                    continue
                val = info[c]
                if val is None or isinstance(val, (dict, list)):
                    continue
                if c in numeric_cols:
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        continue
                df.at[ticker, c] = val
                n += 1
            if n:
                updated += 1
            else:
                no_info.append(ticker)
            print(f"[{i}/{total}] {ticker}: {n} fields")

        if i % PAUSE_EVERY == 0:
            time.sleep(PAUSE_SECONDS)
        else:
            time.sleep(DELAY_PER_TICKER)

    # Keep numeric columns numeric after the assignments
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.to_csv(CSV_PATH)

    print(f"\n{'=' * 50}")
    print(f"Fundamentals refresh complete")
    print(f"  Updated: {updated}/{total}")
    if no_info:
        print(f"  No data returned ({len(no_info)}): {', '.join(no_info[:20])}")
    if failed:
        print(f"  Failed - kept previous values ({len(failed)}): {', '.join(failed[:20])}")
    print(f"  Written: {CSV_PATH}")


if __name__ == "__main__":
    refresh()
