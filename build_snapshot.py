"""
Build latest_snapshot.csv — the scanning source for the screener.

Merges fundamentals (nifty_500_data.csv) with the latest row of every
stock_data/*.csv (technical indicators + As_of date) and derived columns.

Run manually:
    python build_snapshot.py

Also run automatically by CI after data refreshes, and by the app itself
when the snapshot file is missing.
"""

import os

from tools.base import SNAPSHOT_PATH, build_snapshot


def main():
    df = build_snapshot()
    size_mb = os.path.getsize(SNAPSHOT_PATH) / (1024 * 1024)
    as_of_vals = df["As_of"].astype(str) if "As_of" in df.columns else []
    as_of = sorted({a for a in as_of_vals if a})
    print(f"Snapshot written: {SNAPSHOT_PATH}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  As_of range: {as_of[0] if as_of else 'n/a'} .. {as_of[-1] if as_of else 'n/a'}")


if __name__ == "__main__":
    main()
