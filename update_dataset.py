"""
Update nifty_500_data.csv with the latest indicator values from stock_data/.
Takes the first row (most recent) from each enriched CSV and merges it.
"""

import os
import glob
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "nifty_500_data.csv")
STOCK_DATA_DIR = os.path.join(BASE_DIR, "stock_data")


def main():
    # Load original dataset
    print("Loading original nifty_500_data.csv...")
    original_df = pd.read_csv(CSV_PATH, index_col=0)
    original_cols = set(original_df.columns)
    print(f"Original dataset: {len(original_df)} stocks, {len(original_df.columns)} columns\n")

    # Read all enriched CSVs
    csv_files = sorted(glob.glob(os.path.join(STOCK_DATA_DIR, "*.csv")))
    total = len(csv_files)

    if total == 0:
        print("No CSV files found in stock_data/. Run compute_indicators.py first.")
        return

    print(f"Reading latest values from {total} stock files...\n")

    indicator_rows = []
    success = 0
    failed = []

    for i, filepath in enumerate(csv_files, 1):
        ticker = os.path.basename(filepath).replace(".csv", "")
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            if len(df) > 0:
                # Take the first row (most recent)
                latest = df.iloc[0].to_dict()
                latest["Ticker"] = ticker
                indicator_rows.append(latest)
                success += 1
                if i % 100 == 0:
                    print(f"  Read {i}/{total} files...")
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(ticker)
            print(f"  Error reading {ticker}: {e}")

    print(f"Successfully read {success}/{total} files\n")

    # Create indicators DataFrame
    if not indicator_rows:
        print("No indicator data to merge.")
        return

    indicators_df = pd.DataFrame(indicator_rows)
    indicators_df = indicators_df.set_index("Ticker")

    # Identify new columns (not in original)
    new_cols = [c for c in indicators_df.columns if c not in original_cols]
    print(f"New indicator columns: {len(new_cols)}")

    # Merge: only add new columns
    print("Merging with original dataset...")
    for col in new_cols:
        original_df[col] = indicators_df[col]

    # Save updated dataset
    print(f"Saving updated dataset...")
    original_df.to_csv(CSV_PATH)

    print(f"\n{'='*50}")
    print(f"Update complete!")
    print(f"  Stocks: {len(original_df)}")
    print(f"  Total columns: {len(original_df.columns)} (was {len(original_cols)}, +{len(original_df.columns) - len(original_cols)} new)")
    if failed:
        print(f"  Failed ({len(failed)}): {', '.join(failed[:20])}")


if __name__ == "__main__":
    main()
