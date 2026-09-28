"""
Download 1 year of daily OHLCV data for all NIFTY 500 stocks.
Saves each stock as a separate CSV in stock_data/ folder.
"""

import os
import time
import pandas as pd
import yfinance as yf

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "nifty_500_data.csv")
STOCK_DATA_DIR = os.path.join(BASE_DIR, "stock_data")


def get_ticker_list():
    """Read tickers from the main dataset."""
    df = pd.read_csv(CSV_PATH, index_col=0)
    return sorted(df.index.tolist())


def download_stock(ticker, period="1y", interval="1d"):
    """Download historical data for a single stock."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    if hist.empty:
        return None
    hist.index.name = "Date"
    # income_statement = stock.income_stmt        # Annual Income Statement
    # balance_sheet = stock.balance_sheet         # Annual Balance Sheet
    # cash_flow = stock.cashflow                  # Annual Cash Flow Statement
    return hist


def main():
    # Create stock_data directory
    os.makedirs(STOCK_DATA_DIR, exist_ok=True)

    tickers = get_ticker_list()
    total = len(tickers)
    success = 0
    failed = []

    print(f"Downloading 1 year data for {total} stocks...")
    print(f"Output directory: {STOCK_DATA_DIR}\n")

    for i, ticker in enumerate(tickers, 1):
        try:
            df = download_stock(ticker)
            if df is not None and len(df) > 0:
                filepath = os.path.join(STOCK_DATA_DIR, f"{ticker}.csv")
                df.to_csv(filepath)
                success += 1
                print(f"[{i}/{total}] Downloaded {ticker} ({len(df)} rows)")
            else:
                failed.append(ticker)
                print(f"[{i}/{total}] No data for {ticker}")
        except Exception as e:
            failed.append(ticker)
            print(f"[{i}/{total}] Error downloading {ticker}: {e}")

        # Small delay to avoid rate limiting
        if i % 50 == 0:
            time.sleep(1)

    print(f"\n{'='*50}")
    print(f"Download complete: {success}/{total} stocks")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed[:20])}")


if __name__ == "__main__":
    main()
