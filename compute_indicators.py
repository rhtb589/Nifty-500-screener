"""
Compute technical indicators using pandas_ta for all NIFTY 500 stocks.
Reads from stock_data/ and overwrites each CSV with enriched data.
"""

import os
import glob
import pandas as pd
import numpy as np
import pandas_ta as ta

STOCK_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_data")


def compute_moving_averages(df):
    """Compute all moving average indicators."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # SMA
    for length in [20, 50, 100, 200]:
        df[f"SMA_{length}"] = ta.sma(close, length=length)

    # EMA
    for length in [20, 50, 100, 200]:
        df[f"EMA_{length}"] = ta.ema(close, length=length)

    # WMA
    for length in [20, 50]:
        df[f"WMA_{length}"] = ta.wma(close, length=length)

    # DEMA
    for length in [20, 50]:
        df[f"DEMA_{length}"] = ta.dema(close, length=length)

    # TEMA
    for length in [20, 50]:
        df[f"TEMA_{length}"] = ta.tema(close, length=length)

    # HMA
    for length in [20, 50]:
        df[f"HMA_{length}"] = ta.hma(close, length=length)

    # KAMA
    df["KAMA_10"] = ta.kama(close, length=10, fast=2, slow=30)

    # ZLMA
    for length in [20, 50]:
        df[f"ZLMA_{length}"] = ta.zlma(close, length=length)

    # VWMA
    for length in [20, 50]:
        df[f"VWMA_{length}"] = ta.vwma(close, volume, length=length)

    # McGinley Dynamic
    df["MCGD_14"] = ta.mcgd(close, length=14)

    # ALMA
    df["ALMA_10"] = ta.alma(close, length=10, sigma=6, distribution_offset=0.85)

    # Midpoint
    for length in [2, 14]:
        df[f"MIDPOINT_{length}"] = ta.midpoint(close, length=length)

    # Midprice
    df["MIDPRICE_14"] = ta.midprice(high, low, length=14)

    return df


def compute_momentum(df):
    """Compute momentum and oscillator indicators."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # RSI
    df["RSI_14"] = ta.rsi(close, length=14)

    # MACD
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)

    # Stochastic
    stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
    if stoch is not None:
        df = pd.concat([df, stoch], axis=1)

    # Stochastic RSI
    stochrsi = ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3)
    if stochrsi is not None:
        df = pd.concat([df, stochrsi], axis=1)

    # Williams %R
    df["WILLR_14"] = ta.willr(high, low, close, length=14)

    # ROC
    for length in [10, 20]:
        df[f"ROC_{length}"] = ta.roc(close, length=length)

    # Momentum
    df["MOM_10"] = ta.mom(close, length=10)

    # CCI
    df["CCI_20"] = ta.cci(high, low, close, length=20, c=0.015)

    # Awesome Oscillator
    df["AO_5_34"] = ta.ao(high, low, fast=5, slow=34)

    # Ultimate Oscillator
    df["UO_7_14_28"] = ta.uo(high, low, close, fast=7, medium=14, slow=28)

    # Fisher Transform
    fisher = ta.fisher(high, low, length=9, signal=1)
    if fisher is not None:
        df = pd.concat([df, fisher], axis=1)

    # TRIX
    trix = ta.trix(close, length=18, signal=9)
    if trix is not None:
        df = pd.concat([df, trix], axis=1)

    # Coppock Curve
    df["COPPOCK_10_14_11"] = ta.coppock(close, length=10, fast=14, slow=11)

    # Elder Ray Index
    eri = ta.eri(high, low, close, length=13)
    if eri is not None:
        df = pd.concat([df, eri], axis=1)

    # Relative Vigor Index
    rvgi = ta.rvgi(high, low, close, volume, length=14, swma_length=4)
    if rvgi is not None:
        df = pd.concat([df, rvgi], axis=1)

    # Squeeze
    squeeze = ta.squeeze(high, low, close, bb_length=20, bb_std=2, kc_length=20, kc_scalar=1.5)
    if squeeze is not None:
        df = pd.concat([df, squeeze], axis=1)

    # Squeeze Pro
    squeeze_pro = ta.squeeze_pro(high, low, close, bb_length=20, bb_std=2, kc_length=20)
    if squeeze_pro is not None:
        df = pd.concat([df, squeeze_pro], axis=1)

    return df


def compute_volatility(df):
    """Compute volatility indicators."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # ATR
    df["ATR_14"] = ta.atr(high, low, close, length=14)

    # True Range
    df["TR"] = ta.true_range(high, low, close)

    # Bollinger Bands
    bbands = ta.bbands(close, length=20, std=2)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)

    # Keltner Channel
    kc = ta.kc(high, low, close, length=20, scalar=2, atr_length=10)
    if kc is not None:
        df = pd.concat([df, kc], axis=1)

    # Donchian Channel
    donchian = ta.donchian(high, low, lower_length=20, upper_length=20)
    if donchian is not None:
        df = pd.concat([df, donchian], axis=1)

    # Ulcer Index
    df["UI_14"] = ta.ui(close, length=14)

    # Historical Volatility (using log return rolling std)
    log_ret = np.log(close / close.shift(1))
    df["HIST_VOL_20"] = log_ret.rolling(window=20).std() * np.sqrt(252)

    # Acceleration Bands
    accbands = ta.accbands(high, low, close, length=20, c=4)
    if accbands is not None:
        df = pd.concat([df, accbands], axis=1)

    return df


def compute_volume(df):
    """Compute volume indicators."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # On Balance Volume
    df["OBV"] = ta.obv(close, volume)

    # Money Flow Index
    df["MFI_14"] = ta.mfi(high, low, close, volume, length=14)

    # Chaikin Money Flow
    df["CMF_20"] = ta.cmf(high, low, close, volume, length=20)

    # Accumulation/Distribution
    df["AD"] = ta.ad(high, low, close, volume)

    # A/D Oscillator
    df["ADOSC_3_10"] = ta.adosc(high, low, close, volume, fast=3, slow=10)

    # Ease of Movement
    df["EFI_13"] = ta.efi(close, volume, length=13)

    # VWAP (daily reset approximation)
    df["VWAP"] = ta.vwap(high, low, close, volume)

    # Volume Weighted MA
    df["VWMA_20"] = ta.vwma(close, volume, length=20)

    # Price Volume Trend
    df["PVT"] = ta.pvt(close, volume)

    # Negative Volume Index
    df["NVI"] = ta.nvi(close, volume)

    # Volume Oscillator
    pvo = ta.pvo(volume, fast=12, slow=26, signal=9)
    if pvo is not None:
        df = pd.concat([df, pvo], axis=1)

    # Relative Volume (manual: current volume / average volume)
    avg_vol = volume.rolling(window=10).mean()
    df["RVOL_10"] = (volume / avg_vol).where(avg_vol > 0)

    return df


def compute_statistical(df):
    """Compute statistical indicators."""
    close = df["Close"]

    # Z-Score
    df["ZSCORE_20"] = ta.zscore(close, length=20)

    # Standard Deviation
    df["STDEV_20"] = ta.stdev(close, length=20)

    # Variance
    df["VARIANCE_20"] = ta.variance(close, length=20)

    # Median
    df["MEDIAN_20"] = ta.median(close, length=20)

    # Quantile
    df["QUANTILE_20"] = ta.quantile(close, length=20, q=0.5)

    # Skewness
    df["SKEW_20"] = ta.skew(close, length=20)

    # Kurtosis
    df["KURT_20"] = ta.kurtosis(close, length=20)

    # Correlation (rolling correlation of close with itself = 1, use close vs SMA as proxy)
    sma20 = ta.sma(close, length=20)
    df["CORRELATION_20"] = close.rolling(window=20).corr(sma20)

    # Covariance (rolling covariance of close with itself = variance)
    df["COVARIANCE_20"] = close.rolling(window=20).var()

    # Linear Regression
    df["LINREG_20"] = ta.linreg(close, length=20)

    # Slope
    df["SLOPE_20"] = ta.slope(close, length=20)

    # Entropy
    df["ENTROPY_10"] = ta.entropy(close, length=10)

    # Median Absolute Deviation
    df["MAD_30"] = ta.mad(close, length=30)

    return df


def compute_returns(df):
    """Compute return indicators."""
    close = df["Close"]

    # Simple returns
    df["RETURN_1D"] = close.pct_change(periods=1)
    df["RETURN_5D"] = close.pct_change(periods=5)
    df["RETURN_20D"] = close.pct_change(periods=20)
    df["RETURN_60D"] = close.pct_change(periods=60)
    df["RETURN_120D"] = close.pct_change(periods=120)
    df["RETURN_252D"] = close.pct_change(periods=252)

    # Log returns
    df["LOG_RETURN"] = ta.log_return(close, length=1)

    return df


def compute_qoq_growth(df):
    """Compute Quarter-on-Quarter growth metrics from price data."""
    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]
    low = df["Low"]

    quarter_days = 63  # ~1 quarter of trading days

    if len(close) >= quarter_days * 2:
        # QoQ Price Return
        df["QOQ_RETURN"] = (close.iloc[-1] / close.iloc[-quarter_days - 1]) - 1

        # QoQ Volume Growth
        recent_vol = volume.iloc[-quarter_days:].mean()
        prev_vol = volume.iloc[-quarter_days * 2:-quarter_days].mean()
        df["QOQ_VOLUME_GROWTH"] = (recent_vol / prev_vol - 1) if prev_vol > 0 else np.nan

        # QoQ Volatility Change
        recent_ret = close.iloc[-quarter_days:].pct_change().dropna()
        prev_ret = close.iloc[-quarter_days * 2:-quarter_days].pct_change().dropna()
        recent_vol_val = recent_ret.std() * np.sqrt(252)
        prev_vol_val = prev_ret.std() * np.sqrt(252)
        df["QOQ_VOLATILITY"] = (recent_vol_val / prev_vol_val - 1) if prev_vol_val > 0 else np.nan

        # QoQ High-Low Range Change
        recent_range = (high.iloc[-quarter_days:] - low.iloc[-quarter_days:]).mean()
        prev_range = (high.iloc[-quarter_days * 2:-quarter_days] - low.iloc[-quarter_days * 2:-quarter_days]).mean()
        df["QOQ_HL_RANGE"] = (recent_range / prev_range - 1) if prev_range > 0 else np.nan
    else:
        df["QOQ_RETURN"] = np.nan
        df["QOQ_VOLUME_GROWTH"] = np.nan
        df["QOQ_VOLATILITY"] = np.nan
        df["QOQ_HL_RANGE"] = np.nan

    return df


def process_stock(filepath):
    """Process a single stock CSV with all indicators."""
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    except Exception as e:
        return None, f"Read error: {e}"

    if len(df) < 50:
        return None, f"Insufficient data ({len(df)} rows)"

    # Ensure numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any all-NaN rows
    df = df.dropna(how="all")

    try:
        df = compute_moving_averages(df)
        df = compute_momentum(df)
        df = compute_volatility(df)
        df = compute_volume(df)
        df = compute_statistical(df)
        df = compute_returns(df)
        df = compute_qoq_growth(df)
    except Exception as e:
        return None, f"Indicator error: {e}"

    return df, None


def main():
    csv_files = sorted(glob.glob(os.path.join(STOCK_DATA_DIR, "*.csv")))
    total = len(csv_files)

    if total == 0:
        print("No CSV files found in stock_data/. Run download_data.py first.")
        return

    print(f"Computing technical indicators for {total} stocks...\n")

    success = 0
    failed = []

    for i, filepath in enumerate(csv_files, 1):
        ticker = os.path.basename(filepath).replace(".csv", "")
        result, error = process_stock(filepath)

        if result is not None:
            result.to_csv(filepath)
            success += 1
            print(f"[{i}/{total}] Processed {ticker} ({len(result.columns)} columns)")
        else:
            failed.append(ticker)
            print(f"[{i}/{total}] Failed {ticker}: {error}")

    print(f"\n{'='*50}")
    print(f"Processing complete: {success}/{total} stocks")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed[:20])}")


if __name__ == "__main__":
    main()
