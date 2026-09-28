import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nifty_500_data.csv")
STOCK_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "stock_data")
SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "latest_snapshot.csv")

_fund_data = None
_all_data = None
_history_cache: dict = {}
_universe_tech = None
_shortname_map: dict | None = None


def _load_fundamentals():
    """Raw fundamentals from nifty_500_data.csv.

    Source of truth for build_snapshot(); also the last-resort fallback for
    _load_all() when the scanning snapshot cannot be read or built.
    """
    global _fund_data
    if _fund_data is None:
        _fund_data = pd.read_csv(CSV_PATH, index_col=0)
    return _fund_data


def _load_all():
    """Scanning/analysis universe from latest_snapshot.csv, indexed by Ticker.

    The snapshot carries fundamentals + latest technicals + derived columns,
    so every analysis tool reading through this helper (get_row, peer
    universes, cross-sectional ranks) sees the new CSV data.
    """
    global _all_data
    if _all_data is None:
        try:
            uni = get_enriched_universe()
            _all_data = uni.set_index("Ticker") if "Ticker" in uni.columns else uni.copy()
        except Exception:
            _all_data = _load_fundamentals()
    return _all_data


def resolve_ticker(ticker: str) -> str:
    """Map a shortName (what the dropdown/LLM sends) or a symbol to the
    canonical ticker key used by the CSV index and stock_data filenames.

    "ABB INDIA LIMITED" -> "ABB.NS"; "abb.ns"/"ABB.NS" pass through.
    Unknown inputs are returned unchanged (callers decide how to fail).
    """
    global _shortname_map
    name = str(ticker).strip()
    if not name:
        return name
    if _shortname_map is None:
        df = _load_all()
        try:
            _shortname_map = {
                str(s).lower(): str(i) for s, i in zip(df["shortName"], df.index)
            }
        except Exception:
            _shortname_map = {}
    hit = _shortname_map.get(name.lower())
    if hit:
        return hit
    df = _load_all()
    if name in df.index:
        return name
    upper = name.upper()
    return upper if upper in df.index else name


def get_ticker_list():
    df = _load_all()
    return sorted(df["shortName"].tolist())


def get_row(ticker: str) -> pd.Series:
    df = _load_all()
    key = resolve_ticker(ticker)
    if key not in df.index:
        raise ValueError(f"Ticker '{ticker}' not found in Nifty 500 data.")
    return df.loc[key]


def safe_float(val, default=np.nan):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# Comprehensive list of all technical indicators available in stock_data/*.csv
TECH_SNAPSHOT_COLS = [
    # Price & Volume
    "Open", "High", "Low", "Close", "Volume", "VWAP",
    # Moving Averages
    "SMA_20", "SMA_50", "SMA_100", "SMA_200",
    "EMA_20", "EMA_50", "EMA_100", "EMA_200",
    "WMA_20", "WMA_50",
    "DEMA_20", "DEMA_50",
    "TEMA_20", "TEMA_50",
    "HMA_20", "HMA_50",
    "KAMA_10",
    "ZLMA_20", "ZLMA_50",
    "VWMA_20", "VWMA_50",
    "MCGD_14",
    "ALMA_10",
    "MIDPOINT_2", "MIDPOINT_14",
    "MIDPRICE_14",
    # Momentum Oscillators
    "RSI_14",
    "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
    "STOCHk_14_3_3", "STOCHd_14_3_3", "STOCHh_14_3_3",
    "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3",
    "WILLR_14",
    "ROC_10", "ROC_20",
    "MOM_10",
    "CCI_20",
    "AO_5_34",
    "UO_7_14_28",
    "FISHERT_9_1", "FISHERTs_9_1",
    "TRIX_18_9", "TRIXs_18_9",
    "COPPOCK_10_14_11",
    "BULLP_13", "BEARP_13",
    "RVGI_14_4", "RVGIs_14_4",
    # Volatility & Bands
    "ATR_14", "TR",
    "BBL_20_2.0_2.0", "BBM_20_2.0_2.0", "BBU_20_2.0_2.0", "BBB_20_2.0_2.0", "BBP_20_2.0_2.0",
    "KCLe_20_2", "KCBe_20_2", "KCUe_20_2",
    "DCL_20_20", "DCM_20_20", "DCU_20_20",
    "UI_14",
    "HIST_VOL_20",
    "ACCBL_20", "ACCBM_20", "ACCBU_20",
    # Volume Indicators
    "OBV", "MFI_14", "CMF_20", "AD", "ADOSC_3_10", "EFI_13",
    "VWAP", "PVT", "NVI", "PVO_12_26_9", "PVOh_12_26_9", "PVOs_12_26_9", "RVOL_10",
    # Statistical
    "ZSCORE_20", "STDEV_20", "VARIANCE_20", "MEDIAN_20", "QUANTILE_20",
    "SKEW_20", "KURT_20", "CORRELATION_20", "COVARIANCE_20",
    "LINREG_20", "SLOPE_20", "ENTROPY_10", "MAD_30",
    # Returns
    "RETURN_1D", "RETURN_5D", "RETURN_20D", "RETURN_60D", "RETURN_120D", "RETURN_252D",
    "LOG_RETURN", "QOQ_RETURN", "QOQ_VOLUME_GROWTH", "QOQ_VOLATILITY", "QOQ_HL_RANGE",
    # Squeeze
    "SQZ_20_2_20_1.5", "SQZ_ON", "SQZ_OFF", "SQZ_NO",
    "SQZPRO_20_2_20_2.0_1.5_1.0", "SQZPRO_ON_WIDE", "SQZPRO_ON_NORMAL", "SQZPRO_ON_NARROW", "SQZPRO_OFF", "SQZPRO_NO",
    # Channels
    "KCLe_20_2", "KCBe_20_2", "KCUe_20_2",
    "DCL_20_20", "DCM_20_20", "DCU_20_20",
    # Acceleration Bands
    "ACCBL_20", "ACCBM_20", "ACCBU_20",
    # Volume/Flow
    "RVOL_10", "CMF_20", "MFI_14", "VWAP",
    # Returns
    "RETURN_1D", "RETURN_5D", "RETURN_20D", "RETURN_60D", "RETURN_120D", "RETURN_252D",
    "LOG_RETURN", "QOQ_RETURN", "QOQ_VOLUME_GROWTH", "QOQ_VOLATILITY", "QOQ_HL_RANGE",
    # Squeeze
    "SQZ_ON", "SQZ_OFF", "SQZ_NO",
    "SQZPRO_ON_WIDE", "SQZPRO_ON_NORMAL", "SQZPRO_ON_NARROW", "SQZPRO_OFF", "SQZPRO_NO",
    # Other
    "SLOPE_20", "DCU_20_20", "DCL_20_20",
    "BBU_20_2.0_2.0", "BBL_20_2.0_2.0", "BBP_20_2.0_2.0",
    "RVOL_10", "CMF_20", "MFI_14", "RETURN_60D",
]


def _read_tail_row(path: str, wanted: set) -> dict | None:
    """Read only the header and last data line of a CSV (fast; no pandas parse)."""
    try:
        with open(path, "rb") as f:
            header_line = f.readline().decode("utf-8", "replace")
            if not header_line:
                return None
            f.seek(0, 2)
            end = f.tell()
            f.seek(max(0, end - 131072))
            chunk = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    cols = header_line.rstrip("\r\n").split(",")
    lines = [l for l in chunk.split("\n") if l.strip()]
    if not lines:
        return None
    vals = lines[-1].rstrip("\r").split(",")
    if len(vals) != len(cols):
        return None
    return {c: v for c, v in zip(cols, vals) if c in wanted}


def _snapshot_tech_row(ticker: str) -> dict | None:
    """Technical columns for `ticker` read from the scanning snapshot row.

    Fallback for get_stock_snapshot() when the per-stock history file is
    unavailable (e.g. shortName input after resolve, or a missing CSV).
    """
    try:
        uni = get_enriched_universe()
    except Exception:
        return None
    if "Ticker" not in uni.columns:
        return None
    match = uni[uni["Ticker"].astype(str) == ticker]
    if match.empty:
        return None
    row = match.iloc[0]
    return {c: safe_float(row.get(c)) for c in TECH_SNAPSHOT_COLS if c in uni.columns}


def get_stock_snapshot(ticker: str, cols: list = None) -> dict | None:
    """Last bar of stock_data/{ticker}.csv for technical columns (cached).

    Accepts a shortName or a .NS symbol. When the history file is missing,
    falls back to that ticker's row in latest_snapshot.csv.
    """
    ticker = resolve_ticker(ticker).upper().strip()
    if ticker not in _history_cache:
        path = os.path.join(STOCK_DATA_DIR, f"{ticker}.csv")
        raw = _read_tail_row(path, set(TECH_SNAPSHOT_COLS)) if os.path.exists(path) else None
        _history_cache[ticker] = (
            {c: safe_float(v) for c, v in raw.items()} if raw else _snapshot_tech_row(ticker)
        )
    snap = _history_cache[ticker]
    if snap is None:
        return None
    if cols:
        return {c: snap.get(c, np.nan) for c in cols}
    return dict(snap)


def get_universe_tech() -> pd.DataFrame:
    """Last-row CMF_20 / RETURN_60D for the whole universe, indexed by ticker (cached).

    Prefers latest_snapshot.csv (already cached in memory — one read instead
    of scanning 500 files) and falls back to per-file tail reads if the
    snapshot lacks these columns.
    """
    global _universe_tech
    if _universe_tech is None:
        wanted = ["CMF_20", "RETURN_60D"]
        df = None
        try:
            uni = get_enriched_universe()
            if "Ticker" in uni.columns and all(c in uni.columns for c in wanted):
                df = (
                    uni.set_index("Ticker")[wanted]
                    .apply(pd.to_numeric, errors="coerce")
                )
        except Exception:
            df = None
        if df is None:
            rows = {}
            try:
                names = sorted(n for n in os.listdir(STOCK_DATA_DIR) if n.endswith(".csv"))
            except OSError:
                names = []
            for name in names:
                raw = _read_tail_row(os.path.join(STOCK_DATA_DIR, name), set(wanted))
                if not raw:
                    continue
                rows[name[:-4]] = {c: safe_float(raw.get(c)) for c in wanted}
            df = pd.DataFrame.from_dict(rows, orient="index")
        _universe_tech = df
    return _universe_tech


def percentile_rank(value, series) -> float:
    """Empirical percentile rank of value within series: #{x_j <= x} / N (0..1)."""
    v = safe_float(value)
    s = pd.to_numeric(pd.Series(list(series)), errors="coerce").dropna()
    if not np.isfinite(v) or s.empty:
        return np.nan
    return float((s <= v).sum() / len(s))


def percentile_rank_series(series) -> pd.Series:
    """Percentile rank of every value in series (0..1); NaN stays NaN."""
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(pct=True)


def append_as_of(summary: list, row) -> list:
    """Append 'Data as of <date>' to an analysis summary when the row carries
    an As_of value from latest_snapshot.csv. Returns the summary unchanged
    otherwise. Accepts a pandas Series or any mapping with .get()."""
    try:
        v = row.get("As_of")
    except AttributeError:
        v = None
    s = "" if v is None else str(v).strip()
    if s and s.lower() not in ("nan", "nat", "none"):
        summary.append(f"Data as of {s}")
    return summary


def check_row(label: str, ok) -> dict:
    """Metric row rendering a condition as PASS / FAIL / N-A."""
    state = "PASS" if ok is True else "FAIL" if ok is False else "N-A"
    return {"label": label, "value": state}


def verdict_from_checks(checks: list, pass_text: str, fail_text: str,
                        n_a_text: str = "Insufficient data") -> str:
    """Combine boolean/None checks: any False -> fail, else any None -> N-A, else pass."""
    if any(c is False for c in checks):
        return fail_text
    if any(c is None for c in checks):
        return n_a_text
    if all(c is True for c in checks):
        return pass_text
    return n_a_text


def fig_to_html(fig) -> str:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#12121a",
        plot_bgcolor="#0e0e16",
        font=dict(color="#e0e0e0", family="Segoe UI, system-ui, sans-serif"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=350,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def metric_row(label: str, value, fmt: str = "") -> dict:
    if isinstance(value, float):
        if fmt == "pct":
            display = f"{value:.2f}%" if np.isfinite(value) else "--"
        elif fmt == "inr":
            if abs(value) >= 1e12:
                display = f"\u20B9{value/1e12:.2f}L Cr"
            elif abs(value) >= 1e9:
                display = f"\u20B9{value/1e9:.2f} Cr"
            elif abs(value) >= 1e7:
                display = f"\u20B9{value/1e7:.2f} Cr"
            elif abs(value) >= 1e5:
                display = f"\u20B9{value/1e5:.2f} L"
            else:
                display = f"\u20B9{value:,.2f}"
        elif fmt == "price":
            display = f"\u20B9{value:,.2f}" if np.isfinite(value) else "--"
        elif fmt == "ratio":
            display = f"{value:.2f}x" if np.isfinite(value) else "--"
        else:
            display = f"{value:.2f}" if np.isfinite(value) else "--"
    else:
        display = str(value) if value else "--"
    return {"label": label, "value": display}


def make_result(metrics: list[dict], chart_html: str = "", summary: list[str] = None):
    return {
        "metrics": metrics,
        "chart": chart_html,
        "summary": summary or [],
    }


_overview_hist_cache: dict = {}


def _read_history_tail(ticker: str, rows: int = 253) -> pd.DataFrame | None:
    """Last `rows` bars of stock_data/{ticker}.csv (cached).

    Accepts a shortName or a .NS symbol (resolved to the file key).
    """
    ticker = resolve_ticker(ticker).upper().strip()
    if ticker not in _overview_hist_cache:
        path = os.path.join(STOCK_DATA_DIR, f"{ticker}.csv")
        df = None
        if os.path.exists(path):
            try:
                loaded = pd.read_csv(path, index_col=0, parse_dates=True)
                if len(loaded) >= 2:
                    df = loaded.tail(rows)
            except (OSError, ValueError):
                df = None
        _overview_hist_cache[ticker] = df
    return _overview_hist_cache[ticker]


def get_stock_overview(ticker: str) -> dict:
    row = get_row(ticker)
    hist = _read_history_tail(ticker)

    price = prev_close = day_change = day_change_pct = np.nan
    hi_52w = lo_52w = np.nan
    as_of = ""

    if hist is not None:
        close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        if len(close) >= 2:
            price = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            day_change = price - prev_close
            day_change_pct = (day_change / prev_close * 100) if prev_close != 0 else np.nan
        if "High" in hist.columns:
            hi_52w = safe_float(pd.to_numeric(hist["High"], errors="coerce").max())
        if "Low" in hist.columns:
            lo_52w = safe_float(pd.to_numeric(hist["Low"], errors="coerce").min())
        try:
            as_of = pd.to_datetime(hist.index[-1]).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            as_of = ""

    # Fundamentals-CSV fallback when per-stock history is unavailable
    if not np.isfinite(price):
        price = safe_float(row.get("currentPrice"))
        prev_close = safe_float(row.get("previousClose"))
        day_change = safe_float(row.get("regularMarketChange"))
        day_change_pct = safe_float(row.get("fulldayChangePercent"))
        if not np.isfinite(day_change) and np.isfinite(price) and np.isfinite(prev_close):
            day_change = price - prev_close
            day_change_pct = (day_change / prev_close * 100) if prev_close != 0 else np.nan
    if not np.isfinite(lo_52w):
        lo_52w = safe_float(row.get("fiftyTwoWeekLow"))
    if not np.isfinite(hi_52w):
        hi_52w = safe_float(row.get("fiftyTwoWeekHigh"))

    metrics = []
    if as_of:
        metrics.append(metric_row("As of", as_of))
    metrics.extend([
        metric_row("Current Price", price, "price"),
        metric_row("Day Change", day_change, "price"),
        metric_row("Day Change %", day_change_pct, "pct"),
        metric_row("Market Cap", safe_float(row.get("marketCap")), "inr"),
        metric_row("Trailing P/E", safe_float(row.get("trailingPE")), "ratio"),
        metric_row("Forward P/E", safe_float(row.get("forwardPE")), "ratio"),
        metric_row("EPS (TTM)", safe_float(row.get("trailingEps")), "price"),
        metric_row("Price/Book", safe_float(row.get("priceToBook")), "ratio"),
        metric_row("52W Low", lo_52w, "price"),
        metric_row("52W High", hi_52w, "price"),
        metric_row("Dividend Yield", safe_float(row.get("dividendYield")), "pct"),
        metric_row("Beta", safe_float(row.get("beta")), ""),
    ])

    return {
        "ticker": ticker,
        "name": str(row.get("shortName", "") or row.get("longName", "") or ticker),
        "sector": str(row.get("sector", "") or ""),
        "industry": str(row.get("industry", "") or ""),
        "metrics": metrics,
    }


def get_stock_chart(ticker: str) -> str:
    """Build price candlestick + RSI + MACD chart from stock_data CSV."""
    csv_path = os.path.join(STOCK_DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        return ""

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if len(df) < 20:
        return ""

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])

    dates = df.index
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    opn = df["Open"]

    # Create subplots: 2 rows, 2 columns
    # Row 1: Candlestick (spans both columns)
    # Row 2: RSI (left) + MACD (right)
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.7, 0.3],
        column_widths=[0.5, 0.5],
        subplot_titles=("Price", "RSI (14)", "MACD (12, 26, 9)"),
        vertical_spacing=0.06,
        horizontal_spacing=0.06,
        specs=[[{"colspan": 2}, None],
               [{}, {}]],
    )

    # --- Candlestick ---
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(close, opn)]
    fig.add_trace(go.Candlestick(
        x=dates, open=opn, high=high, low=low, close=close,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
        name="Price", showlegend=False,
    ), row=1, col=1)

    # --- SMA overlays ---
    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df["SMA_20"], mode="lines",
            line=dict(color="#ffd93d", width=1.2), name="SMA 20",
        ), row=1, col=1)
    if "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df["SMA_50"], mode="lines",
            line=dict(color="#00d2ff", width=1.2), name="SMA 50",
        ), row=1, col=1)

    # --- RSI ---
    if "RSI_14" in df.columns:
        rsi = df["RSI_14"]
        fig.add_trace(go.Scatter(
            x=dates, y=rsi, mode="lines",
            line=dict(color="#e0e0e0", width=1.2), name="RSI",
            showlegend=False,
        ), row=2, col=1)
        # Overbought / Oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="#ef5350",
                      line_width=0.8, opacity=0.6, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#26a69a",
                      line_width=0.8, opacity=0.6, row=2, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="#ffffff", opacity=0.03,
                      line_width=0, row=2, col=1)

    # --- MACD ---
    if "MACD_12_26_9" in df.columns and "MACDs_12_26_9" in df.columns:
        macd_line = df["MACD_12_26_9"]
        signal_line = df["MACDs_12_26_9"]
        histogram = df["MACDh_12_26_9"] if "MACDh_12_26_9" in df.columns else macd_line - signal_line

        # Histogram bars
        hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in histogram]
        fig.add_trace(go.Bar(
            x=dates, y=histogram, marker_color=hist_colors,
            name="MACD Hist", showlegend=False, opacity=0.6,
        ), row=2, col=2)

        fig.add_trace(go.Scatter(
            x=dates, y=macd_line, mode="lines",
            line=dict(color="#00d2ff", width=1.2), name="MACD",
            showlegend=False,
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=dates, y=signal_line, mode="lines",
            line=dict(color="#ff9800", width=1.2), name="Signal",
            showlegend=False,
        ), row=2, col=2)
        fig.add_hline(y=0, line_color="#555555", line_width=0.5, row=2, col=2)

    # --- Layout ---
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#12121a",
        plot_bgcolor="#0e0e16",
        font=dict(color="#e0e0e0", family="Segoe UI, system-ui, sans-serif"),
        height=520,
        margin=dict(l=50, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        xaxis_rangeslider_visible=False,
    )

    # Style subplot axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(gridcolor="#1a1a28", row=i, col=j)
            fig.update_yaxes(gridcolor="#1a1a28", row=i, col=j)

    fig.update_yaxes(title_text="Price (\u20B9)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=2)

    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


_enriched_universe = None


def build_snapshot(write_path: str = SNAPSHOT_PATH) -> pd.DataFrame:
    """
    Build the enriched universe used for scanning and persist it to CSV.

    One row per ticker:
      - fundamentals from nifty_500_data.csv (376 columns)
      - latest technical indicators from stock_data/*.csv (last row)
      - derived columns (_compute_derived_enriched)
      - As_of: latest bar date for that ticker

    Writes `latest_snapshot.csv` (unless write_path is falsy) and caches
    the DataFrame in memory. Returns the DataFrame.
    """
    global _enriched_universe

    # 1. Load raw fundamentals (NOT _load_all — that reads this function's
    #    own output via get_enriched_universe).
    df_fund = _load_fundamentals().copy()
    df_fund.index.name = "Ticker"
    df_fund = df_fund.reset_index()

    # Technical column names also exist in nifty_500_data.csv (empty or stale
    # ~Sep 11 snapshot). Drop them so the merge has no _x/_y suffixes and the
    # fresh stock_data values win.
    tech_names = set(TECH_SNAPSHOT_COLS)
    df_fund = df_fund[[c for c in df_fund.columns if c not in tech_names]]

    # 2. Extract latest technical indicators from all stock_data files
    wanted_cols = set(TECH_SNAPSHOT_COLS) | {"Date"}
    tech_rows = {}
    as_of_rows = {}

    try:
        names = sorted(n for n in os.listdir(STOCK_DATA_DIR) if n.endswith(".csv"))
    except OSError:
        names = []

    for name in names:
        ticker = name[:-4]  # remove .csv
        raw = _read_tail_row(os.path.join(STOCK_DATA_DIR, name), wanted_cols)
        if not raw:
            continue
        date_str = str(raw.get("Date") or "")
        as_of_rows[ticker] = date_str.split(" ")[0]
        tech_rows[ticker] = {c: safe_float(raw.get(c)) for c in (wanted_cols - {"Date"})}

    if not tech_rows:
        merged = df_fund
    else:
        df_tech = pd.DataFrame.from_dict(tech_rows, orient="index")
        df_tech.index.name = "Ticker"
        df_tech = df_tech.reset_index()

        # 3. Merge fundamentals + technicals on Ticker (outer join keeps all tickers)
        merged = pd.merge(df_fund, df_tech, on="Ticker", how="outer")

    # 4. Latest bar date per ticker
    merged["As_of"] = merged["Ticker"].map(as_of_rows).fillna("")

    # 5. Recompute derived columns that depend on technicals
    merged = _compute_derived_enriched(merged)

    if write_path:
        merged.to_csv(write_path, index=False)

    _enriched_universe = merged
    return merged


def get_enriched_universe() -> pd.DataFrame:
    """
    Return the scanning universe from latest_snapshot.csv.

    The snapshot holds one row per ticker: fundamentals + latest technical
    indicators + derived columns + As_of date. If the file is missing or
    empty it is rebuilt automatically (and written to disk).
    """
    global _enriched_universe
    if _enriched_universe is not None:
        return _enriched_universe

    if os.path.exists(SNAPSHOT_PATH) and os.path.getsize(SNAPSHOT_PATH) > 0:
        try:
            df = pd.read_csv(SNAPSHOT_PATH)
        except (OSError, ValueError, pd.errors.ParserError):
            df = None
        if df is not None and len(df):
            _enriched_universe = df
            return _enriched_universe

    # Auto-rebuild if missing (or unreadable/empty)
    return build_snapshot()


def _compute_derived_enriched(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived columns on the enriched universe DataFrame."""
    df = df.copy()

    def _num(name: str) -> pd.Series:
        # Always returns a Series — df.get() would yield None for a missing
        # column, and pd.to_numeric(None) is a scalar NaN (no .notna()).
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    price = _num("currentPrice")
    high52 = _num("fiftyTwoWeekHigh")
    low52 = _num("fiftyTwoWeekLow")
    sma50 = _num("SMA_50")
    if not sma50.notna().any():
        sma50 = _num("fiftyDayAverage")
    sma200 = _num("SMA_200")
    if not sma200.notna().any():
        sma200 = _num("twoHundredDayAverage")
    vol = _num("Volume")
    avg_vol = _num("averageVolume")
    fwd_pe = _num("forwardPE")
    earnings_g = _num("earningsGrowth")

    range_52 = high52 - low52
    df["price_position_52w"] = ((price - low52) / range_52 * 100).where(range_52 > 0)

    df["dist_from_50dma"] = ((price - sma50) / sma50 * 100).where(sma50 > 0)
    df["dist_from_200dma"] = ((price - sma200) / sma200 * 100).where(sma200 > 0)

    df["volume_ratio"] = (vol / avg_vol).where(avg_vol > 0)

    df["forward_peg"] = (fwd_pe / (earnings_g * 100)).where((earnings_g > 0) & fwd_pe.notna())

    # Technical-derived columns
    bb_upper = _num("BBU_20_2.0_2.0")
    bb_lower = _num("BBL_20_2.0_2.0")
    bb_mid = _num("BBM_20_2.0_2.0")
    df["bb_width"] = ((bb_upper - bb_lower) / bb_mid * 100).where(bb_mid > 0)

    donchian_upper = _num("DCU_20_20")
    donchian_lower = _num("DCL_20_20")
    donchian_mid = _num("DCM_20_20")
    df["donchian_width"] = ((donchian_upper - donchian_lower) / donchian_mid * 100).where(donchian_mid > 0)

    kc_upper = _num("KCUe_20_2")
    kc_lower = _num("KCBe_20_2")
    kc_mid = _num("KCLe_20_2")
    df["kc_width"] = ((kc_upper - kc_lower) / kc_mid * 100).where(kc_mid > 0)

    return df
