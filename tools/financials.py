"""Derive fundamentals the yfinance snapshot is missing from data/.

Every NIFTY 500 ticker has annual statements on disk (refresh_financials.py):

    data/balance_sheet/{TICKER}.csv
    data/income_statement/{TICKER}.csv
    data/cashflow/{TICKER}.csv
    data/quarterly_results/income_statement/{TICKER}.csv

`statement_metrics()` turns the latest (and prior) reporting period of those
files into the same field names/units used by nifty_500_data.csv, so the
analysis tools can fill a metric that came back NaN from the snapshot.

Only finite values are returned — callers decide what to fill.
This module must not import tools.base (base imports it).
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# statement folder (relative to data/) -> statement kind used by load_statement
KINDS = (
    "balance_sheet",
    "income_statement",
    "cashflow",
    "quarterly_results/income_statement",
)

# field -> short label used in the analysis summary source note
DERIVED_LABELS = {
    "returnOnEquity": "ROE",
    "returnOnAssets": "ROA",
    "profitMargins": "Net Margin",
    "grossMargins": "Gross Margin",
    "operatingMargins": "Operating Margin",
    "ebitdaMargins": "EBITDA Margin",
    "revenuePerShare": "Revenue/Share",
    "currentRatio": "Current Ratio",
    "quickRatio": "Quick Ratio",
    "totalDebt": "Total Debt",
    "totalCash": "Total Cash",
    "totalCashPerShare": "Cash/Share",
    "debtToEquity": "D/E",
    "operatingCashflow": "Operating Cash Flow",
    "freeCashflow": "Free Cash Flow",
    "payoutRatio": "Payout Ratio",
    "dividendRate": "Dividend Rate",
    "earningsGrowth": "Earnings Growth",
    "revenueGrowth": "Revenue Growth",
    "earningsQuarterlyGrowth": "Quarterly Growth",
    "trailingPE": "Trailing P/E",
    "enterpriseToEbitda": "EV/EBITDA",
    "enterpriseToRevenue": "EV/Revenue",
}

_STMT_CACHE: dict = {}
_METRICS_CACHE: dict = {}


def reset_caches():
    """Drop cached statement frames/metrics (wired into base.reset_caches)."""
    _STMT_CACHE.clear()
    _METRICS_CACHE.clear()


def _f(val) -> float:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def load_statement(ticker: str, kind: str) -> pd.DataFrame | None:
    """One ticker's statement: line items x periods, newest period first.

    Missing/unreadable files yield None so a partial data/ folder never
    breaks an analysis run.
    """
    key = (ticker, kind)
    if key in _STMT_CACHE:
        return _STMT_CACHE[key]

    path = os.path.join(DATA_DIR, kind, f"{ticker}.csv")
    df = None
    if os.path.isfile(path):
        try:
            df = pd.read_csv(path, index_col=0)
        except (OSError, ValueError, pd.errors.ParserError):
            df = None

    if df is not None and len(df):
        df = df[~df.index.duplicated(keep="first")]
        dated = []
        for col in df.columns:
            try:
                ts = pd.Timestamp(str(col))
            except (ValueError, TypeError):
                continue
            dated.append((ts, col))
        if dated:
            dated.sort(key=lambda p: p[0], reverse=True)
            df = df[[c for _, c in dated]]
            df = df.apply(pd.to_numeric, errors="coerce")
        else:
            df = None
    else:
        df = None

    _STMT_CACHE[key] = df
    return df


def _series(df, name: str) -> pd.Series | None:
    if df is None or name not in df.index:
        return None
    s = df.loc[name]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[0]
    return s


def _latest(df, *names) -> float:
    """Value of the newest period that actually carries a number."""
    for name in names:
        s = _series(df, name)
        if s is None:
            continue
        for v in s.values:
            f = _f(v)
            if np.isfinite(f):
                return f
    return np.nan


def _yoy(df, *names) -> float:
    """Growth vs the previous reporting period (latest finite / prior finite)."""
    if df is None:
        return np.nan
    for name in names:
        s = _series(df, name)
        if s is None:
            continue
        vals = [_f(v) for v in s.values]
        i = next((k for k, v in enumerate(vals) if np.isfinite(v)), None)
        if i is None:
            continue
        j = next((k for k in range(i + 1, len(vals)) if np.isfinite(vals[k])), None)
        if j is None:
            continue
        if vals[j] == 0:
            continue
        return vals[i] / vals[j] - 1.0
    return np.nan


def _quarterly_yoy(df, *names) -> float:
    """YoY growth between the latest quarter and the same quarter a year ago."""
    if df is None or len(df.columns) < 2:
        return np.nan
    dates = []
    for col in df.columns:
        try:
            dates.append(pd.Timestamp(str(col)))
        except (ValueError, TypeError):
            return np.nan
    for name in names:
        s = _series(df, name)
        if s is None:
            continue
        vals = [_f(v) for v in s.values]
        i = next((k for k, v in enumerate(vals) if np.isfinite(v)), None)
        if i is None:
            continue
        target = dates[i] - pd.Timedelta(days=365)
        best = None
        for j in range(i + 1, len(vals)):
            if not np.isfinite(vals[j]):
                continue
            delta = abs((dates[j] - target).days)
            if delta <= 45 and (best is None or delta < best[0]):
                best = (delta, j)
        if best and vals[best[1]] != 0:
            return vals[i] / vals[best[1]] - 1.0
    return np.nan


def _ratio(num: float, den: float) -> float:
    if np.isfinite(num) and np.isfinite(den) and den != 0:
        return num / den
    return np.nan


def statement_metrics(ticker: str, price=np.nan, market_cap=np.nan,
                      enterprise_value=np.nan) -> tuple[dict, list]:
    """Fundamentals derivable from data/ for one ticker.

    Returns ({field: value}, [filled field keys]) using nifty_500_data.csv
    units: ROE/margin/growth ratios as decimals, debtToEquity in percent,
    cash/debt/FCF in absolute INR, per-share values in INR.

    `price`, `market_cap` and `enterprise_value` are optional inputs from the
    snapshot row (market_cap / enterprise_value feed the EV multiples).
    """
    cache_key = (
        ticker,
        _f(price),
        _f(market_cap),
        _f(enterprise_value),
    )
    if cache_key in _METRICS_CACHE:
        return _METRICS_CACHE[cache_key]

    bs = load_statement(ticker, "balance_sheet")
    inc = load_statement(ticker, "income_statement")
    cf = load_statement(ticker, "cashflow")
    qinc = load_statement(ticker, "quarterly_results/income_statement")

    out: dict = {}

    # --- balance sheet (latest period) ---
    equity = _latest(bs, "Stockholders Equity", "Common Stock Equity")
    total_assets = _latest(bs, "Total Assets")
    current_assets = _latest(bs, "Current Assets")
    current_liab = _latest(bs, "Current Liabilities")
    # Banks/financials often omit Inventory — treat as 0 there so quick
    # ratio still lands close to current ratio instead of staying blank.
    inventory = _latest(bs, "Inventory")
    if not np.isfinite(inventory):
        inventory = 0.0 if np.isfinite(current_assets) else np.nan
    total_debt = _latest(bs, "Total Debt")
    cash = _latest(bs, "Cash Cash Equivalents And Short Term Investments",
                   "Cash And Cash Equivalents")
    shares = _latest(bs, "Ordinary Shares Number")

    # --- income statement (latest period) ---
    revenue = _latest(inc, "Total Revenue", "Operating Revenue")
    gross_profit = _latest(inc, "Gross Profit")
    operating_income = _latest(inc, "Operating Income")
    net_income = _latest(inc, "Net Income", "Net Income Common Stockholders")
    ni_common = _latest(inc, "Net Income Common Stockholders", "Net Income")
    ebitda = _latest(inc, "EBITDA", "Normalized EBITDA")
    diluted_eps = _latest(inc, "Diluted EPS")
    diluted_shares = _latest(inc, "Diluted Average Shares")

    # --- cash flow (latest period) ---
    ocf = _latest(cf, "Operating Cash Flow")
    fcf = _latest(cf, "Free Cash Flow")
    div_paid = _latest(cf, "Cash Dividends Paid")
    if np.isfinite(div_paid):
        div_paid = abs(div_paid)

    def _put(key: str, val: float):
        if np.isfinite(val):
            out[key] = float(val)

    _put("returnOnEquity", _ratio(ni_common, equity))
    _put("returnOnAssets", _ratio(net_income, total_assets))
    _put("profitMargins", _ratio(net_income, revenue))
    _put("grossMargins", _ratio(gross_profit, revenue))
    _put("operatingMargins", _ratio(operating_income, revenue))
    _put("ebitdaMargins", _ratio(ebitda, revenue))
    _put("revenuePerShare", _ratio(revenue, diluted_shares))

    _put("currentRatio", _ratio(current_assets, current_liab))
    _put("quickRatio", _ratio(
        current_assets - inventory if np.isfinite(current_assets) else np.nan,
        current_liab,
    ))
    _put("totalDebt", total_debt)
    _put("totalCash", cash)
    _put("totalCashPerShare", _ratio(cash, shares))
    de = _ratio(total_debt, equity)
    _put("debtToEquity", de * 100 if np.isfinite(de) else np.nan)

    _put("operatingCashflow", ocf)
    _put("freeCashflow", fcf)
    _put("payoutRatio", _ratio(div_paid, net_income if net_income > 0 else np.nan))
    _put("dividendRate", _ratio(div_paid, diluted_shares))

    _put("earningsGrowth", _yoy(inc, "Net Income Common Stockholders", "Net Income"))
    _put("revenueGrowth", _yoy(inc, "Total Revenue", "Operating Revenue"))
    _put("earningsQuarterlyGrowth",
         _quarterly_yoy(qinc, "Net Income Common Stockholders", "Net Income"))

    # --- price-dependent multiples ---
    _put("trailingPE", _ratio(price, diluted_eps) if diluted_eps and diluted_eps > 0 else np.nan)

    ev = _f(enterprise_value)
    if not np.isfinite(ev):
        mcap = _f(market_cap)
        if np.isfinite(mcap):
            debt_for_ev = total_debt if np.isfinite(total_debt) else 0.0
            cash_for_ev = cash if np.isfinite(cash) else 0.0
            ev = mcap + debt_for_ev - cash_for_ev
    _put("enterpriseToEbitda", _ratio(ev, ebitda) if ebitda and ebitda > 0 else np.nan)
    _put("enterpriseToRevenue", _ratio(ev, revenue))

    filled = list(out.keys())
    result = (out, filled)
    _METRICS_CACHE[cache_key] = result
    return result


def derived_labels(keys) -> list:
    """Short display labels for fields filled from data/ statements."""
    return [DERIVED_LABELS.get(k, k) for k in keys]
