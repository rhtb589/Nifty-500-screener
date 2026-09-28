from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .base import _load_all, get_enriched_universe


FIELD_MAP = {
    # Price / Value
    "ticker": "Ticker",
    "price": "currentPrice",
    "close": "currentPrice",
    "previous_close": "previousClose",
    "open": "regularMarketOpen",
    "day_low": "regularMarketDayLow",
    "day_high": "regularMarketDayHigh",
    "market_cap": "marketCap",
    "pe": "trailingPE",
    "trailing_pe": "trailingPE",
    "forward_pe": "forwardPE",
    "pb": "priceToBook",
    "price_to_book": "priceToBook",
    "ps": "priceToSalesTrailing12Months",
    "price_to_sales": "priceToSalesTrailing12Months",
    "ev": "enterpriseValue",
    "enterprise_value": "enterpriseValue",
    "ev_revenue": "enterpriseToRevenue",
    "ev_ebitda": "enterpriseToEbitda",
    "roe": "returnOnEquity",
    "roa": "returnOnAssets",
    "profit_margin": "profitMargins",
    "gross_margin": "grossMargins",
    "operating_margin": "operatingMargins",
    "ebitda_margin": "ebitdaMargins",
    "revenue_growth": "revenueGrowth",
    "earnings_growth": "earningsGrowth",
    "dividend_yield": "dividendYield",
    "payout_ratio": "payoutRatio",
    "five_year_avg_yield": "fiveYearAvgDividendYield",
    "beta": "beta",
    "total_debt": "totalDebt",
    "total_cash": "totalCash",
    "debt_to_equity": "debtToEquity",
    "quick_ratio": "quickRatio",
    "current_ratio": "currentRatio",
    "free_cashflow": "freeCashflow",
    "operating_cashflow": "operatingCashflow",
    "volume": "Volume",
    "avg_volume": "averageVolume",
    "avg_volume_10d": "averageVolume10days",
    "week52_high": "fiftyTwoWeekHigh",
    "week52_low": "fiftyTwoWeekLow",
    "week52_change": "52WeekChange",
    "sma_50": "fiftyDayAverage",
    "sma_200": "twoHundredDayAverage",
    "sma_50_change": "fiftyDayAverageChangePercent",
    "sma_200_change": "twoHundredDayAverageChangePercent",
    "target_mean": "targetMeanPrice",
    "target_median": "targetMedianPrice",
    "analyst_count": "numberOfAnalystOpinions",
    "recommendation": "recommendationMean",
    "day_change": "regularMarketChange",
    "day_change_pct": "regularMarketChangePercent",
    "short_name": "shortName",
    "long_name": "longName",
    "sector": "sector",
    "industry": "industry",
    "eps": "trailingEps",
    "forward_eps": "forwardEps",
    "peg": "trailingPegRatio",
    "book_value": "bookValue",
    "price_position_52w": "price_position_52w",
    "dist_from_50dma": "dist_from_50dma",
    "dist_from_200dma": "dist_from_200dma",
    "volume_ratio": "volume_ratio",
    "forward_peg": "forward_peg",

    # --- NEW: Moving Averages (from stock_data) ---
    "sma_20": "SMA_20",
    "sma_100": "SMA_100",
    "ema_20": "EMA_20",
    "ema_50": "EMA_50",
    "ema_100": "EMA_100",
    "ema_200": "EMA_200",
    "wma_20": "WMA_20",
    "wma_50": "WMA_50",
    "dema_20": "DEMA_20",
    "dema_50": "DEMA_50",
    "tema_20": "TEMA_20",
    "tema_50": "TEMA_50",
    "hma_20": "HMA_20",
    "hma_50": "HMA_50",
    "kama_10": "KAMA_10",
    "zlma_20": "ZLMA_20",
    "zlma_50": "ZLMA_50",
    "vwma_20": "VWMA_20",
    "vwma_50": "VWMA_50",
    "mcgd_14": "MCGD_14",
    "alma_10": "ALMA_10",

    # --- NEW: Momentum Oscillators ---
    "rsi": "RSI_14",
    "rsi_14": "RSI_14",
    "macd": "MACD_12_26_9",
    "macd_hist": "MACDh_12_26_9",
    "macd_signal": "MACDs_12_26_9",
    "stoch_k": "STOCHk_14_3_3",
    "stoch_d": "STOCHd_14_3_3",
    "stoch_h": "STOCHh_14_3_3",
    "stochrsi_k": "STOCHRSIk_14_14_3_3",
    "stochrsi_d": "STOCHRSId_14_14_3_3",
    "willr": "WILLR_14",
    "roc_10": "ROC_10",
    "roc_20": "ROC_20",
    "mom_10": "MOM_10",
    "cci": "CCI_20",
    "ao": "AO_5_34",
    "uo": "UO_7_14_28",
    "fisher": "FISHERT_9_1",
    "fisher_signal": "FISHERTs_9_1",
    "trix": "TRIX_18_9",
    "trix_signal": "TRIXs_18_9",
    "coppock": "COPPOCK_10_14_11",
    "bullp": "BULLP_13",
    "bearp": "BEARP_13",
    "rvgi": "RVGI_14_4",
    "rvgi_signal": "RVGIs_14_4",

    # --- NEW: Volatility & Bands ---
    "atr": "ATR_14",
    "tr": "TR",
    "bb_upper": "BBU_20_2.0_2.0",
    "bb_middle": "BBM_20_2.0_2.0",
    "bb_lower": "BBL_20_2.0_2.0",
    "bb_bandwidth": "BBB_20_2.0_2.0",
    "bb_pct": "BBP_20_2.0_2.0",
    "kc_upper": "KCUe_20_2",
    "kc_middle": "KCLe_20_2",
    "kc_lower": "KCBe_20_2",
    "donchian_upper": "DCU_20_20",
    "donchian_middle": "DCM_20_20",
    "donchian_lower": "DCL_20_20",
    "ui": "UI_14",
    "hist_vol": "HIST_VOL_20",
    "accbl": "ACCBL_20",
    "accbm": "ACCBM_20",
    "accbu": "ACCBU_20",

    # --- NEW: Volume Indicators ---
    "obv": "OBV",
    "mfi": "MFI_14",
    "cmf": "CMF_20",
    "ad": "AD",
    "adosc": "ADOSC_3_10",
    "efi": "EFI_13",
    "vwap": "VWAP",
    "pvt": "PVT",
    "nvi": "NVI",
    "pvo": "PVO_12_26_9",
    "pvo_hist": "PVOh_12_26_9",
    "pvo_signal": "PVOs_12_26_9",
    "rvol_10": "RVOL_10",

    # --- NEW: Statistical ---
    "zscore": "ZSCORE_20",
    "stdev": "STDEV_20",
    "variance": "VARIANCE_20",
    "median": "MEDIAN_20",
    "quantile": "QUANTILE_20",
    "skew": "SKEW_20",
    "kurtosis": "KURT_20",
    "correlation": "CORRELATION_20",
    "covariance": "COVARIANCE_20",
    "linreg": "LINREG_20",
    "slope": "SLOPE_20",
    "entropy": "ENTROPY_10",
    "mad": "MAD_30",

    # --- NEW: Returns ---
    "return_1d": "RETURN_1D",
    "return_5d": "RETURN_5D",
    "return_20d": "RETURN_20D",
    "return_60d": "RETURN_60D",
    "return_120d": "RETURN_120D",
    "return_252d": "RETURN_252D",
    "log_return": "LOG_RETURN",
    "qoq_return": "QOQ_RETURN",
    "qoq_volume_growth": "QOQ_VOLUME_GROWTH",
    "qoq_volatility": "QOQ_VOLATILITY",
    "qoq_hl_range": "QOQ_HL_RANGE",

    # --- NEW: Squeeze ---
    "sqz": "SQZ_20_2_20_1.5",
    "sqz_on": "SQZ_ON",
    "sqz_off": "SQZ_OFF",
    "sqz_no": "SQZ_NO",
    "sqzpro": "SQZPRO_20_2_20_2.0_1.5_1.0",
    "sqzpro_on_wide": "SQZPRO_ON_WIDE",
    "sqzpro_on_normal": "SQZPRO_ON_NORMAL",
    "sqzpro_on_narrow": "SQZPRO_ON_NARROW",
    "sqzpro_off": "SQZPRO_OFF",
    "sqzpro_no": "SQZPRO_NO",

    # --- NEW: Other Technical ---
    "bb_width": "bb_width",
    "donchian_width": "donchian_width",
    "kc_width": "kc_width",
    "adx": "ADX_14",
    "stochrsi_k": "STOCHRSIk_14_14_3_3",
    "stochrsi_d": "STOCHRSId_14_14_3_3",
}

DISPLAY_COLUMNS = [
    "Ticker",
    "shortName",
    "sector",
    "industry",
    "currentPrice",
    "regularMarketChangePercent",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "returnOnEquity",
    "profitMargins",
    "revenueGrowth",
    "earningsGrowth",
    "dividendYield",
    "debtToEquity",
    "volume",
    "fiftyDayAverage",
    "twoHundredDayAverage",
    "52WeekChange",
    # Technical columns for display
    "RSI_14",
    "MACD_12_26_9",
    "MACDs_12_26_9",
    "MACDh_12_26_9",
    "BBP_20_2.0_2.0",
    "SMA_20",
    "SMA_50",
    "SMA_200",
    "EMA_20",
    "EMA_50",
    "EMA_200",
    "ATR_14",
    "BBU_20_2.0_2.0",
    "BBL_20_2.0_2.0",
    "DCU_20_20",
    "DCL_20_20",
    "RVOL_10",
    "CMF_20",
    "MFI_14",
    "VWAP",
    "RETURN_60D",
    "price_position_52w",
    "dist_from_50dma",
    "dist_from_200dma",
    "volume_ratio",
    "forward_peg",
]


def _translate_query(query: str) -> str:
    """
    Replace friendly field names in the query with actual DataFrame column names.
    This handles cases where the LLM uses aliases from FIELD_MAP.
    """
    import re

    result = query
    # Sort by length descending to avoid partial replacements (e.g., 'price' before 'price_to_book')
    
    for alias, col_name in sorted(FIELD_MAP.items(), key=lambda x: -len(x[0])):
        if alias == col_name:
            continue  # Skip if alias is already the column name
        # Match df['alias'] or df["alias"] patterns
        result = re.sub(
            rf"df\['{re.escape(alias)}'\]",
            f"df['{col_name}']",
            result
        )
        result = re.sub(
            rf'df\["{re.escape(alias)}"\]',
            f'df["{col_name}"]',
            result
        )
    return result


def _extract_query_columns(query: str) -> list[str]:
    """
    Determine which DataFrame columns the (already translated) query
    references, so they can be included in the output table.

    Catches df['col'] / df["col"] accesses and bare column names passed to
    sort_values/nlargest/nsmallest.
    """
    import re

    cols: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            cols.append(name)

    for m in re.finditer(r"""df\s*\[\s*['"]([^'"]+)['"]\s*\]""", query):
        _add(m.group(1))
    for m in re.finditer(
        r"""(?:sort_values|sort_values_by|nlargest|nsmallest)\(\s*['"]([^'"]+)['"]""",
        query,
    ):
        _add(m.group(1))
    return cols


def screen_nifty500(query: str) -> Dict[str, Any]:
    """
    Screen NIFTY 500 stocks using a pandas filter query.
    Returns ALL matching stocks. Use .head() in the query to limit if needed.

    The DataFrame 'df' comes from latest_snapshot.csv: fundamentals (nifty_500_data.csv)
    plus latest technical indicators (stock_data/*.csv last row) plus derived columns.
    """
    df = get_enriched_universe()

    # Translate friendly field names to column names
    translated_query = _translate_query(query)
    print(f"[Screener] Original query: {query}")
    print(f"[Screener] Translated query: {translated_query}")

    input_rows = len(df)
    result = _safe_exec_query(df, translated_query)

    output_cols = [c for c in DISPLAY_COLUMNS if c in result.columns]

    # Columns the scan actually filtered/sorted on — always reported back
    # (query_columns) so the output table can show them; appended to the
    # result only if not already present.
    query_cols = [c for c in _extract_query_columns(translated_query) if c in result.columns]
    for c in query_cols:
        if c not in output_cols:
            output_cols.append(c)

    result = result[output_cols]

    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.where(pd.notnull(result), None)

    output_rows = len(result)
    print(f"[Screener] Input rows: {input_rows}, Output rows: {output_rows}")

    # Validation: warn if the query didn't actually filter anything
    if output_rows == input_rows and ".head(" not in translated_query and ".tail(" not in translated_query:
        print("[Screener] WARNING: Query returned all rows — filtering may not have worked as intended")

    return {
        "total_matches": int(output_rows),
        "query_columns": query_cols,
        "results": result.to_dict(orient="records"),
    }


def _safe_exec_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Execute a pandas filter query safely.
    Handles both bare expressions and reassignment statements.
    """
    query_clean = query.strip()

    # 1. Ensure reassignment back to 'df' if it is a bare expression
    # E.g., if query is "df[(df['PE'] < 20)]", convert to "df = df[(df['PE'] < 20)]"
    if not query_clean.startswith("df") or (
        not query_clean.startswith("df =") and not query_clean.startswith("df=")
    ):
        if query_clean.startswith("df[") or query_clean.startswith("df."):
            query_clean = f"df = {query_clean}"

    # 2. Use a single shared namespace dictionary
    # Passing the same dict for globals and locals ensures assignments directly update it
    shared_ns = {
        "__builtins__": {},
        "df": df.copy(),
        "pd": pd,
        "np": np,
    }

    try:
        # Pass shared_ns as the single namespace
        exec(query_clean, shared_ns, shared_ns)
    except TypeError as e:
        if "&" in str(e) or "|" in str(e):
            raise ValueError(
                "Operator precedence error: use parentheses around each comparison. "
                "CORRECT: df = df[(df['col1'] < 20) & (df['col2'] > 0.02)]"
            )
        raise ValueError(f"Query failed: {e}")
    except Exception as e:
        raise ValueError(f"Query failed: {type(e).__name__}: {e}")

    # 3. Retrieve the updated 'df'
    result = shared_ns.get("df")

    if not isinstance(result, pd.DataFrame):
        raise ValueError("Query must leave 'df' as a DataFrame")

    return result

class ScreenerInput(BaseModel):
    query: str = Field(
        description=(
            "Python pandas filter code to execute. The DataFrame 'df' is already loaded with 500 NIFTY stocks "
            "including BOTH fundamentals AND latest technical indicators from stock_data/*.csv. "
            "Write filter expressions on df. ALL matching stocks are returned by default. "
            "Use .head(n) ONLY if you want to limit the number of rows. "
            "CORRECT syntax — always use parentheses around each comparison:\n"
            "  df = df[(df['trailingPE'] < 20) & (df['dividendYield'] > 0.02)]\n"
            "  df = df.sort_values('currentPrice')\n"
            "WRONG — never do this:\n"
            "  df = df[df['trailingPE'] < 20 & df['dividendYield'] > 0.02]\n"
            "Available fundamental columns: Ticker, currentPrice, trailingPE, forwardPE, priceToBook, "
            "marketCap, returnOnEquity, profitMargins, revenueGrowth, earningsGrowth, "
            "dividendYield, debtToEquity, volume, fiftyDayAverage, twoHundredDayAverage, "
            "52WeekChange, price_position_52w, dist_from_50dma, dist_from_200dma, "
            "volume_ratio, forward_peg, sector, industry, eps, forward_eps, beta, "
            "quickRatio, currentRatio, freeCashflow, and more.\n"
            "Available technical columns (latest values from stock_data/*.csv): "
            "RSI_14, MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9, "
            "SMA_20, SMA_50, SMA_100, SMA_200, EMA_20, EMA_50, EMA_100, EMA_200, "
            "BBU_20_2.0_2.0, BBM_20_2.0_2.0, BBL_20_2.0_2.0, BBP_20_2.0_2.0, "
            "DCU_20_20, DCM_20_20, DCL_20_20, "
            "KCLe_20_2, KCBe_20_2, KCUe_20_2, "
            "ATR_14, RVOL_10, CMF_20, MFI_14, VWAP, "
            "RETURN_1D, RETURN_5D, RETURN_20D, RETURN_60D, RETURN_120D, RETURN_252D, "
            "STOCHk_14_3_3, STOCHd_14_3_3, WILLR_14, CCI_20, AO_5_34, UO_7_14_28, "
            "OBV, CMF_20, MFI_14, VWAP, AD, ADOSC_3_10, "
            "SLOPE_20, LINREG_20, "
            "DCU_20_20, DCL_20_20, BBU_20_2.0_2.0, BBL_20_2.0_2.0, BBP_20_2.0_2.0, "
            "RVOL_10, CMF_20, MFI_14, VWAP, RETURN_60D, RETURN_20D, RETURN_120D, "
            "SQZ_ON, SQZ_OFF, SQZ_NO, ATR_14, ADX_14, CCI_20, WILLR_14, UO_7_14_28, "
            "and many more.\n"
            "ALIASES (also accepted): rsi/rsi_14 → RSI_14, macd → MACD_12_26_9, "
            "bb_upper/bb_lower/bb_middle/bb_pct → BBU_20_2.0_2.0/BBL_20_2.0_2.0/BBM_20_2.0_2.0/BBP_20_2.0_2.0, "
            "donchian_upper/donchian_lower → DCU_20_20/DCL_20_20, "
            "kc_upper/kc_lower → KCUe_20_2/KCBe_20_2, "
            "atr → ATR_14, rvol_10 → RVOL_10, cmf → CMF_20, mfi → MFI_14, vwap → VWAP, "
            "rsi > 70, macd > 0, bb_pct > 0.8, etc."
        )
    )


def _screen_wrapper(query: str) -> Dict[str, Any]:
    return screen_nifty500(query)


nifty500_screener_tool = StructuredTool.from_function(
    func=_screen_wrapper,
    name="screen_nifty500",
    description=(
        "Screen NIFTY 500 stocks by executing pandas filter code. "
        "Use this when the user asks to find/filter/search/screen stocks matching any criteria. "
        "The DataFrame 'df' is pre-loaded with 500 Indian stocks AND latest technical indicators "
        "from per-stock historical data (stock_data/*.csv). "
        "Write pandas filter expressions to find matching stocks. "
        "ALL matching stocks are returned automatically — do NOT use .head() unless the user asks for a specific number. "
        "Chain multiple filters with & (AND) or | (OR). "
        "CRITICAL: Always use parentheses around EACH comparison when combining with & or |:\n"
        "  CORRECT: df = df[(df['trailingPE'] < 20) & (df['dividendYield'] > 0.02)]\n"
        "  WRONG:   df = df[df['trailingPE'] < 20 & df['dividendYield'] > 0.02]\n"
        "Use .sort_values() to order results. Only use .head() if user requests a specific limit. "
        "Return only the code string, no explanations."
    ),
    args_schema=ScreenerInput,
)