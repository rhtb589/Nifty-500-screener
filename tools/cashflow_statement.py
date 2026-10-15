"""Cash Flow Statement payload for the standalone /cashflow/<ticker> page.

Reads data/cashflow/{TICKER}.csv (annual statement, newest period first,
written by refresh_financials.py) and turns it into a JSON-ready payload:
KPI cards with YoY, a sectioned statement table (curated rows like the
reference dashboard), and series for the two charts.

Sections use the subtotal rows yfinance already stores (Operating / Investing
/ Financing Cash Flow, Changes In Cash, Beginning/End Cash Position), so no
totals are recomputed.
"""
from langchain_core.tools import tool

import pandas as pd

from .financials import load_statement
from .base import resolve_ticker, get_row

# Curated display rows, in rendering order. "names" are tried in order until
# one exists in the ticker's cashflow CSV; "label" is the friendly text.
# style: normal | subtotal | grand | highlight   — indent: child of a parent row
OPERATING_ROWS = [
    {"names": ("Net Income From Continuing Operations",), "label": "Net Income",
     "income_fallback": True},
    {"names": ("Depreciation And Amortization", "Depreciation",
               "Depreciation Amortization Depletion"),
     "label": "Depreciation & Amortization"},
    {"names": ("Stock Based Compensation",), "label": "Stock Based Compensation"},
    {"names": ("Other Non Cash Items",), "label": "Other Non-Cash Items"},
    {"names": ("Taxes Refund Paid", "Income Tax Paid Supplemental Data"),
     "label": "Income Taxes Paid (Net)"},
    {"names": ("Gain Loss On Sale Of PPE",), "label": "Gain/Loss on Sale of PPE"},
    {"names": ("Gain Loss On Investment Securities",),
     "label": "Gain/Loss on Investments"},
    {"names": ("Net Foreign Currency Exchange Gain Loss",),
     "label": "Foreign Exchange Gain/Loss"},
    {"names": ("Provisionand Write Offof Assets", "Asset Impairment Charge"),
     "label": "Provisions & Write-offs"},
    {"names": ("Change In Working Capital", "Change In Other Working Capital"),
     "label": "Change in Working Capital"},
    {"names": ("Change In Receivables", "Changes In Account Receivables"),
     "label": "Changes in Receivables", "indent": True},
    {"names": ("Change In Inventory",), "label": "Change in Inventory", "indent": True},
    {"names": ("Change In Payable", "Change In Account Payable"),
     "label": "Change in Payables", "indent": True},
    {"names": ("Change In Other Current Assets",), "label": "Change in Other Current Assets",
     "indent": True},
    {"names": ("Change In Other Current Liabilities",),
     "label": "Change in Other Current Liabilities", "indent": True},
    {"names": ("Operating Cash Flow",), "label": "Operating Cash Flow (A)",
     "style": "subtotal"},
]

INVESTING_ROWS = [
    {"names": ("Capital Expenditure", "Capital Expenditure Reported"),
     "label": "Capital Expenditure (CapEx)"},
    {"names": ("Sale Of PPE",), "label": "Proceeds from Sale of PPE"},
    {"names": ("Net Investment Purchase And Sale",), "label": "Net Investment Purchase / Sales"},
    {"names": ("Net Business Purchase And Sale",), "label": "Net Acquisition / Disposal of Business"},
    {"names": ("Net Intangibles Purchase And Sale",), "label": "Net Intangibles Purchase / Sale"},
    {"names": ("Net Other Investing Changes",), "label": "Other Investing Activities"},
    {"names": ("Interest Received Cfi", "Dividends Received Cfi"),
     "label": "Interest / Dividends Received (Investing)"},
    {"names": ("Investing Cash Flow",), "label": "Investing Cash Flow (B)", "style": "subtotal"},
]

FINANCING_ROWS = [
    {"names": ("Cash Dividends Paid", "Common Stock Dividend Paid", "Preferred Stock Dividend Paid"),
     "label": "Cash Dividends Paid"},
    {"names": ("Net Issuance Payments Of Debt",), "label": "Net Debt Issuance / Repayment"},
    {"names": ("Issuance Of Debt",), "label": "Gross Debt Issuance"},
    {"names": ("Repayment Of Debt", "Long Term Debt Payments"), "label": "Debt Repayments"},
    {"names": ("Net Common Stock Issuance", "Issuance Of Capital Stock"),
     "label": "Net Common Stock Issuance"},
    {"names": ("Repurchase Of Capital Stock",), "label": "Share Buyback / Repurchase"},
    {"names": ("Interest Paid Cff",), "label": "Interest Paid (Financing)"},
    {"names": ("Net Other Financing Charges",), "label": "Other Financing Activities"},
    {"names": ("Financing Cash Flow",), "label": "Financing Cash Flow (C)", "style": "subtotal"},
]

SUMMARY_ROWS = [
    {"names": ("Effect Of Exchange Rate Changes",), "label": "Effect of Exchange Rate Changes"},
    {"names": ("Changes In Cash",), "label": "Net Change in Cash (A + B + C)", "style": "grand"},
    {"names": ("Beginning Cash Position",), "label": "Beginning Cash Position"},
    {"names": ("End Cash Position",), "label": "End Cash Position", "style": "grand"},
    {"names": ("Free Cash Flow",), "label": "Free Cash Flow (FCF = OCF − CapEx)",
     "style": "highlight"},
]

SECTIONS = [
    ("operating", "1. Cash Flow from Operating Activities", OPERATING_ROWS),
    ("investing", "2. Cash Flow from Investing Activities", INVESTING_ROWS),
    ("financing", "3. Cash Flow from Financing Activities", FINANCING_ROWS),
    ("summary", "4. Cash & Cash Equivalents Reconciliation", SUMMARY_ROWS),
]

# KPI cards: (key, label, row names, use absolute value, delta rendered as ₹ not %)
KPIS = [
    ("ocf", "Operating Cash Flow", ("Operating Cash Flow",), False, False),
    ("fcf", "Free Cash Flow (FCF)", ("Free Cash Flow",), False, False),
    ("capex", "Capital Expenditure", ("Capital Expenditure", "Capital Expenditure Reported"), True, False),
    ("dividends", "Cash Dividends Paid",
     ("Cash Dividends Paid", "Common Stock Dividend Paid"), True, False),
    ("end_cash", "Ending Cash Position", ("End Cash Position",), False, True),
]

PERIODS_TO_SHOW = 4


def _series_map(df, name: str) -> dict | None:
    """{period: value} for one statement row (None when the row is absent).

    A period the filing leaves blank is an unreported amount, so it is
    filled with 0.0 — only an entirely absent row yields None.
    """
    if df is None or name not in df.index:
        return None
    s = df.loc[name]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[0]
    return {str(c): (0.0 if pd.isna(v) else float(v)) for c, v in s.items()}


def _row_values(names, cashflow_df, income_df, periods):
    """Values for `names` aligned to `periods` (raw INR, None when unknown)."""
    for name in names:
        got = _series_map(cashflow_df, name)
        if got is not None:
            return [got.get(p) for p in periods]
    # net income fallback: the cashflow file lacks it for ~10 tickers
    for name in ("Net Income", "Net Income Common Stockholders",
                 "Net Income From Continuing And Discontinued Operation"):
        got = _series_map(income_df, name)
        if got is not None:
            return [got.get(p) for p in periods]
    return [None] * len(periods)


def _yoy(latest, prior, absolute=False):
    """YoY percent (or absolute delta when the prior base is unusable)."""
    if latest is None or prior is None:
        return None, None
    if absolute:
        latest, prior = abs(latest), abs(prior)
    delta = latest - prior
    if prior == 0:
        return None, delta
    return delta / abs(prior) * 100.0, delta

@tool
def get_cashflow_statement(ticker: str) -> dict:
    """Cash flow statement for one ticker (JSON-ready)."""
    return build_cashflow_payload(ticker)

def build_cashflow_payload(ticker: str) -> dict:
    """JSON-ready cash flow dashboard payload for one ticker.

    Raises ValueError when the ticker is unknown to the universe or has no
    statement file under data/cashflow/.
    """
    ticker = resolve_ticker(str(ticker).strip())
    row = get_row(ticker)  # raises ValueError for unknown tickers

    cashflow_df = load_statement(ticker, "cashflow")
    if cashflow_df is None or not len(cashflow_df.columns):
        raise ValueError(f"No cash flow statement on disk for '{ticker}'.")

    income_df = load_statement(ticker, "income_statement")
    periods = [str(c) for c in cashflow_df.columns[:PERIODS_TO_SHOW]]
    # display order: oldest -> newest (latest column highlighted on the page)
    display_periods = list(reversed(periods))

    def _fy(p: str) -> str:
        return f"FY {p[:4]}"

    period_meta = [{"label": _fy(p), "date": p} for p in display_periods]

    sections = []
    for key, title, specs in SECTIONS:
        rows = []
        for spec in specs:
            values = _row_values(spec["names"], cashflow_df, income_df, periods)
            values = list(reversed(values))  # align with display_periods
            if all(v is None for v in values):
                continue
            rows.append({
                "label": spec["label"],
                "indent": bool(spec.get("indent")),
                "style": spec.get("style", "normal"),
                "values": values,
            })
        if rows:
            sections.append({"key": key, "title": title, "rows": rows})

    if not sections:
        raise ValueError(f"Cash flow statement for '{ticker}' is empty.")

    kpis = []
    for key, label, names, absolute, cash_delta in KPIS:
        values = list(reversed(_row_values(names, cashflow_df, income_df, periods)))
        present = [v for v in values if v is not None]
        latest = present[-1] if present else None
        prior = present[-2] if len(present) >= 2 else None
        if latest is None:
            continue
        yoy, delta = _yoy(latest, prior, absolute=absolute)
        kpis.append({
            "key": key,
            "label": label,
            "latest": abs(latest) if absolute else latest,
            "prior": abs(prior) if (absolute and prior is not None) else prior,
            "yoy_pct": None if cash_delta else yoy,
            "delta": delta,
            "cash_delta": cash_delta,
            "period": display_periods[-1],
            "prior_period": display_periods[-2] if len(display_periods) > 1 else None,
        })

    def _series(names):
        vals = list(reversed(_row_values(names, cashflow_df, income_df, periods)))
        # absent rows chart as 0 rather than NaN (invalid strict JSON)
        return [0.0 if v is None else v for v in vals]

    trend = {
        "labels": [m["label"] for m in period_meta],
        "net_income": _series(("Net Income From Continuing Operations",)),
        "ocf": _series(("Operating Cash Flow",)),
        "fcf": _series(("Free Cash Flow",)),
    }

    def _latest(names):
        vals = _row_values(names, cashflow_df, income_df, periods)
        for v in vals:  # periods is newest-first, so first non-None is latest
            if v is not None:
                return v
        return None

    breakdown = {
        "labels": ["Operating (A)", "Investing (B)", "Financing (C)", "Net Cash Change"],
        "values": [
            _latest(("Operating Cash Flow",)),
            _latest(("Investing Cash Flow",)),
            _latest(("Financing Cash Flow",)),
            _latest(("Changes In Cash",)),
        ],
    }

    meta = {
        "ticker": ticker,
        "name": str(row.get("shortName") or row.get("longName") or ticker),
        "sector": str(row.get("sector") or ""),
        "industry": str(row.get("industry") or ""),
    }
    return {
        **meta,
        "periods": period_meta,
        "kpis": kpis,
        "sections": sections,
        "trend": trend,
        "breakdown": breakdown,
        "summary": _summary_lines(kpis, breakdown, display_periods),
    }


def _summary_lines(kpis, breakdown, periods) -> list:
    """Short insights for the page footer/insight strip."""
    by_key = {k["key"]: k for k in kpis}
    lines = []

    def _cr(v):
        return f"\u20B9{abs(v) / 1e7:,.0f} Cr"

    fcf = by_key.get("fcf")
    if fcf and fcf.get("yoy_pct") is not None:
        direction = "up" if fcf["yoy_pct"] >= 0 else "down"
        lines.append(
            f"Free cash flow {_cr(fcf['latest'])} in {fcf['period']} — {direction} "
            f"{abs(fcf['yoy_pct']):.1f}% YoY."
        )
    elif fcf:
        lines.append(f"Free cash flow {_cr(fcf['latest'])} in {fcf['period']}.")

    ocf = by_key.get("ocf")
    capex = by_key.get("capex")
    if ocf and capex and capex["latest"]:
        coverage = ocf["latest"] / capex["latest"]
        lines.append(
            f"Operating cash flow covers capex {coverage:.1f}x — "
            f"{'self-funded expansion' if coverage >= 1 else 'external funding required'}."
        )

    end_cash = by_key.get("end_cash")
    if end_cash and end_cash.get("delta") is not None and periods:
        delta = end_cash["delta"]
        if delta > 0:
            lines.append(f"Cash balance grew by {_cr(delta)} during {periods[-1]}.")
        elif delta < 0:
            lines.append(f"Cash balance shrank by {_cr(delta)} during {periods[-1]}.")
        else:
            lines.append(f"Cash balance was flat during {periods[-1]}.")

    if periods:
        lines.append(f"Data as of {periods[-1]}")
    return lines
