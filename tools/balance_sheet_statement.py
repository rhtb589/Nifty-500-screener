"""Balance Sheet Statement payload for the standalone /balance_sheet/<ticker> page.

Reads data/balance_sheet/{TICKER}.csv (annual statement, newest period first,
written by refresh_financials.py) and turns it into a JSON-ready payload:
KPI cards (Total Assets, Net Worth, Current Ratio, Debt-to-Equity), a
sectioned statement table (curated rows like the reference dashboard), and
series for the capital-structure chart + asset-composition doughnut.

Sections use the subtotal rows yfinance already stores (Total Non Current
Assets, Current Assets, Stockholders Equity, Total Non Current Liabilities
Net Minority Interest, Current Liabilities), so no totals are recomputed —
except TOTAL EQUITY & LIABILITIES, which is equity + liabilities.
"""



import pandas as pd
from langchain_core.tools import tool

from .financials import load_statement
from .base import resolve_ticker, get_row

# Curated display rows, in rendering order. "names" are tried in order until
# one exists in the ticker's balance sheet CSV; "label" is the friendly text.
# style: normal | subtotal | grand   — indent: child of a parent row
ASSETS_NONCURRENT_ROWS = [
    {"names": ("Net PPE", "Gross PPE"), "label": "Property, Plant & Equipment"},
    {"names": ("Construction In Progress",), "label": "Capital Work-in-Progress"},
    {"names": ("Goodwill And Other Intangible Assets", "Other Intangible Assets"),
     "label": "Goodwill & Other Intangible Assets"},
    {"names": ("Investmentin Financial Assets", "Available For Sale Securities"),
     "label": "Financial Investments"},
    {"names": ("Non Current Deferred Taxes Assets",), "label": "Deferred Tax Assets (Non-Current)"},
    {"names": ("Other Non Current Assets",), "label": "Other Non-Current Assets"},
    {"names": ("Total Non Current Assets",), "label": "Total Non-Current Assets",
     "style": "subtotal"},
]

ASSETS_CURRENT_ROWS = [
    {"names": ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments",
               "Cash Cash Equivalents And Federal Funds Sold"),
     "label": "Cash & Cash Equivalents"},
    {"names": ("Accounts Receivable", "Gross Accounts Receivable", "Receivables"),
     "label": "Trade Receivables"},
    {"names": ("Inventory", "Other Inventories"), "label": "Inventories"},
    {"names": ("Prepaid Assets",), "label": "Prepaid Expenses"},
    {"names": ("Other Current Assets", "Restricted Cash"), "label": "Other Current Assets"},
    {"names": ("Current Assets",), "label": "Total Current Assets", "style": "subtotal"},
]

ASSETS_TOTAL_ROWS = [
    {"names": ("Total Assets",), "label": "TOTAL ASSETS", "style": "grand"},
]

EQUITY_ROWS = [
    {"names": ("Capital Stock", "Common Stock", "Share Issued"), "label": "Equity Share Capital"},
    {"names": ("Additional Paid In Capital",), "label": "Additional Paid-in Capital"},
    {"names": ("Retained Earnings",), "label": "Retained Earnings"},
    {"names": ("Other Equity Interest", "Gains Losses Not Affecting Retained Earnings",
               "Treasury Stock"), "label": "Other Equity Adjustments"},
    {"names": ("Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"),
     "label": "Total Equity / Net Worth", "style": "subtotal"},
]

LIAB_NONCURRENT_ROWS = [
    {"names": ("Long Term Debt And Capital Lease Obligation", "Long Term Debt"),
     "label": "Non-Current Borrowings / Lease Liabilities"},
    {"names": ("Long Term Provisions",), "label": "Long-Term Provisions"},
    {"names": ("Other Non Current Liabilities", "Non Current Deferred Liabilities"),
     "label": "Other Non-Current Liabilities"},
    {"names": ("Total Non Current Liabilities Net Minority Interest",),
     "label": "Total Non-Current Liabilities", "style": "subtotal"},
]

LIAB_CURRENT_ROWS = [
    {"names": ("Current Debt And Capital Lease Obligation", "Current Debt",
               "Other Current Borrowings"), "label": "Current Borrowings / Lease Liabilities"},
    {"names": ("Accounts Payable", "Payables", "Payables And Accrued Expenses"),
     "label": "Trade Payables"},
    {"names": ("Current Provisions", "Pensionand Other Post Retirement Benefit Plans Current"),
     "label": "Current Provisions"},
    {"names": ("Other Current Liabilities", "Current Accrued Expenses"),
     "label": "Other Current Liabilities"},
    {"names": ("Current Liabilities",), "label": "Total Current Liabilities", "style": "subtotal"},
]

SECTIONS = [
    ("assets_noncurrent", "1. Non-Current Assets", ASSETS_NONCURRENT_ROWS),
    ("assets_current", "2. Current Assets", ASSETS_CURRENT_ROWS),
    ("assets_total", "3. Total Assets", ASSETS_TOTAL_ROWS),
    ("equity", "4. Equity", EQUITY_ROWS),
    ("liab_noncurrent", "5. Non-Current Liabilities", LIAB_NONCURRENT_ROWS),
    ("liab_current", "6. Current Liabilities", LIAB_CURRENT_ROWS),
]

PERIODS_TO_SHOW = 4

# Doughnut chart: label -> row names (tried in order)
COMPOSITION_BUCKETS = [
    ("Cash & Equivalents", ("Cash And Cash Equivalents",
                            "Cash Cash Equivalents And Short Term Investments",
                            "Cash Cash Equivalents And Federal Funds Sold")),
    ("Trade Receivables", ("Accounts Receivable", "Gross Accounts Receivable", "Receivables")),
    ("Inventories", ("Inventory", "Other Inventories")),
    ("Net PPE", ("Net PPE", "Gross PPE")),
]


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


def _row_values(names, df, periods):
    """Values for `names` aligned to `periods` (raw INR, None when unknown)."""
    for name in names:
        got = _series_map(df, name)
        if got is not None:
            return [got.get(p) for p in periods]
    return [None] * len(periods)


def _yoy(latest, prior):
    """YoY percent (None when the prior base is unusable)."""
    if latest is None or prior is None:
        return None, None
    delta = latest - prior
    if prior == 0:
        return None, delta
    return delta / abs(prior) * 100.0, delta
@tool
def get_balance_sheet(ticker: str) -> dict:
    """Balance sheet statement for one ticker (JSON-ready)."""
    return build_balance_sheet_payload(ticker)

def build_balance_sheet_payload(ticker: str) -> dict:
    """JSON-ready balance sheet dashboard payload for one ticker.

    Raises ValueError when the ticker is unknown to the universe or has no
    statement file under data/balance_sheet/.
    """
    ticker = resolve_ticker(str(ticker).strip())
    row = get_row(ticker)  # raises ValueError for unknown tickers

    bs_df = load_statement(ticker, "balance_sheet")
    if bs_df is None or not len(bs_df.columns):
        raise ValueError(f"No balance sheet statement on disk for '{ticker}'.")

    periods = [str(c) for c in bs_df.columns[:PERIODS_TO_SHOW]]
    # display order: oldest -> newest (latest column highlighted on the page)
    display_periods = list(reversed(periods))

    def _fy(p: str) -> str:
        return f"FY {p[:4]}"

    period_meta = [{"label": _fy(p), "date": p} for p in display_periods]

    sections = []
    for key, title, specs in SECTIONS:
        rows = []
        for spec in specs:
            values = list(reversed(_row_values(spec["names"], bs_df, periods)))
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

    # TOTAL EQUITY & LIABILITIES: equity + total liabilities (not stored by yfinance)
    equity_vals = list(reversed(_row_values(
        ("Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"),
        bs_df, periods)))
    liab_vals = list(reversed(_row_values(
        ("Total Liabilities Net Minority Interest",), bs_df, periods)))
    total_el = [
        (e + l) if (e is not None and l is not None) else None
        for e, l in zip(equity_vals, liab_vals)
    ]
    if any(v is not None for v in total_el):
        sections.append({
            "key": "total_el",
            "title": "7. Total Equity & Liabilities",
            "rows": [{
                "label": "TOTAL EQUITY & LIABILITIES",
                "indent": False,
                "style": "grand",
                "values": total_el,
            }],
        })

    if not sections:
        raise ValueError(f"Balance sheet for '{ticker}' is empty.")

    # ---- KPI cards ----
    kpis = []

    def _latest_prior(names):
        vals = list(reversed(_row_values(names, bs_df, periods)))
        present = [v for v in vals if v is not None]
        latest = present[-1] if present else None
        prior = present[-2] if len(present) >= 2 else None
        return latest, prior

    # 1. Total Assets (YoY %)
    assets_latest, assets_prior = _latest_prior(("Total Assets",))
    if assets_latest is not None:
        yoy, _ = _yoy(assets_latest, assets_prior)
        kpis.append({
            "key": "assets",
            "label": "Total Assets",
            "latest": assets_latest,
            "prior": assets_prior,
            "yoy_pct": yoy,
            "period": display_periods[-1],
            "prior_period": display_periods[-2] if len(display_periods) > 1 else None,
        })

    # 2. Net Worth / Equity (with paid-up + reserves sub-line)
    equity_latest, equity_prior = _latest_prior(
        ("Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"))
    if equity_latest is not None:
        paid_up, _ = _latest_prior(("Capital Stock", "Common Stock", "Share Issued"))
        reserves, _ = _latest_prior(("Retained Earnings", "Other Equity Interest"))
        yoy, _ = _yoy(equity_latest, equity_prior)
        kpis.append({
            "key": "equity",
            "label": "Net Worth / Equity",
            "latest": equity_latest,
            "prior": equity_prior,
            "yoy_pct": yoy,
            "paid_up": paid_up,
            "reserves": reserves,
            "period": display_periods[-1],
            "prior_period": display_periods[-2] if len(display_periods) > 1 else None,
        })

    # 3. Current Ratio (CA / CL) — skipped when rows absent (e.g. banks)
    ca_latest, ca_prior = _latest_prior(("Current Assets",))
    cl_latest, cl_prior = _latest_prior(("Current Liabilities",))
    if ca_latest is not None and cl_latest:
        ratio = ca_latest / cl_latest
        prior_ratio = (ca_prior / cl_prior) if (ca_prior is not None and cl_prior) else None
        if ratio >= 2.0:
            health, tone = "Healthy liquidity", "good"
        elif ratio >= 1.0:
            health, tone = "Adequate liquidity", "warn"
        else:
            health, tone = "Liquidity stress", "bad"
        kpis.append({
            "key": "current_ratio",
            "label": "Current Ratio",
            "ratio": round(ratio, 2),
            "prior_ratio": round(prior_ratio, 2) if prior_ratio is not None else None,
            "health": health,
            "tone": tone,
            "period": display_periods[-1],
        })

    # 4. Debt-to-Equity (Total Debt / Equity)
    debt_latest, debt_prior = _latest_prior(("Total Debt",))
    if debt_latest is not None and equity_latest:
        de = debt_latest / equity_latest
        prior_de = (debt_prior / equity_prior) if (debt_prior is not None and equity_prior) else None
        if de < 0.5:
            health, tone = "Conservative leverage", "good"
        elif de < 1.5:
            health, tone = "Moderate leverage", "warn"
        else:
            health, tone = "High leverage", "bad"
        kpis.append({
            "key": "debt_equity",
            "label": "Debt-to-Equity (D/E)",
            "ratio": round(de, 2),
            "prior_ratio": round(prior_de, 2) if prior_de is not None else None,
            "health": health,
            "tone": tone,
            "period": display_periods[-1],
        })

    # ---- Capital structure trend (Total Assets vs Equity vs Total Liabilities) ----
    liab_display = list(reversed(liab_vals))
    trend = {
        "labels": [m["label"] for m in period_meta],
        # absent rows chart as 0 rather than NaN (invalid strict JSON)
        "assets": [0.0 if v is None else v for v in list(reversed(_row_values(("Total Assets",), bs_df, periods)))],
        "equity": [0.0 if v is None else v for v in equity_vals],
        "liabilities": [0.0 if v is None else v for v in liab_display],
    }

    # ---- Asset composition (latest FY) ----
    total_assets_latest, _ = _latest_prior(("Total Assets",))
    composition = {"labels": [], "values": []}
    if total_assets_latest is not None:
        named_sum = 0.0
        named_any = False
        for label, names in COMPOSITION_BUCKETS:
            val, _ = _latest_prior(names)
            if val is not None and val > 0:
                composition["labels"].append(label)
                composition["values"].append(abs(val))
                named_sum += abs(val)
                named_any = True
        if named_any:
            other = total_assets_latest - named_sum
            composition["labels"].append("Other Assets")
            composition["values"].append(max(other, 0.0))

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
        "composition": composition,
        "summary": _summary_lines(kpis, trend, display_periods),
    }


def _summary_lines(kpis, trend, periods) -> list:
    """Short insights for the page footer/insight strip."""
    by_key = {k["key"]: k for k in kpis}
    lines = []

    def _cr(v):
        return f"₹{abs(v) / 1e7:,.0f} Cr"

    assets = by_key.get("assets")
    if assets and assets.get("yoy_pct") is not None:
        direction = "grew" if assets["yoy_pct"] >= 0 else "shrank"
        lines.append(
            f"Total assets {direction} {abs(assets['yoy_pct']):.1f}% to "
            f"{_cr(assets['latest'])} in {assets['period']}."
        )
    elif assets:
        lines.append(f"Total assets {_cr(assets['latest'])} as of {assets['period']}.")

    equity = by_key.get("equity")
    if equity and assets and assets.get("latest") and equity.get("latest"):
        eq_pct = equity["latest"] / assets["latest"] * 100
        if eq_pct >= 50:
            strength = "strong"
        elif eq_pct >= 25:
            strength = "moderate"
        else:
            strength = "thin"
        lines.append(
            f"Equity funds {eq_pct:.1f}% of the balance sheet — a {strength} capital base."
        )

    de = by_key.get("debt_equity")
    if de:
        lines.append(
            f"Debt-to-equity at {de['ratio']:.2f}x ({de['health'].lower()})."
        )

    cr = by_key.get("current_ratio")
    if cr:
        lines.append(f"Current ratio {cr['ratio']:.2f}x ({cr['health'].lower()}).")

    if periods:
        lines.append(f"Data as of {periods[-1]}")
    return lines
