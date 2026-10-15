"""Income Statement payload for the standalone /income_statement/<ticker> page.

Reads data/income_statement/{TICKER}.csv (annual statement, newest period
first, written by refresh_financials.py) and turns it into a JSON-ready
payload: KPI cards (Total Revenue, Gross Margin, EBITDA Margin, Net Profit,
Diluted EPS), a sectioned statement table (curated rows like the reference
dashboard), and series for the profitability-trend chart + latest-FY expense
allocation doughnut.

Sections use the subtotal rows yfinance already stores (Total Revenue,
Gross Profit, Operating Expense, EBITDA, Pretax Income, Net Income), so no
totals are recomputed.

Rows carry a `unit` hint consumed by the frontend:
  money (default) — scaled by the ₹ Crores / ₹ Millions toggle
  eps             — ₹ per share, two decimals, never scaled
  count           — absolute share count, never scaled
"""

from langchain_core.tools import tool

import pandas as pd

from .financials import load_statement
from .base import resolve_ticker, get_row

# Curated display rows, in rendering order. "names" are tried in order until
# one exists in the ticker's income statement CSV; "label" is the friendly text.
# style: normal | subtotal | grand — indent: child of a parent row
REVENUE_ROWS = [
    {"names": ("Operating Revenue",), "label": "Operating Revenue"},
    {"names": ("Total Revenue",), "label": "Total Revenue", "style": "subtotal"},
]

COST_ROWS = [
    {"names": ("Cost Of Revenue", "Reconciled Cost Of Revenue"),
     "label": "Cost Of Revenue (Raw Materials & Spares)"},
    {"names": ("Gross Profit",), "label": "Gross Profit", "style": "subtotal"},
]

OPEX_ROWS = [
    {"names": ("Selling General And Administration",),
     "label": "Selling, General & Administrative Expense"},
    {"names": ("Other Operating Expenses",), "label": "Other Operating Expenses"},
    {"names": ("Depreciation And Amortization In Income Statement",
               "Depreciation Income Statement", "Reconciled Depreciation"),
     "label": "Depreciation & Amortization"},
    {"names": ("Operating Expense",), "label": "Total Operating Expenses",
     "style": "subtotal"},
]

EBITDA_ROWS = [
    {"names": ("Operating Income",), "label": "Operating Income (EBIT)",
     "style": "subtotal"},
    {"names": ("EBITDA", "Normalized EBITDA"), "label": "EBITDA", "style": "subtotal"},
]

OTHER_ROWS = [
    {"names": ("Interest Income", "Interest Income Non Operating"),
     "label": "Interest Income"},
    {"names": ("Interest Expense", "Interest Expense Non Operating"),
     "label": "Interest Expense"},
    {"names": ("Net Interest Income",), "label": "Net Interest Income"},
    {"names": ("Other Non Operating Income Expenses",),
     "label": "Other Non-Operating Income / (Expenses)"},
    {"names": ("Special Income Charges",), "label": "Special Income / (Charges)"},
    {"names": ("Other Special Charges",), "label": "Other Special / Unusual Charges"},
]

EARNINGS_ROWS = [
    {"names": ("Pretax Income",), "label": "Pretax Income (PBT)", "style": "subtotal"},
    {"names": ("Tax Provision",), "label": "Tax Provision"},
    {"names": ("Net Income", "Net Income Common Stockholders",
               "Net Income Continuous Operations"),
     "label": "NET INCOME (PAT)", "style": "grand"},
]

PER_SHARE_ROWS = [
    {"names": ("Basic EPS",), "label": "Basic EPS", "unit": "eps"},
    {"names": ("Diluted EPS",), "label": "Diluted EPS", "unit": "eps"},
    {"names": ("Basic Average Shares",), "label": "Basic Average Shares", "unit": "count"},
    {"names": ("Diluted Average Shares",), "label": "Diluted Average Shares",
     "unit": "count"},
]

SECTIONS = [
    ("revenue", "1. Revenue from Operations", REVENUE_ROWS),
    ("cost", "2. Cost of Revenue & Gross Profit", COST_ROWS),
    ("opex", "3. Operating Expenses", OPEX_ROWS),
    ("ebitda", "4. Operating Results & EBITDA", EBITDA_ROWS),
    ("other", "5. Other Income / (Finance Costs)", OTHER_ROWS),
    ("earnings", "6. Earnings & Taxes", EARNINGS_ROWS),
    ("per_share", "7. Per Share Metrics", PER_SHARE_ROWS),
]

PERIODS_TO_SHOW = 4

# Doughnut chart (latest FY): label -> row names (tried in order)
COMPOSITION_BUCKETS = [
    ("Cost of Revenue", ("Cost Of Revenue", "Reconciled Cost Of Revenue")),
    ("Operating Expenses", ("Operating Expense", "Selling General And Administration")),
    ("Finance Costs", ("Interest Expense", "Interest Expense Non Operating")),
    ("Taxes", ("Tax Provision",)),
    ("Net Profit (PAT)", ("Net Income", "Net Income Common Stockholders")),
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
        return None
    if prior == 0:
        return None
    return latest / abs(prior) - 1.0 if prior > 0 else (latest - prior) / abs(prior)


def _margin_pct(numerator, denominator):
    """Percent margin (None when inputs unusable)."""
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator * 100.0


def _tone_from_delta(delta_pp):
    """good/warn/bad tone for a margin change in percentage points."""
    if delta_pp is None:
        return "flat"
    if delta_pp >= 0:
        return "good"
    if delta_pp >= -1.0:
        return "warn"
    return "bad"

@tool
def get_income_statement(ticker: str) -> dict:
    """Income statement for one ticker (JSON-ready)."""
    return build_income_statement_payload(ticker)

def build_income_statement_payload(ticker: str) -> dict:
    """JSON-ready income statement dashboard payload for one ticker.

    Raises ValueError when the ticker is unknown to the universe or has no
    statement file under data/income_statement/.
    """
    ticker = resolve_ticker(str(ticker).strip())
    row = get_row(ticker)  # raises ValueError for unknown tickers

    inc_df = load_statement(ticker, "income_statement")
    if inc_df is None or not len(inc_df.columns):
        raise ValueError(f"No income statement on disk for '{ticker}'.")

    periods = [str(c) for c in inc_df.columns[:PERIODS_TO_SHOW]]
    # display order: oldest -> newest (latest column highlighted on the page)
    display_periods = list(reversed(periods))

    def _fy(p: str) -> str:
        return f"FY {p[:4]}"

    period_meta = [{"label": _fy(p), "date": p} for p in display_periods]

    def _display(names):
        """Row values in display order (oldest -> newest)."""
        return list(reversed(_row_values(names, inc_df, periods)))

    def _latest_prior(names):
        vals = _display(names)
        present = [v for v in vals if v is not None]
        latest = present[-1] if present else None
        prior = present[-2] if len(present) >= 2 else None
        return latest, prior

    sections = []
    for key, title, specs in SECTIONS:
        rows = []
        for spec in specs:
            values = _display(spec["names"])
            if all(v is None for v in values):
                continue
            rows.append({
                "label": spec["label"],
                "indent": bool(spec.get("indent")),
                "style": spec.get("style", "normal"),
                "unit": spec.get("unit", "money"),
                "values": values,
            })
        if rows:
            sections.append({"key": key, "title": title, "rows": rows})

    if not sections:
        raise ValueError(f"Income statement for '{ticker}' is empty.")

    # ---- KPI cards ----
    kpis = []
    latest_label = display_periods[-1] if display_periods else None
    prior_label = display_periods[-2] if len(display_periods) > 1 else None

    # 1. Total Revenue (YoY %)
    rev_latest, rev_prior = _latest_prior(("Total Revenue", "Operating Revenue"))
    if rev_latest is not None:
        kpis.append({
            "key": "revenue",
            "kind": "money",
            "label": "Total Revenue",
            "latest": rev_latest,
            "prior": rev_prior,
            "yoy_pct": (_yoy(rev_latest, rev_prior) or 0.0) * 100.0
            if _yoy(rev_latest, rev_prior) is not None else None,
            "period": latest_label,
            "prior_period": prior_label,
        })

    gp_latest, gp_prior = _latest_prior(("Gross Profit",))
    # 2. Gross Margin (%)
    gm_latest = _margin_pct(gp_latest, rev_latest)
    gm_prior = _margin_pct(gp_prior, rev_prior)
    if gm_latest is not None:
        delta_pp = (gm_latest - gm_prior) if gm_prior is not None else None
        kpis.append({
            "key": "gross_margin",
            "kind": "percent",
            "label": "Gross Margin",
            "value": round(gm_latest, 2),
            "prior_value": round(gm_prior, 2) if gm_prior is not None else None,
            "health": "Margin expanded" if (delta_pp or 0) >= 0 else "Margin compressed",
            "tone": _tone_from_delta(delta_pp),
            "sub_amount": gp_latest,
            "period": latest_label,
        })

    ebitda_latest, ebitda_prior = _latest_prior(("EBITDA", "Normalized EBITDA"))
    # 3. EBITDA / Operating Margin (%)
    om_latest = _margin_pct(ebitda_latest, rev_latest)
    om_prior = _margin_pct(ebitda_prior, rev_prior)
    if om_latest is not None:
        delta_pp = (om_latest - om_prior) if om_prior is not None else None
        kpis.append({
            "key": "ebitda_margin",
            "kind": "percent",
            "label": "EBITDA Margin",
            "value": round(om_latest, 2),
            "prior_value": round(om_prior, 2) if om_prior is not None else None,
            "health": "Margin expanded" if (delta_pp or 0) >= 0 else "Margin compressed",
            "tone": _tone_from_delta(delta_pp),
            "sub_amount": ebitda_latest,
            "period": latest_label,
        })

    # 4. Net Profit / PAT (YoY %)
    pat_latest, pat_prior = _latest_prior(
        ("Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"))
    if pat_latest is not None:
        kpis.append({
            "key": "pat",
            "kind": "money",
            "label": "Net Profit (PAT)",
            "latest": pat_latest,
            "prior": pat_prior,
            "yoy_pct": (_yoy(pat_latest, pat_prior) or 0.0) * 100.0
            if _yoy(pat_latest, pat_prior) is not None else None,
            "period": latest_label,
            "prior_period": prior_label,
        })

    # 5. Diluted EPS (+ share count sub-line)
    eps_latest, eps_prior = _latest_prior(("Diluted EPS", "Basic EPS"))
    if eps_latest is not None:
        shares_latest, _ = _latest_prior(("Diluted Average Shares", "Basic Average Shares"))
        kpis.append({
            "key": "eps",
            "kind": "eps",
            "label": "Diluted EPS",
            "value": eps_latest,
            "prior": eps_prior,
            "yoy_pct": (_yoy(eps_latest, eps_prior) or 0.0) * 100.0
            if _yoy(eps_latest, eps_prior) is not None else None,
            "shares_crores": shares_latest / 1e7 if shares_latest is not None else None,
            "period": latest_label,
        })

    # ---- Profitability trend (Revenue vs Gross Profit vs EBITDA vs PAT) ----
    def _trend_series(names):
        # absent rows (e.g. Gross Profit at banks) chart as 0 rather than
        # NaN, which would also serialize as invalid strict JSON
        return [0.0 if v is None else v for v in _display(names)]

    trend = {
        "labels": [m["label"] for m in period_meta],
        "revenue": _trend_series(("Total Revenue", "Operating Revenue")),
        "gross_profit": _trend_series(("Gross Profit",)),
        "ebitda": _trend_series(("EBITDA", "Normalized EBITDA")),
        "net_income": _trend_series(
            ("Net Income", "Net Income Common Stockholders",
             "Net Income Continuous Operations")),
    }

    # ---- Expense allocation (latest FY) ----
    composition = {"labels": [], "values": []}
    if rev_latest is not None:
        named_sum = 0.0
        for label, names in COMPOSITION_BUCKETS:
            val, _ = _latest_prior(names)
            if val is not None and val > 0:
                composition["labels"].append(label)
                composition["values"].append(abs(val))
                named_sum += abs(val)
        residual = rev_latest - named_sum
        if composition["values"] and residual > 0:
            composition["labels"].append("Other Income / (Expense)")
            composition["values"].append(residual)

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
        "summary": _summary_lines(kpis, display_periods),
    }


def _summary_lines(kpis, periods) -> list:
    """Short insights for the page footer/insight strip."""
    by_key = {k["key"]: k for k in kpis}
    lines = []

    def _cr(v):
        return f"₹{abs(v) / 1e7:,.0f} Cr"

    rev = by_key.get("revenue")
    if rev:
        if rev.get("yoy_pct") is not None:
            direction = "grew" if rev["yoy_pct"] >= 0 else "declined"
            lines.append(
                f"Revenue {direction} {abs(rev['yoy_pct']):.1f}% to "
                f"{_cr(rev['latest'])} in {rev['period']}."
            )
        else:
            lines.append(f"Revenue {_cr(rev['latest'])} in {rev['period']}.")

    gm = by_key.get("gross_margin")
    if gm and gm.get("prior_value") is not None:
        delta = gm["value"] - gm["prior_value"]
        direction = "widened" if delta >= 0 else "narrowed"
        lines.append(
            f"Gross margin {direction} {abs(delta):.1f} pp to {gm['value']:.1f}% "
            f"({_cr(gm['sub_amount'])} gross profit)."
        )
    elif gm:
        lines.append(f"Gross margin {gm['value']:.1f}% in {gm.get('period') or ''}.")

    om = by_key.get("ebitda_margin")
    if om:
        lines.append(f"EBITDA margin at {om['value']:.1f}%.")

    pat = by_key.get("pat")
    if pat and pat.get("yoy_pct") is not None:
        direction = "grew" if pat["yoy_pct"] >= 0 else "fell"
        lines.append(
            f"Net profit (PAT) {direction} {abs(pat['yoy_pct']):.1f}% to "
            f"{_cr(pat['latest'])}."
        )
    elif pat:
        lines.append(f"Net profit (PAT) {_cr(pat['latest'])} in {pat['period'] or ''}.")

    eps = by_key.get("eps")
    if eps:
        line = f"Diluted EPS at ₹{eps['value']:,.2f}"
        if eps.get("shares_crores"):
            line += f" on {eps['shares_crores']:.2f} Cr diluted shares"
        lines.append(line + ".")

    if periods:
        lines.append(f"Data as of {periods[-1]}")
    return lines
