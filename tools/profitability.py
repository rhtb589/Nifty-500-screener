import numpy as np
import plotly.graph_objects as go
from langchain_core.tools import tool
from .base import (
    get_row, safe_float, fig_to_html, metric_row, make_result, append_as_of,
    check_row, verdict_from_checks,
)


@tool
def profitability_analysis(ticker: str) -> dict:
    """Profitability & DuPont 3-Step Decomposition (absolute, per-stock).
    ROE = Net Profit Margin x Asset Turnover x Equity Multiplier,
    where Asset Turnover = ROA / Net Margin and Equity Multiplier = ROE / ROA.
    High Quality = (ROE > 0.15) and (profitMargins > 0.10) and (ROA > 0.07)."""
    row = get_row(ticker)
    roe = safe_float(row.get("returnOnEquity"))       # decimal
    roa = safe_float(row.get("returnOnAssets"))       # decimal
    margin = safe_float(row.get("profitMargins"))     # decimal
    op_margin = safe_float(row.get("operatingMargins"))
    gross_margin = safe_float(row.get("grossMargins"))
    ebitda_margin = safe_float(row.get("ebitdaMargins"))
    rev_per_share = safe_float(row.get("revenuePerShare"))

    roe_pct = roe * 100 if np.isfinite(roe) else np.nan
    roa_pct = roa * 100 if np.isfinite(roa) else np.nan
    margin_pct = margin * 100 if np.isfinite(margin) else np.nan
    op_pct = op_margin * 100 if np.isfinite(op_margin) else np.nan
    gross_pct = gross_margin * 100 if np.isfinite(gross_margin) else np.nan
    ebitda_pct = ebitda_margin * 100 if np.isfinite(ebitda_margin) else np.nan

    # DuPont components (derived — totalAssets column does not exist in the dataset)
    asset_turnover = np.nan
    if np.isfinite(roa) and np.isfinite(margin) and margin != 0:
        asset_turnover = roa / margin
    equity_multiplier = np.nan
    if np.isfinite(roe) and np.isfinite(roa) and roa != 0:
        equity_multiplier = roe / roa
    dupont_roe = np.nan
    if np.isfinite(margin) and np.isfinite(asset_turnover) and np.isfinite(equity_multiplier):
        dupont_roe = margin * asset_turnover * equity_multiplier

    # High-quality filter (tri-state)
    c_roe = None if not np.isfinite(roe) else (roe > 0.15)
    c_pm = None if not np.isfinite(margin) else (margin > 0.10)
    c_roa = None if not np.isfinite(roa) else (roa > 0.07)
    verdict = verdict_from_checks(
        [c_roe, c_pm, c_roa],
        pass_text="HIGH QUALITY",
        fail_text="Below quality bar",
    )

    metrics = [
        metric_row("ROE", roe_pct, "pct"),
        metric_row("ROA", roa_pct, "pct"),
        metric_row("Net Profit Margin", margin_pct, "pct"),
        metric_row("Gross Margin", gross_pct, "pct"),
        metric_row("Operating Margin", op_pct, "pct"),
        metric_row("EBITDA Margin", ebitda_pct, "pct"),
        metric_row("Asset Turnover", asset_turnover),
        metric_row("Equity Multiplier", equity_multiplier),
        metric_row("DuPont ROE (check)", dupont_roe * 100 if np.isfinite(dupont_roe) else np.nan, "pct"),
        metric_row("Revenue/Share", rev_per_share, "price"),
        check_row("Check: ROE > 15%", c_roe),
        check_row("Check: Net Margin > 10%", c_pm),
        check_row("Check: ROA > 7%", c_roa),
        metric_row("Quality Filter", verdict),
    ]

    cats = ["ROE", "ROA", "Net Margin", "Operating Margin", "Gross Margin"]
    vals = [roe_pct, roa_pct, margin_pct, op_pct, gross_pct]
    colors = ["#6bcb77", "#00d2ff", "#ffd93d", "#7a5cff", "#ff6b6b"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cats, y=[v if np.isfinite(v) else None for v in vals],
        marker_color=colors,
        text=[f"{v:.1f}%" if np.isfinite(v) else "N/A" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(title=f"{ticker} — Returns & Margins", yaxis_title="%")

    summary = [f"Verdict: {verdict}"]
    if np.isfinite(dupont_roe) and np.isfinite(asset_turnover) and np.isfinite(equity_multiplier):
        summary.append(
            f"DuPont: ROE {roe_pct:.1f}% = Net Margin {margin_pct:.1f}% x "
            f"Asset Turnover {asset_turnover:.2f} x Equity Multiplier {equity_multiplier:.2f}."
        )
        if equity_multiplier > 2.0:
            summary.append(f"Equity multiplier of {equity_multiplier:.2f} — returns are meaningfully leverage-driven.")
        elif equity_multiplier > 0:
            summary.append(f"Equity multiplier of {equity_multiplier:.2f} — modest leverage, quality driven by operations.")
    else:
        summary.append("DuPont decomposition unavailable — ROE/ROA/margin data missing for this ticker.")
    if np.isfinite(gross_pct) and np.isfinite(op_pct):
        spread = gross_pct - op_pct
        summary.append(f"Gross-to-Operating margin spread: {spread:.1f}pp — {'tight' if spread < 10 else 'wide'} overhead structure.")
    if np.isfinite(margin) and margin < 0.05:
        summary.append(f"Thin net margin ({margin_pct:.1f}%) — high sensitivity to cost shocks.")

    return make_result(metrics, fig_to_html(fig), append_as_of(summary, row))
