import numpy as np
import plotly.graph_objects as go
from langchain_core.tools import tool
from .base import (
    get_row, safe_float, fig_to_html, metric_row, make_result, append_as_of,
    check_row, verdict_from_checks,
)


@tool
def growth_analysis(ticker: str) -> dict:
    """GARP & PEG Modeling (absolute, per-stock).
    PEG = trailingPE / (earningsGrowth x 100).
    GARP Candidate = (0 < PEG <= 1.2) and (earningsGrowth > 0.12) and (revenueGrowth > 0.08)."""
    row = get_row(ticker)
    pe = safe_float(row.get("trailingPE"))
    eg = safe_float(row.get("earningsGrowth"))       # decimal
    rg = safe_float(row.get("revenueGrowth"))        # decimal
    qg = safe_float(row.get("earningsQuarterlyGrowth"))
    peg_yahoo = safe_float(row.get("pegRatio"))
    peg_trailing = safe_float(row.get("trailingPegRatio"))

    eg_pct = eg * 100 if np.isfinite(eg) else np.nan
    rg_pct = rg * 100 if np.isfinite(rg) else np.nan
    qg_pct = qg * 100 if np.isfinite(qg) else np.nan

    peg = np.nan
    if np.isfinite(pe) and pe > 0 and np.isfinite(eg) and eg > 0:
        peg = pe / (eg * 100)

    # Condition tri-state: True / False / None (missing data)
    c_peg = None if not np.isfinite(peg) else (0 < peg <= 1.2)
    c_eg = None if not np.isfinite(eg) else (eg > 0.12)
    c_rg = None if not np.isfinite(rg) else (rg > 0.08)
    verdict = verdict_from_checks(
        [c_peg, c_eg, c_rg],
        pass_text="YES — GARP Candidate",
        fail_text="NO — fails GARP rule",
    )

    metrics = [
        metric_row("Trailing P/E", pe, "ratio"),
        metric_row("PEG (computed)", peg),
        metric_row("PEG (Yahoo)", peg_yahoo),
        metric_row("Trailing PEG (Yahoo)", peg_trailing),
        metric_row("Earnings Growth", eg_pct, "pct"),
        metric_row("Revenue Growth", rg_pct, "pct"),
        metric_row("Quarterly Growth", qg_pct, "pct"),
        check_row("Check: 0 < PEG <= 1.2", c_peg),
        check_row("Check: Earnings Growth > 12%", c_eg),
        check_row("Check: Revenue Growth > 8%", c_rg),
        metric_row("GARP Candidate", verdict),
    ]

    labels = ["PEG", "Earnings Growth (%)", "Revenue Growth (%)"]
    actual = [peg, eg_pct, rg_pct]
    thresholds = [1.2, 12.0, 8.0]
    colors = []
    texts = []
    checks = [c_peg, c_eg, c_rg]
    for v, ok in zip(actual, checks):
        colors.append("#6bcb77" if ok is True else "#ff6b6b" if ok is False else "#7a5cff")
        texts.append(f"{v:.2f}" if np.isfinite(v) else "N/A")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Actual", x=labels,
                         y=[v if np.isfinite(v) else None for v in actual],
                         marker_color=colors, text=texts, textposition="outside"))
    fig.add_trace(go.Bar(name="Threshold", x=labels, y=thresholds,
                         marker_color="#ffd93d", opacity=0.45,
                         text=[f"{t:.2f}" for t in thresholds], textposition="outside"))
    fig.update_layout(barmode="group", title=f"{ticker} — GARP Inputs vs Thresholds",
                      yaxis_title="Value")

    summary = [f"Verdict: {verdict}"]
    if np.isfinite(peg):
        if peg <= 1.2:
            summary.append(f"PEG of {peg:.2f} <= 1.2 — price is justified by growth.")
        else:
            summary.append(f"PEG of {peg:.2f} > 1.2 — paying too much for the growth rate.")
    else:
        summary.append("PEG not computable — need positive P/E and positive earnings growth.")
    if np.isfinite(eg):
        summary.append(
            f"Earnings growth {eg_pct:.1f}% {'>' if eg > 0.12 else '<='} 12% hurdle."
        )
    if np.isfinite(rg):
        summary.append(
            f"Revenue growth {rg_pct:.1f}% {'>' if rg > 0.08 else '<='} 8% hurdle."
        )
    if np.isfinite(eg) and np.isfinite(rg):
        if eg > rg + 0.05:
            summary.append("Earnings growing faster than revenue — operating leverage expanding.")
        elif rg > eg + 0.05:
            summary.append("Revenue outpacing earnings — margin compression or rising costs.")

    return make_result(metrics, fig_to_html(fig), append_as_of(summary, row))
