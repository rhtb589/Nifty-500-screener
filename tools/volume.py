import numpy as np
import plotly.graph_objects as go
from langchain_core.tools import tool
from .base import (
    get_row, safe_float, fig_to_html, metric_row, make_result, append_as_of,
    get_stock_snapshot, get_universe_tech, percentile_rank,
    check_row, verdict_from_checks,
)


@tool
def volume_analysis(ticker: str) -> dict:
    """Volume Anomaly & Accumulation (hybrid — own history + universe cross-section).
    RVOL_10 = Volume / mean(Volume, last 10 bars); CMF_20 = Chaikin Money Flow.
    P_CMF = percentile rank of this stock's CMF_20 across all 500 stocks.
    Institutional Accumulation = (RVOL_10 >= 1.8) and (CMF_20 > 0.08)
        and (Close > VWAP) and (P_CMF >= 0.70)."""
    row = get_row(ticker)
    snap = get_stock_snapshot(ticker) or {}

    csv_price = safe_float(row.get("currentPrice"))
    close = safe_float(snap.get("Close"), csv_price)
    vwap = safe_float(snap.get("VWAP"))
    rvol10 = safe_float(snap.get("RVOL_10"))
    cmf = safe_float(snap.get("CMF_20"))
    mfi = safe_float(snap.get("MFI_14"))

    universe_cmf = None
    p_cmf = np.nan
    try:
        universe_cmf = get_universe_tech()["CMF_20"]
        p_cmf = percentile_rank(cmf, universe_cmf)
    except Exception:
        pass

    close_above_vwap = None
    if np.isfinite(close) and np.isfinite(vwap):
        close_above_vwap = close > vwap

    c_rvol = None if not np.isfinite(rvol10) else (rvol10 >= 1.8)
    c_cmf = None if not np.isfinite(cmf) else (cmf > 0.08)
    c_vwap = close_above_vwap
    c_p = None if not np.isfinite(p_cmf) else (p_cmf >= 0.70)
    verdict = verdict_from_checks(
        [c_rvol, c_cmf, c_vwap, c_p],
        pass_text="INSTITUTIONAL ACCUMULATION",
        fail_text="NO ACCUMULATION SIGNAL",
    )

    n_universe = int(universe_cmf.dropna().shape[0]) if universe_cmf is not None else 0

    metrics = [
        metric_row("Close", close, "price"),
        metric_row("VWAP", vwap, "price"),
        metric_row("RVOL (10-day)", rvol10),
        metric_row("CMF (20)", cmf),
        metric_row("MFI (14)", mfi),
        metric_row("CMF Percentile (Nifty 500)", p_cmf * 100 if np.isfinite(p_cmf) else np.nan, "pct"),
        metric_row("Universe Size", n_universe),
        check_row("Check: RVOL_10 >= 1.8", c_rvol),
        check_row("Check: CMF_20 > 0.08", c_cmf),
        check_row("Check: Close > VWAP", c_vwap),
        check_row("Check: CMF Pct >= 70%", c_p),
        metric_row("Accumulation Flag", verdict),
    ]

    labels = ["RVOL >= 1.8", "CMF > 0.08", "Close > VWAP", "CMF Pct >= 70%"]
    checks = [c_rvol, c_cmf, c_vwap, c_p]
    actuals = [
        f"{rvol10:.2f}x" if np.isfinite(rvol10) else "N/A",
        f"{cmf:+.3f}" if np.isfinite(cmf) else "N/A",
        (f"{close:,.0f} vs {vwap:,.0f}" if np.isfinite(close) and np.isfinite(vwap) else "N/A"),
        (f"{p_cmf * 100:.0f}%" if np.isfinite(p_cmf) else "N/A"),
    ]
    values = [1.0 if c is True else 0.0 if c is False else np.nan for c in checks]
    colors = ["#6bcb77" if c is True else "#ff6b6b" if c is False else "#7a5cff" for c in checks]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=[v if np.isfinite(v) else None for v in values],
        marker_color=colors,
        text=[f"{a} — {'PASS' if c is True else 'FAIL' if c is False else 'N-A'}"
              for a, c in zip(actuals, checks)],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{ticker} — Accumulation Conditions (1 = met)",
        yaxis=dict(range=[0, 1.35], showticklabels=False),
        showlegend=False,
    )

    summary = [f"Verdict: {verdict}"]
    if np.isfinite(rvol10):
        if rvol10 >= 1.8:
            summary.append(f"RVOL of {rvol10:.2f}x >= 1.8 — volume surge above the 10-day norm.")
        else:
            summary.append(f"RVOL of {rvol10:.2f}x < 1.8 — no significant volume expansion.")
    else:
        summary.append("RVOL_10 unavailable — no 10-day volume history for this ticker.")
    if np.isfinite(cmf):
        summary.append(
            f"CMF_20 of {cmf:+.3f} {'>' if cmf > 0.08 else '<='} 0.08 — "
            f"{'buying pressure' if cmf > 0.08 else 'money flow not constructive'}."
        )
    if np.isfinite(p_cmf):
        summary.append(f"CMF ranks in the {p_cmf * 100:.0f}th percentile of the Nifty 500 universe.")
    if np.isfinite(close) and np.isfinite(vwap):
        summary.append(
            f"Close {'above' if close > vwap else 'below'} VWAP — "
            f"{'buyers in control intraday' if close > vwap else 'weakness vs volume-weighted average price'}."
        )
    if np.isfinite(mfi):
        if mfi > 50:
            summary.append(f"MFI of {mfi:.1f} — positive money flow momentum.")
        elif mfi < 30:
            summary.append(f"MFI of {mfi:.1f} — oversold money flow, potential reversal watch.")
        else:
            summary.append(f"MFI of {mfi:.1f} — neutral money flow.")

    return make_result(metrics, fig_to_html(fig), append_as_of(summary, row))
