import numpy as np
import plotly.graph_objects as go
from langchain_core.tools import tool
from .base import (
    get_row, safe_float, fig_to_html, metric_row, make_result, append_as_of,
    get_stock_snapshot, check_row, verdict_from_checks,
)


@tool
def trend_analysis(ticker: str) -> dict:
    """Trend & Moving Averages (absolute, own price history).
    Stage 2 Bull = (Close > EMA_20) and (EMA_20 > SMA_50) and (SMA_50 > SMA_200) and (SLOPE_20 > 0),
    where SLOPE_20 is the 20-bar linear regression slope of Close."""
    row = get_row(ticker)
    snap = get_stock_snapshot(ticker) or {}

    csv_price = safe_float(row.get("currentPrice"))
    close = safe_float(snap.get("Close"), csv_price)

    ema20 = safe_float(snap.get("EMA_20"))
    slope20 = safe_float(snap.get("SLOPE_20"))
    sma50 = safe_float(snap.get("SMA_50"))
    if not np.isfinite(sma50):
        sma50 = safe_float(row.get("SMA_50"))
    if not np.isfinite(sma50):
        sma50 = safe_float(row.get("fiftyDayAverage"))
    sma200 = safe_float(snap.get("SMA_200"))
    if not np.isfinite(sma200):
        sma200 = safe_float(row.get("SMA_200"))
    if not np.isfinite(sma200):
        sma200 = safe_float(row.get("twoHundredDayAverage"))

    def _dist(val):
        if np.isfinite(close) and np.isfinite(val) and val != 0:
            return (close - val) / val * 100
        return np.nan

    dist_ema = _dist(ema20)
    dist_50 = _dist(sma50)
    dist_200 = _dist(sma200)

    c_above_ema = None if not (np.isfinite(close) and np.isfinite(ema20)) else (close > ema20)
    c_ema_sma = None if not (np.isfinite(ema20) and np.isfinite(sma50)) else (ema20 > sma50)
    c_sma_stack = None if not (np.isfinite(sma50) and np.isfinite(sma200)) else (sma50 > sma200)
    c_slope = None if not np.isfinite(slope20) else (slope20 > 0)
    verdict = verdict_from_checks(
        [c_above_ema, c_ema_sma, c_sma_stack, c_slope],
        pass_text="STAGE 2 UPTREND",
        fail_text="NOT STAGE 2",
    )

    metrics = [
        metric_row("Close", close, "price"),
        metric_row("EMA 20", ema20, "price"),
        metric_row("SMA 50", sma50, "price"),
        metric_row("SMA 200", sma200, "price"),
        metric_row("Linear Reg Slope (20)", slope20),
        metric_row("Distance to EMA 20", dist_ema, "pct"),
        metric_row("Distance to SMA 50", dist_50, "pct"),
        metric_row("Distance to SMA 200", dist_200, "pct"),
        check_row("Check: Close > EMA 20", c_above_ema),
        check_row("Check: EMA 20 > SMA 50", c_ema_sma),
        check_row("Check: SMA 50 > SMA 200", c_sma_stack),
        check_row("Check: Slope(20) > 0", c_slope),
        metric_row("Trend Regime", verdict),
    ]

    labels = ["Close", "EMA 20", "SMA 50", "SMA 200"]
    vals = [close, ema20, sma50, sma200]
    colors = ["#ffd93d", "#00d2ff", "#6bcb77", "#7a5cff"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=[v if np.isfinite(v) else None for v in vals],
        marker_color=colors,
        text=[f"\u20B9{v:,.2f}" if np.isfinite(v) else "N/A" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(title=f"{ticker} — Price vs Own Moving Averages",
                      yaxis_title="Price (\u20B9)")

    summary = [f"Verdict: {verdict}"]
    if verdict == "STAGE 2 UPTREND":
        summary.append("Stage 2 bull confirmed: Close > EMA20 > SMA50 > SMA200 with positive 20-bar slope.")
    else:
        failed = []
        if c_above_ema is False:
            failed.append("Close below EMA 20")
        if c_ema_sma is False:
            failed.append("EMA 20 below SMA 50")
        if c_sma_stack is False:
            failed.append("SMA 50 below SMA 200 (Death Cross)")
        if c_slope is False:
            failed.append("negative 20-bar slope")
        if failed:
            summary.append("Stage 2 criteria not met — " + "; ".join(failed) + ".")
        else:
            summary.append("Some trend inputs unavailable — regime could not be fully verified.")
    if np.isfinite(sma50) and np.isfinite(sma200):
        if sma50 > sma200:
            summary.append("Golden Cross in effect (SMA 50 above SMA 200) — bullish long-term structure.")
        else:
            summary.append("Death Cross in effect (SMA 50 below SMA 200) — bearish long-term structure.")
    if np.isfinite(slope20):
        summary.append(
            f"20-bar linear regression slope is {slope20:+.4f} — "
            f"{'rising' if slope20 > 0 else 'falling'} near-term trend."
        )
    if np.isfinite(dist_200):
        summary.append(f"Price is {dist_200:+.1f}% vs its 200-day average.")

    return make_result(metrics, fig_to_html(fig), append_as_of(summary, row))
