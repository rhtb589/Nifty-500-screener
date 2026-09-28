import numpy as np
import plotly.graph_objects as go
from langchain_core.tools import tool
from .base import (
    get_row, safe_float, fig_to_html, metric_row, make_result, append_as_of,
    get_stock_snapshot, check_row, verdict_from_checks,
)


@tool
def breakout_analysis(ticker: str) -> dict:
    """Support / Resistance & Breakout (absolute, own price history).
    Bollinger: BBU = SMA20 + 2*sigma, BBL = SMA20 - 2*sigma, %B = (Close - BBL)/(BBU - BBL).
    Donchian: DCU_20 = max High over last 20 bars.
    Breakout = (Close >= DCU_20) and (%B > 1.0)."""
    row = get_row(ticker)
    snap = get_stock_snapshot(ticker) or {}

    csv_price = safe_float(row.get("currentPrice"))
    close = safe_float(snap.get("Close"), csv_price)
    bbu = safe_float(snap.get("BBU_20_2.0_2.0"))
    bbl = safe_float(snap.get("BBL_20_2.0_2.0"))
    pct_b = safe_float(snap.get("BBP_20_2.0_2.0"))
    dcu = safe_float(snap.get("DCU_20_20"))
    dcl = safe_float(snap.get("DCL_20_20"))

    if not np.isfinite(pct_b) and np.isfinite(close) and np.isfinite(bbu) and np.isfinite(bbl) and bbu != bbl:
        pct_b = (close - bbl) / (bbu - bbl)

    high_52w = safe_float(row.get("fiftyTwoWeekHigh"))
    low_52w = safe_float(row.get("fiftyTwoWeekLow"))
    chg_from_high = safe_float(row.get("fiftyTwoWeekHighChangePercent"))
    chg_from_high = chg_from_high * 100 if np.isfinite(chg_from_high) else np.nan
    ath = safe_float(row.get("allTimeHigh"))

    drawdown_from_ath = np.nan
    if np.isfinite(close) and np.isfinite(ath) and ath > 0:
        drawdown_from_ath = (close - ath) / ath * 100

    position_in_range = np.nan
    if (np.isfinite(close) and np.isfinite(high_52w) and np.isfinite(low_52w)
            and high_52w != low_52w):
        position_in_range = (close - low_52w) / (high_52w - low_52w) * 100

    dist_to_dcu = np.nan
    if np.isfinite(close) and np.isfinite(dcu) and dcu != 0:
        dist_to_dcu = (close - dcu) / dcu * 100

    c_donchian = None if not (np.isfinite(close) and np.isfinite(dcu)) else (close >= dcu)
    c_pctb = None if not np.isfinite(pct_b) else (pct_b > 1.0)
    verdict = verdict_from_checks(
        [c_donchian, c_pctb],
        pass_text="BREAKOUT",
        fail_text="NO BREAKOUT",
    )

    metrics = [
        metric_row("Close", close, "price"),
        metric_row("Bollinger Upper (20,2)", bbu, "price"),
        metric_row("Bollinger Lower (20,2)", bbl, "price"),
        metric_row("%B", pct_b),
        metric_row("Donchian Upper (20)", dcu, "price"),
        metric_row("Donchian Lower (20)", dcl, "price"),
        metric_row("Distance to Donchian High", dist_to_dcu, "pct"),
        metric_row("52-Week High", high_52w, "price"),
        metric_row("52-Week Low", low_52w, "price"),
        metric_row("From 52W High", chg_from_high, "pct"),
        metric_row("Position in 52W Range", position_in_range, "pct"),
        metric_row("All-Time High", ath, "price"),
        metric_row("Drawdown from ATH", drawdown_from_ath, "pct"),
        check_row("Check: Close >= DCU_20", c_donchian),
        check_row("Check: %B > 1.0", c_pctb),
        metric_row("Breakout Signal", verdict),
    ]

    labels = ["52W Low", "Bollinger Low", "Close", "Bollinger High", "Donchian High", "52W High"]
    vals = [low_52w, bbl, close, bbu, dcu, high_52w]
    colors = ["#ff6b6b", "#ff9800", "#ffd93d", "#6bcb77", "#00d2ff", "#7a5cff"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=[v if np.isfinite(v) else None for v in vals],
        marker_color=colors,
        text=[f"\u20B9{v:,.2f}" if np.isfinite(v) else "N/A" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(title=f"{ticker} — Support / Resistance Levels", yaxis_title="Price (\u20B9)")

    summary = [f"Verdict: {verdict}"]
    if np.isfinite(pct_b):
        if pct_b > 1.0:
            summary.append(f"%B of {pct_b:.2f} — close is above the upper Bollinger Band (overbought/breakout zone).")
        elif pct_b < 0.0:
            summary.append(f"%B of {pct_b:.2f} — close is below the lower Bollinger Band (breakdown zone).")
        elif pct_b > 0.8:
            summary.append(f"%B of {pct_b:.2f} — pressing the upper band, watch for breakout.")
        elif pct_b < 0.2:
            summary.append(f"%B of {pct_b:.2f} — near the lower band, support test zone.")
        else:
            summary.append(f"%B of {pct_b:.2f} — mid-range within the Bollinger Bands.")
    else:
        summary.append("%B unavailable — no Bollinger band data for this ticker.")
    if np.isfinite(dist_to_dcu):
        if dist_to_dcu >= 0:
            summary.append(f"Close is at/above the 20-bar Donchian high ({dist_to_dcu:+.1f}%).")
        else:
            summary.append(f"Close sits {abs(dist_to_dcu):.1f}% below the 20-bar Donchian high.")
    if np.isfinite(chg_from_high) and -5 <= chg_from_high <= 0:
        summary.append(f"High-Tight Flag: only {abs(chg_from_high):.1f}% below 52W high — consolidation near breakout.")
    if np.isfinite(chg_from_high) and chg_from_high < -30:
        summary.append(f"Deep drawdown of {abs(chg_from_high):.1f}% from 52W high — heavy overhead resistance.")
    if np.isfinite(position_in_range):
        if position_in_range > 80:
            summary.append(f"Trading in top {100 - position_in_range:.0f}% of 52W range — near resistance.")
        elif position_in_range < 20:
            summary.append(f"Trading in bottom {position_in_range:.0f}% of 52W range — near support.")

    return make_result(metrics, fig_to_html(fig), append_as_of(summary, row))
