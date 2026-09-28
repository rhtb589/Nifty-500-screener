import numpy as np
import plotly.graph_objects as go
from langchain_core.tools import tool
from .base import (
    get_row, safe_float, fig_to_html, metric_row, make_result, append_as_of,
    check_row, verdict_from_checks,
)


@tool
def sentiment_analysis(ticker: str) -> dict:
    """Analyst Consensus & Sentiment (absolute, per-stock).
    Analyst Upside % = (targetMeanPrice - currentPrice) / currentPrice * 100.
    High Conviction Buy = (recommendationMean <= 2.2) and (numberOfAnalystOpinions >= 3)
        and (Analyst Upside >= 15%). Rating scale: 1.0 Strong Buy ... 5.0 Strong Sell."""
    row = get_row(ticker)
    price = safe_float(row.get("currentPrice"))
    target_mean = safe_float(row.get("targetMeanPrice"))
    target_high = safe_float(row.get("targetHighPrice"))
    target_low = safe_float(row.get("targetLowPrice"))
    target_median = safe_float(row.get("targetMedianPrice"))
    rec_mean = safe_float(row.get("recommendationMean"))
    rec_key = row.get("recommendationKey", "")
    num_analysts = safe_float(row.get("numberOfAnalystOpinions"), 0)

    upside = np.nan
    if np.isfinite(target_mean) and np.isfinite(price) and price > 0:
        upside = (target_mean - price) / price * 100
    upside_median = np.nan
    if np.isfinite(target_median) and np.isfinite(price) and price > 0:
        upside_median = (target_median - price) / price * 100

    rec_labels = {1: "Strong Buy", 2: "Buy", 3: "Hold", 4: "Sell", 5: "Strong Sell"}
    rec_display = rec_labels.get(round(rec_mean), rec_key) if np.isfinite(rec_mean) else rec_key

    c_rec = None if not np.isfinite(rec_mean) else (rec_mean <= 2.2)
    c_n = None if not num_analysts else (num_analysts >= 3)
    c_up = None if not np.isfinite(upside) else (upside >= 15.0)
    verdict = verdict_from_checks(
        [c_rec, c_n, c_up],
        pass_text="HIGH CONVICTION BUY",
        fail_text="NOT HIGH CONVICTION",
    )

    metrics = [
        metric_row("Current Price", price, "price"),
        metric_row("Mean Target", target_mean, "price"),
        metric_row("Analyst Upside", upside, "pct"),
        metric_row("Median Target", target_median, "price"),
        metric_row("Upside (Median)", upside_median, "pct"),
        metric_row("High Target", target_high, "price"),
        metric_row("Low Target", target_low, "price"),
        metric_row("Recommendation (1-5)", rec_mean),
        metric_row("Recommendation", rec_display),
        metric_row("Analyst Count", int(num_analysts)),
        check_row("Check: Recommendation <= 2.2", c_rec),
        check_row("Check: Analysts >= 3", c_n),
        check_row("Check: Upside >= 15%", c_up),
        metric_row("Sentiment Filter", verdict),
    ]

    targets = [target_low, price, target_mean, target_high]
    labels = ["Low Target", "Current", "Mean Target", "High Target"]
    colors = ["#ff6b6b", "#ffd93d", "#00d2ff", "#6bcb77"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=[t if np.isfinite(t) else None for t in targets],
        marker_color=colors,
        text=[f"\u20B9{t:,.0f}" if np.isfinite(t) else "N/A" for t in targets],
        textposition="outside",
    ))
    fig.update_layout(title=f"{ticker} — Analyst Price Targets", yaxis_title="Price (\u20B9)")

    summary = [f"Verdict: {verdict}"]
    if np.isfinite(upside):
        if upside >= 15:
            summary.append(f"Mean target implies {upside:+.1f}% upside — clears the 15% conviction hurdle.")
        elif upside < 0:
            summary.append(f"Mean target implies {upside:+.1f}% — consensus sees downside from here.")
        else:
            summary.append(f"Mean target implies {upside:+.1f}% upside — below the 15% hurdle.")
    else:
        summary.append("No mean analyst target available for this ticker.")
    if np.isfinite(rec_mean):
        summary.append(
            f"Average recommendation {rec_mean:.2f} (1=Strong Buy ... 5=Strong Sell) "
            f"— {'Buy-side consensus' if rec_mean <= 2.2 else 'Hold-side or weaker'}."
        )
    if num_analysts:
        summary.append(f"Based on {int(num_analysts)} analyst opinions.")
    else:
        summary.append("No analyst coverage available.")
    if np.isfinite(target_high) and np.isfinite(target_low) and np.isfinite(price) and price > 0:
        spread = (target_high - target_low) / price * 100
        if spread > 50:
            summary.append(f"Wide target range ({spread:.0f}% of price) — high uncertainty among analysts.")

    return make_result(metrics, fig_to_html(fig), append_as_of(summary, row))
