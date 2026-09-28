import numpy as np
import plotly.graph_objects as go
from langchain_core.tools import tool
from .base import (
    get_row, safe_float, fig_to_html, metric_row, make_result, append_as_of,
    check_row, verdict_from_checks,
)


@tool
def dividend_analysis(ticker: str) -> dict:
    """Dividend Sustainability (absolute, per-stock).
    FCF Coverage = freeCashflow / (dividendRate x sharesOutstanding); OCF used as fallback.
    Safe Dividend = (dividendYield > 1%) and (0 < payoutRatio <= 0.65) and (freeCashflow > 0).
    NOTE: dividendYield and fiveYearAvgDividendYield are stored in percent."""
    row = get_row(ticker)
    div_rate = safe_float(row.get("dividendRate"))                 # INR per share
    div_yield_pct = safe_float(row.get("dividendYield"))           # percent units
    avg_5y_pct = safe_float(row.get("fiveYearAvgDividendYield"))   # percent units
    payout = safe_float(row.get("payoutRatio"))                    # decimal
    trailing_yield = safe_float(row.get("trailingAnnualDividendYield"))  # decimal
    fcf = safe_float(row.get("freeCashflow"))
    ocf = safe_float(row.get("operatingCashflow"))
    net_income = safe_float(row.get("netIncomeToCommon"))
    shares = safe_float(row.get("sharesOutstanding"))

    total_dividends = np.nan
    if np.isfinite(div_rate) and np.isfinite(shares) and div_rate > 0:
        total_dividends = div_rate * shares

    cash_source = fcf
    cash_label = "FCF"
    if not np.isfinite(cash_source):
        cash_source = ocf
        cash_label = "OCF (fallback)"

    fcf_coverage = np.nan
    if np.isfinite(cash_source) and np.isfinite(total_dividends) and total_dividends > 0:
        fcf_coverage = cash_source / total_dividends

    ni_coverage = np.nan
    if np.isfinite(net_income) and np.isfinite(total_dividends) and total_dividends > 0:
        ni_coverage = net_income / total_dividends

    yield_vs_avg = np.nan
    if np.isfinite(div_yield_pct) and np.isfinite(avg_5y_pct):
        yield_vs_avg = div_yield_pct - avg_5y_pct

    # Safe Dividend rule (dividendYield compared as percent: > 0.01 decimal => > 1.0 pct)
    c_yield = None if not np.isfinite(div_yield_pct) else (div_yield_pct > 1.0)
    c_payout = None if not np.isfinite(payout) else (0 < payout <= 0.65)
    c_cash = None if not np.isfinite(cash_source) else (cash_source > 0)
    verdict = verdict_from_checks(
        [c_yield, c_payout, c_cash],
        pass_text="SAFE DIVIDEND",
        fail_text="UNSAFE / NO DIVIDEND",
    )

    metrics = [
        metric_row("Dividend Rate", div_rate, "price"),
        metric_row("Dividend Yield", div_yield_pct, "pct"),
        metric_row("5Y Avg Yield", avg_5y_pct, "pct"),
        metric_row("Yield vs 5Y Avg", yield_vs_avg, "pct"),
        metric_row("Payout Ratio", payout * 100 if np.isfinite(payout) else np.nan, "pct"),
        metric_row("Total Dividends Paid", total_dividends, "inr"),
        metric_row(f"{cash_label}", cash_source, "inr"),
        metric_row("FCF Dividend Coverage", fcf_coverage),
        metric_row("Net Income Coverage", ni_coverage),
        check_row("Check: Yield > 1%", c_yield),
        check_row("Check: 0 < Payout <= 65%", c_payout),
        check_row(f"Check: {cash_label} > 0", c_cash),
        metric_row("Dividend Safety", verdict),
    ]

    labels = ["Dividend Yield (%)", "Payout Ratio (%)"]
    actual = [div_yield_pct, payout * 100 if np.isfinite(payout) else np.nan]
    thresholds = [1.0, 65.0]
    checks = [c_yield, c_payout]
    colors = ["#6bcb77" if c is True else "#ff6b6b" if c is False else "#7a5cff" for c in checks]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Actual", x=labels,
                         y=[v if np.isfinite(v) else None for v in actual],
                         marker_color=colors,
                         text=[f"{v:.2f}" if np.isfinite(v) else "N/A" for v in actual],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="Threshold", x=labels, y=thresholds,
                         marker_color="#ffd93d", opacity=0.45,
                         text=[f"{t:.1f}" for t in thresholds], textposition="outside"))
    fig.update_layout(barmode="group", title=f"{ticker} — Dividend Safety vs Thresholds",
                      yaxis_title="Percent")

    summary = [f"Verdict: {verdict}"]
    if np.isfinite(div_yield_pct):
        summary.append(
            f"Yield of {div_yield_pct:.2f}% {'>' if div_yield_pct > 1.0 else '<='} 1% minimum hurdle."
        )
    else:
        summary.append("Dividend yield unavailable for this ticker.")
    if np.isfinite(payout):
        summary.append(
            f"Payout ratio {payout * 100:.1f}% "
            f"{'within' if 0 < payout <= 0.65 else 'outside'} the 0-65% safe band."
        )
    if np.isfinite(fcf_coverage):
        if fcf_coverage < 1.0:
            summary.append(f"{cash_label} covers dividends only {fcf_coverage:.1f}x — distribution not self-funded.")
        elif fcf_coverage > 2.0:
            summary.append(f"{cash_label} covers dividends {fcf_coverage:.1f}x — well supported.")
        else:
            summary.append(f"{cash_label} covers dividends {fcf_coverage:.1f}x — adequately supported.")
    else:
        summary.append(f"{cash_label} coverage not computable — cash flow or dividend data missing.")
    if np.isfinite(ni_coverage):
        summary.append(
            f"Net income covers distributions {ni_coverage:.1f}x — "
            f"{'earnings support the dividend' if ni_coverage >= 1 else 'dividend exceeds earnings'}."
        )
    if np.isfinite(yield_vs_avg):
        if yield_vs_avg > 2:
            summary.append(f"Yield is {yield_vs_avg:.1f}pp above 5Y average — mean reversion opportunity or distress.")
        elif yield_vs_avg < -2:
            summary.append(f"Yield is {abs(yield_vs_avg):.1f}pp below 5Y average — possibly overvalued or cut risk.")
        else:
            summary.append("Yield near its 5Y average — fairly valued on dividend basis.")

    return make_result(metrics, fig_to_html(fig), append_as_of(summary, row))
