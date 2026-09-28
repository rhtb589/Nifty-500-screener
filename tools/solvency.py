import numpy as np
import plotly.graph_objects as go
from langchain_core.tools import tool
from .base import (
    get_row, safe_float, fig_to_html, metric_row, make_result, append_as_of,
    check_row, verdict_from_checks,
)


@tool
def solvency_analysis(ticker: str) -> dict:
    """Solvency, Liquidity & Distress Screening (absolute, per-stock).
    Cash Flow Coverage = operatingCashflow / totalDebt.
    Solvent = (currentRatio >= 1.3) and (quickRatio >= 0.9) and (D/E < 1.0) and (operatingCashflow > 0).
    NOTE: debtToEquity in the dataset is stored in percent — converted to a ratio here."""
    row = get_row(ticker)
    total_debt = safe_float(row.get("totalDebt"))
    total_cash = safe_float(row.get("totalCash"))
    de_pct = safe_float(row.get("debtToEquity"))      # stored as percent
    current_ratio = safe_float(row.get("currentRatio"))
    quick_ratio = safe_float(row.get("quickRatio"))
    cash_per_share = safe_float(row.get("totalCashPerShare"))
    fcf = safe_float(row.get("freeCashflow"))
    ocf = safe_float(row.get("operatingCashflow"))

    de = de_pct / 100 if np.isfinite(de_pct) else np.nan

    net_debt = np.nan
    if np.isfinite(total_debt) and np.isfinite(total_cash):
        net_debt = total_debt - total_cash

    cash_coverage = np.nan
    if np.isfinite(ocf) and np.isfinite(total_debt) and total_debt > 0:
        cash_coverage = ocf / total_debt

    c_cr = None if not np.isfinite(current_ratio) else (current_ratio >= 1.3)
    c_qr = None if not np.isfinite(quick_ratio) else (quick_ratio >= 0.9)
    c_de = None if not np.isfinite(de) else (de < 1.0)
    c_ocf = None if not np.isfinite(ocf) else (ocf > 0)
    verdict = verdict_from_checks(
        [c_cr, c_qr, c_de, c_ocf],
        pass_text="SOLVENT",
        fail_text="AT RISK",
    )

    metrics = [
        metric_row("Total Debt", total_debt, "inr"),
        metric_row("Total Cash", total_cash, "inr"),
        metric_row("Net Debt", net_debt, "inr"),
        metric_row("Debt/Equity", de, "ratio"),
        metric_row("Current Ratio", current_ratio),
        metric_row("Quick Ratio", quick_ratio),
        metric_row("Operating Cash Flow", ocf, "inr"),
        metric_row("Free Cash Flow", fcf, "inr"),
        metric_row("Cash/Share", cash_per_share, "price"),
        metric_row("OCF/Debt Coverage", cash_coverage),
        check_row("Check: Current Ratio >= 1.3", c_cr),
        check_row("Check: Quick Ratio >= 0.9", c_qr),
        check_row("Check: D/E < 1.0x", c_de),
        check_row("Check: Operating CF > 0", c_ocf),
        metric_row("Solvency Filter", verdict),
    ]

    debt_v = total_debt if np.isfinite(total_debt) else 0.0
    cash_v = total_cash if np.isfinite(total_cash) else 0.0
    net_v = net_debt if np.isfinite(net_debt) else 0.0
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Total Debt", "Total Cash", "Net Debt"],
                         y=[debt_v, cash_v, net_v],
                         marker_color=["#ff6b6b", "#6bcb77", "#ffd93d"],
                         text=[f"\u20B9{v / 1e7:,.0f} Cr" for v in [debt_v, cash_v, net_v]],
                         textposition="outside"))
    fig.update_layout(title=f"{ticker} — Debt vs Cash Position", yaxis_title="Amount (\u20B9)")

    summary = [f"Verdict: {verdict}"]
    if np.isfinite(current_ratio):
        summary.append(
            f"Current ratio {current_ratio:.2f} {'>=' if current_ratio >= 1.3 else '<'} 1.3 — "
            f"{'adequate' if current_ratio >= 1.3 else 'tight'} short-term liquidity buffer."
        )
    else:
        summary.append("Current ratio unavailable for this ticker.")
    if np.isfinite(de):
        summary.append(
            f"D/E of {de:.2f}x {'<' if de < 1.0 else '>='} 1.0 — "
            f"{'conservative' if de < 1.0 else 'elevated'} leverage."
        )
    if np.isfinite(ocf) and np.isfinite(total_debt) and total_debt > 0:
        summary.append(f"Operating cash flow covers {cash_coverage * 100:.1f}% of total debt.")
    elif np.isfinite(ocf) and (not np.isfinite(total_debt) or total_debt == 0):
        summary.append("No material debt — operating cash flow fully unencumbered.")
    else:
        summary.append("Operating cash flow unavailable — coverage not verifiable.")
    if np.isfinite(net_debt) and net_debt < 0:
        summary.append(f"Net cash position of \u20B9{abs(net_debt) / 1e7:,.0f} Cr — debt free after cash.")

    return make_result(metrics, fig_to_html(fig), append_as_of(summary, row))
