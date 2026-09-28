import numpy as np
import pandas as pd
import plotly.graph_objects as go
from langchain_core.tools import tool
from .base import (
    get_row, safe_float, fig_to_html, metric_row, make_result, append_as_of, _load_all,
)


def _positive_median(s) -> float:
    s = pd.to_numeric(s, errors="coerce")
    s = s[s > 0]
    return float(s.median()) if len(s) >= 3 else np.nan


@tool
def valuation_analysis(ticker: str) -> dict:
    """Relative Valuation & Multiples — cross-sectional comparison against sector peers.
    Relative P/E = stock P/E / median sector P/E (<1 discount, >1 premium).
    Sector-Adjusted P/E Z-Score = (P/E - mu_sector) / sigma_sector."""
    row = get_row(ticker)
    sector = str(row.get("sector", "") or "")

    pe = safe_float(row.get("trailingPE"))
    fpe = safe_float(row.get("forwardPE"))
    pb = safe_float(row.get("priceToBook"))
    ps = safe_float(row.get("priceToSalesTrailing12Months"))
    ev = safe_float(row.get("enterpriseValue"))
    ev_rev = safe_float(row.get("enterpriseToRevenue"))
    ev_ebitda = safe_float(row.get("enterpriseToEbitda"))
    roe = safe_float(row.get("returnOnEquity"))
    roe_pct = roe * 100 if np.isfinite(roe) else np.nan

    df = _load_all()
    if sector:
        peers = df[df["sector"].astype(str) == sector]
    else:
        peers = df.iloc[0:0]
    n_peers = len(peers)

    med_pe = _positive_median(peers["trailingPE"]) if n_peers >= 3 else np.nan
    med_pb = _positive_median(peers["priceToBook"]) if n_peers >= 3 else np.nan
    med_ev = _positive_median(peers["enterpriseToEbitda"]) if n_peers >= 3 else np.nan

    pe_pos = pd.to_numeric(peers["trailingPE"], errors="coerce")
    pe_pos = pe_pos[pe_pos > 0]
    if len(pe_pos) >= 3:
        mu = float(pe_pos.mean())
        sigma = float(pe_pos.std(ddof=0))
    else:
        mu = sigma = np.nan

    z_pe = np.nan
    if np.isfinite(pe) and pe > 0 and np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
        z_pe = (pe - mu) / sigma

    def _rel(val, med):
        if np.isfinite(val) and val > 0 and np.isfinite(med) and med > 0:
            return val / med
        return np.nan

    rel_pe = _rel(pe, med_pe)
    rel_pb = _rel(pb, med_pb)
    rel_ev = _rel(ev_ebitda, med_ev)

    metrics = [
        metric_row("Sector", sector or "N/A"),
        metric_row("Sector Peers", int(n_peers)),
        metric_row("Trailing P/E", pe, "ratio"),
        metric_row("Sector Median P/E", med_pe, "ratio"),
        metric_row("Relative P/E", rel_pe, "ratio"),
        metric_row("P/E Z-Score", z_pe),
        metric_row("Price/Book", pb, "ratio"),
        metric_row("Sector Median P/B", med_pb, "ratio"),
        metric_row("Relative P/B", rel_pb, "ratio"),
        metric_row("EV/EBITDA", ev_ebitda, "ratio"),
        metric_row("Sector Median EV/EBITDA", med_ev, "ratio"),
        metric_row("Relative EV/EBITDA", rel_ev, "ratio"),
        metric_row("Forward P/E", fpe, "ratio"),
        metric_row("Price/Sales", ps, "ratio"),
        metric_row("EV/Revenue", ev_rev, "ratio"),
        metric_row("Enterprise Value", ev, "inr"),
        metric_row("ROE", roe_pct, "pct"),
    ]

    labels = ["P/E", "P/B", "EV/EBITDA"]
    stock_vals = [pe, pb, ev_ebitda]
    med_vals = [med_pe, med_pb, med_ev]

    def _y(vals):
        return [v if np.isfinite(v) else None for v in vals]

    def _txt(vals, suffix="x"):
        return [f"{v:.2f}{suffix}" if np.isfinite(v) else "N/A" for v in vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(name=ticker, x=labels, y=_y(stock_vals),
                         marker_color="#00d2ff", text=_txt(stock_vals),
                         textposition="outside"))
    fig.add_trace(go.Bar(name="Sector Median", x=labels, y=_y(med_vals),
                         marker_color="#7a5cff", text=_txt(med_vals),
                         textposition="outside"))
    fig.update_layout(barmode="group",
                      title=f"{ticker} vs {sector or 'Sector'} Median Multiples",
                      xaxis_title="Multiple", yaxis_title="Value (x)")

    summary = []
    if n_peers < 3:
        summary.append(f"Only {n_peers} sector peers available — relative metrics not reliable.")

    def _rel_line(name, rel, med):
        if np.isfinite(rel):
            pct = (rel - 1) * 100
            side = "discount" if rel < 1 else "premium"
            verdict = "relatively cheap" if rel < 1 else "relatively expensive"
            summary.append(
                f"{name}: {rel:.2f}x sector median of {med:.2f} — "
                f"{abs(pct):.1f}% {side}, {verdict} vs peers."
            )

    _rel_line("Relative P/E", rel_pe, med_pe)
    _rel_line("Relative P/B", rel_pb, med_pb)
    _rel_line("Relative EV/EBITDA", rel_ev, med_ev)

    if np.isfinite(z_pe):
        if z_pe >= 1.0:
            summary.append(f"P/E Z-Score {z_pe:+.2f} — priced richly (>1 sigma above sector mean).")
        elif z_pe <= -1.0:
            summary.append(f"P/E Z-Score {z_pe:+.2f} — priced cheaply (>1 sigma below sector mean).")
        else:
            summary.append(f"P/E Z-Score {z_pe:+.2f} — within one sigma of the sector mean (fairly valued).")

    return make_result(metrics, fig_to_html(fig), append_as_of(summary, row))
