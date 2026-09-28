import numpy as np
import pandas as pd
import plotly.graph_objects as go
from langchain_core.tools import tool
from .base import (
    get_row, safe_float, fig_to_html, metric_row, make_result,
    append_as_of, resolve_ticker,
    _load_all, get_universe_tech, percentile_rank_series,
)


def _blend(parts: list) -> pd.Series:
    """Weighted mean of percentile-rank Series skipping NaN, weights renormalized."""
    idx = parts[0][1].index
    total = pd.Series(0.0, index=idx)
    wsum = pd.Series(0.0, index=idx)
    for w, s in parts:
        s = pd.to_numeric(s, errors="coerce")
        m = s.notna()
        total[m] += w * s[m]
        wsum[m] += w
    return (total / wsum).where(wsum > 0)


@tool
def composite_analysis(ticker: str) -> dict:
    """Multi-Factor Composite Score — cross-sectional percentile ranks across all 500 stocks.
    Score = (0.25*Value + 0.25*Quality + 0.20*Growth + 0.20*Momentum + 0.10*Volume) * 100,
    where Value = rank(1/trailingPE), Quality = 0.5*rank(ROE)+0.5*rank(profitMargins),
    Growth = 0.6*rank(earningsGrowth)+0.4*rank(revenueGrowth), Momentum = rank(RETURN_60D),
    Volume = rank(CMF_20); rank = #{x_j <= x_i} / N in [0,1]."""
    df = _load_all()
    ticker = resolve_ticker(ticker.upper().strip())
    if ticker not in df.index:
        raise ValueError(f"Ticker '{ticker}' not found in Nifty 500 data.")
    row = df.loc[ticker]

    pe = pd.to_numeric(df["trailingPE"], errors="coerce")
    inv_pe = (1 / pe).where(pe.abs() > 0)
    r_value = percentile_rank_series(inv_pe)

    roe = pd.to_numeric(df["returnOnEquity"], errors="coerce")
    pm = pd.to_numeric(df["profitMargins"], errors="coerce")
    r_quality = _blend([(0.5, percentile_rank_series(roe)),
                        (0.5, percentile_rank_series(pm))])

    eg = pd.to_numeric(df["earningsGrowth"], errors="coerce")
    rg = pd.to_numeric(df["revenueGrowth"], errors="coerce")
    r_growth = _blend([(0.6, percentile_rank_series(eg)),
                       (0.4, percentile_rank_series(rg))])

    try:
        uni = get_universe_tech().reindex(df.index)
        r_momentum = percentile_rank_series(pd.to_numeric(uni["RETURN_60D"], errors="coerce"))
        r_volume = percentile_rank_series(pd.to_numeric(uni["CMF_20"], errors="coerce"))
    except Exception:
        r_momentum = pd.Series(np.nan, index=df.index)
        r_volume = pd.Series(np.nan, index=df.index)

    composite = _blend([
        (0.25, r_value),
        (0.25, r_quality),
        (0.20, r_growth),
        (0.20, r_momentum),
        (0.10, r_volume),
    ]) * 100

    score = safe_float(composite.get(ticker, np.nan))
    rv = safe_float(r_value.get(ticker, np.nan))
    rq = safe_float(r_quality.get(ticker, np.nan))
    rgk = safe_float(r_growth.get(ticker, np.nan))
    rm = safe_float(r_momentum.get(ticker, np.nan))
    rvv = safe_float(r_volume.get(ticker, np.nan))

    factors = {"Value": rv, "Quality": rq, "Growth": rgk, "Momentum": rm, "Volume": rvv}
    grade = (
        "A+" if np.isfinite(score) and score >= 80 else
        "A" if np.isfinite(score) and score >= 70 else
        "B+" if np.isfinite(score) and score >= 60 else
        "B" if np.isfinite(score) and score >= 50 else
        "C+" if np.isfinite(score) and score >= 40 else
        "C" if np.isfinite(score) and score >= 30 else
        "D" if np.isfinite(score) else "N/A"
    )

    def _pct(v):
        return v * 100 if np.isfinite(v) else np.nan

    pe_val = safe_float(row.get("trailingPE"))
    roe_val = safe_float(row.get("returnOnEquity"))
    w52 = safe_float(row.get("52WeekChange"))

    metrics = [
        metric_row("Composite Score", score),
        metric_row("Grade", grade),
        metric_row("Value Percentile (1/PE)", _pct(rv), "pct"),
        metric_row("Quality Percentile", _pct(rq), "pct"),
        metric_row("Growth Percentile", _pct(rgk), "pct"),
        metric_row("Momentum Percentile (60D)", _pct(rm), "pct"),
        metric_row("Volume Percentile (CMF)", _pct(rvv), "pct"),
        metric_row("Trailing P/E", pe_val, "ratio"),
        metric_row("ROE", roe_val * 100 if np.isfinite(roe_val) else np.nan, "pct"),
        metric_row("60D Return Percentile", _pct(rm), "pct"),
        metric_row("52W Change", w52 * 100 if np.isfinite(w52) else np.nan, "pct"),
    ]

    cats = ["Value", "Quality", "Growth", "Momentum", "Volume"]
    vals = [_pct(v) if np.isfinite(v) else 0 for v in [rv, rq, rgk, rm, rvv]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
        fillcolor="rgba(122,92,255,0.15)", line=dict(color="#7a5cff"), name=ticker,
    ))
    fig.update_layout(
        title=f"{ticker} — Factor Percentiles vs Nifty 500 ({grade})",
        polar=dict(bgcolor="#0e0e16",
                   radialaxis=dict(range=[0, 100], gridcolor="#1e1e2e"),
                   angularaxis=dict(gridcolor="#1e1e2e")),
        showlegend=False,
    )

    summary = [f"Composite Score: {score:.0f}/100 — Grade {grade}" if np.isfinite(score)
               else "Composite score unavailable for this ticker."]
    available = {k: v for k, v in factors.items() if np.isfinite(v)}
    if available:
        strongest = max(available.items(), key=lambda x: x[1])
        weakest = min(available.items(), key=lambda x: x[1])
        summary.append(f"Strongest factor: {strongest[0]} ({strongest[1] * 100:.0f}th percentile).")
        summary.append(f"Weakest factor: {weakest[0]} ({weakest[1] * 100:.0f}th percentile).")
    missing = [k for k, v in factors.items() if not np.isfinite(v)]
    if missing:
        summary.append("Missing inputs (weights redistributed): " + ", ".join(missing) + ".")
    if np.isfinite(score):
        if score >= 70:
            summary.append("Top-30% multi-factor profile — strong investment candidate.")
        elif score >= 50:
            summary.append("Middle-of-pack profile — mixed signals across factors.")
        else:
            summary.append("Bottom-half composite score — multiple factors below average.")

    return make_result(metrics, fig_to_html(fig), append_as_of(summary, row))
