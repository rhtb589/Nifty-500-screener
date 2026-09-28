from .valuation import valuation_analysis
from .growth import growth_analysis
from .profitability import profitability_analysis
from .solvency import solvency_analysis
from .dividend import dividend_analysis
from .trend import trend_analysis
from .breakout import breakout_analysis
from .volume import volume_analysis
from .sentiment import sentiment_analysis
from .composite import composite_analysis
from .screener import nifty500_screener_tool

ANALYSIS_MAP = {
    1: ("Relative Valuation & Multiples", valuation_analysis),
    2: ("GARP & PEG Modeling", growth_analysis),
    3: ("Profitability & DuPont", profitability_analysis),
    4: ("Solvency & Liquidity", solvency_analysis),
    5: ("Dividend Sustainability", dividend_analysis),
    6: ("Trend & Moving Averages", trend_analysis),
    7: ("Support/Resistance & Breakout", breakout_analysis),
    8: ("Volume Anomaly & Accumulation", volume_analysis),
    9: ("Analyst Consensus & Sentiment", sentiment_analysis),
    10: ("Multi-Factor Composite Score", composite_analysis),
    11: ("Stock Screener & Filter", nifty500_screener_tool),
}

ANALYSIS_LIST = [{"id": k, "name": v[0]} for k, v in sorted(ANALYSIS_MAP.items()) if k != 11]
ANALYSIS_LIST.append({"id": 11, "name": "Ask AI"})
