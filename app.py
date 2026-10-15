import os
import re
import dotenv
from flask import Flask, render_template, jsonify, request, redirect
from tools import (
    ANALYSIS_MAP, ANALYSIS_LIST,
    valuation_analysis, growth_analysis, profitability_analysis,
    solvency_analysis, dividend_analysis, trend_analysis,
    breakout_analysis, volume_analysis, sentiment_analysis,
    composite_analysis,
)
from tools.screener import nifty500_screener_tool
from tools.base import (
    get_ticker_options, get_stock_overview, get_row, safe_float,
    reset_caches, resolve_ticker,
)
from tools.cashflow_statement import build_cashflow_payload , get_cashflow_statement
from tools.balance_sheet_statement import build_balance_sheet_payload , get_balance_sheet
from tools.income_statement import build_income_statement_payload , get_income_statement
#from langchain_ollama import ChatOllama
from langchain_xai import ChatXAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)
dotenv.load_dotenv()
# Ollama endpoint/model are env-configurable for container integration.
# llm = ChatOllama(
#     model=os.environ.get("LLM_MODEL", "qwen3.5:4b"),
#     temperature=0.7,
#     base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
# )


# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",  # or "gemini-2.5-pro"
    temperature=0.2,
)

all_tools = [
    valuation_analysis, growth_analysis, profitability_analysis,
    solvency_analysis, dividend_analysis, trend_analysis,
    breakout_analysis, volume_analysis, sentiment_analysis,
    composite_analysis, nifty500_screener_tool,get_income_statement , get_cashflow_statement , get_balance_sheet
]

TOOL_NAME_MAP = {t.name: t for t in all_tools}


def _tool_reply(tool_messages: list, tc: dict, content: str):
    """Append an AIMessage(tool_calls=[tc]) + ToolMessage pair so the model
    always receives a result for every tool call — including failures and
    unknown tools. Missing that pair makes the next invoke fail."""
    tool_messages.append(AIMessage(content="", tool_calls=[tc]))
    tool_messages.append(ToolMessage(
        content=content,
        tool_call_id=tc.get("id", ""),
    ))


def _content_text(content) -> str:
    """Normalize an AIMessage content (str, or list of content blocks) to str."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def _history_text(content: str) -> str:
    """Compact screener HTML tables in earlier turns into plain-text rows so the
    conversation history stays small enough to keep the full chat in context."""
    if not content or "screener-table-wrap" not in content:
        return content
    head, _, rest = content.partition('<div class="screener-table-wrap">')
    rows = re.findall(r"<tr.*?</tr>", rest, flags=re.S)
    lines = []
    for r in rows[:25]:
        cells = [
            re.sub(r"<.*?>", "", c).strip()
            for c in re.findall(r"<t[hd].*?</t[hd]>", r, flags=re.S)
        ]
        if cells:
            lines.append(" | ".join(cells))
    omitted = len(rows) - len(lines)
    tail = f"\n[... {omitted} more matching stocks omitted ...]" if omitted > 0 else ""
    return head + "\n".join(lines) + tail


TICKER_GUIDANCE = """If the user asks about a stock by name, resolve it to the ticker with the closest
symbol, or the stock with a similar name in the shortName column of the dataset.
Always use the short name while addressing the stock in your response, and use the
full name in the summary of the analysis."""

TOOL_GUIDE = """You have access to 14 tools:

Single-stock analysis tools (take a ticker string, return metrics + summary):
1. valuation_analysis — Relative valuation & multiples
2. growth_analysis — GARP & PEG modeling
3. profitability_analysis — Profitability & DuPont decomposition
4. solvency_analysis — Solvency, liquidity & distress screening
5. dividend_analysis — Dividend sustainability & cash yield
6. trend_analysis — Trend regime & moving average alignment
7. breakout_analysis — Support, resistance & breakout proximity
8. volume_analysis — Volume anomaly & accumulation signals
9. sentiment_analysis — Analyst consensus & price targets
10. composite_analysis — Multi-factor composite quant score

Stock screener tool:
11. screen_nifty500 — Executes pandas filter code against 500 Indian stocks.
    Takes a 'query' parameter: Python pandas code that filters the DataFrame 'df'.
    ALL matching stocks are returned automatically — do NOT use .head() unless user asks for a specific number.

    ALIASES (also accepted): price/close → currentPrice, pe/trailing_pe → trailingPE,
    pb/price_to_book → priceToBook, market_cap → marketCap, roe → returnOnEquity,
    profit_margin → profitMargins, dividend_yield → dividendYield

    CRITICAL SYNTAX RULES:
    1. Always use parentheses around EACH comparison when combining with & or |:
       CORRECT: df = df[(df['trailingPE'] < 20) & (df['dividendYield'] > 0.02)]
       WRONG:   df = df[df['trailingPE'] < 20 & df['dividendYield'] > 0.02]
    2. For "less than" use < operator, for "greater than" use > operator
    3. For "less than or equal" use <=, for "greater than or equal" use >=

    EXAMPLES OF CORRECT QUERIES:
    - "stocks with price less than 50":
      df = df[df['currentPrice'] < 50]
    - "stocks with PE ratio less than 20":
      df = df[df['trailingPE'] < 20]
    - "stocks with dividend yield greater than 3%":
      df = df[df['dividendYield'] > 0.03]
    - "stocks with market cap greater than 1 trillion":
      df = df[df['marketCap'] > 1e12]
    - "stocks with ROE greater than 15% and profit margin greater than 10%":
      df = df[(df['returnOnEquity'] > 0.15) & (df['profitMargins'] > 0.10)]
    - "stocks with ROE greater than 15% or profit margin greater than 10%":
      df = df[(df['returnOnEquity'] > 0.15) | (df['profitMargins'] > 0.10)]
    - "sort by price ascending":
      df = df.sort_values('currentPrice')
    - "top 10 stocks by market cap":
      df = df.sort_values('marketCap', ascending=False).head(10)

    - if there are queries with multiple filtering then merge the queries instead of chaining them

Financial statement tools (take a ticker string, return the full statement):
12. get_cashflow_statement — Cash flow statement
13. get_balance_sheet — Balance sheet
14. get_income_statement — Income statement

When a user asks about metrics, valuations, trends, or any financial data for a specific stock, use tools 1-10 or 12-14.
When a user asks to find, filter, or screen stocks matching criteria, use tool 11 (screen_nifty500).
Always provide clear, concise, and actionable insights. Use Indian Rupee (₹) formatting when discussing prices.

RESPONSE FORMAT for screener results:
- The screener tool returns a table with ALL matching stocks. Display this table to the user as-is.
- Your text summary should ONLY highlight the top 5-10 most notable picks with brief reasoning.
- Do NOT repeat the full table in your text — the table is already shown separately.

IMPORTANT TOOL RULES:
- After calling ANY tool and receiving results, present those results directly to the user. DO NOT call the same tool again.
- Each tool returns complete data. One call is enough.
- After receiving tool results, write your final response to the user summarizing the findings."""

CONVERSATION_GUIDE = """CONVERSATION RULES:
- This is an ongoing, multi-turn conversation. Every message below is part of the
  running conversation — read it before answering, and never treat a question in isolation.
- Answer the user's LATEST question, using the earlier turns to resolve references such as
  "this stock", "it", "this", "what would you suggest", "should I buy", "compare them", or
  "how about the other one".
- If the figures or analysis needed to answer are already present in the conversation
  history or in the analysis findings above, answer directly from them — do NOT call the
  same tool again just to re-fetch data you already have.
- Recommendations, suggestions, and opinions are welcome: base them on the data discussed
  in this conversation and clearly state the reasoning and any risks.
- Every turn MUST end with a final, plain-language answer written for the user. Never end a
  turn on a tool call, and never return an empty response.
- If something is genuinely missing (for example, no stock has been mentioned yet), ask one
  short clarifying question instead of failing silently."""


@app.route("/")
def index():
    return render_template("index.html", analyses=ANALYSIS_LIST)


@app.route("/api/tickers")
def api_tickers():
    return jsonify(get_ticker_options())


@app.route("/api/stock/<ticker>")
def api_stock(ticker):
    try:
        data = get_stock_overview(ticker)
        return jsonify(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to load stock data: {str(e)}"}), 500


@app.route("/cashflow/<ticker>")
def cashflow_page(ticker):
    """Standalone Cash Flow Statement dashboard (dark theme, Plotly)."""
    try:
        key = resolve_ticker(ticker)
        get_row(key)
    except (ValueError, KeyError):
        return redirect("/")
    return render_template("cashflow.html", ticker=key)


@app.route("/api/cashflow/<ticker>")
def api_cashflow(ticker):
    try:
        return jsonify(build_cashflow_payload(ticker))
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to load cash flow statement: {str(e)}"}), 500


@app.route("/balance_sheet/<ticker>")
def balance_sheet_page(ticker):
    """Standalone Balance Sheet dashboard (dark theme, Plotly)."""
    try:
        key = resolve_ticker(ticker)
        get_row(key)
    except (ValueError, KeyError):
        return redirect("/")
    return render_template("balance_sheet.html", ticker=key)


@app.route("/api/balance_sheet/<ticker>")
def api_balance_sheet(ticker):
    try:
        return jsonify(build_balance_sheet_payload(ticker))
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to load balance sheet: {str(e)}"}), 500


@app.route("/income_statement/<ticker>")
def income_statement_page(ticker):
    """Standalone Income Statement dashboard (dark theme, Plotly)."""
    try:
        key = resolve_ticker(ticker)
        get_row(key)
    except (ValueError, KeyError):
        return redirect("/")
    return render_template("income_statement.html", ticker=key)


@app.route("/api/income_statement/<ticker>")
def api_income_statement(ticker):
    try:
        return jsonify(build_income_statement_payload(ticker))
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to load income statement: {str(e)}"}), 500


@app.route("/api/reload", methods=["POST"])
def api_reload():
    """Drop cached CSVs so data refreshed by the daily pipeline (scheduler.py)
    is served immediately, without restarting the app.

    Optionally protected: set RELOAD_TOKEN in the environment and send it as
    the X-Reload-Token header.
    """
    expected = os.environ.get("RELOAD_TOKEN")
    if expected and request.headers.get("X-Reload-Token") != expected:
        return jsonify({"error": "invalid token"}), 403
    reset_caches()
    return jsonify({"status": "ok", "detail": "csv caches reset"})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json()
    ticker = data.get("ticker", "").upper().strip()
    analysis_id = data.get("analysis_id")

    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400
    if analysis_id not in ANALYSIS_MAP:
        return jsonify({"error": f"Invalid analysis ID: {analysis_id}"}), 400

    name, tool_fn = ANALYSIS_MAP[analysis_id]
    try:
        result = tool_fn.invoke(ticker)
        return jsonify({"ticker": ticker, "analysis": name, **result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    ticker = data.get("ticker", "")
    if ticker:
        ticker = ticker.upper().strip()
    else:
        ticker = None
    analysis_id = data.get("analysis_id")
    messages_raw = data.get("messages", [])

    if not messages_raw:
        return jsonify({"error": "No messages provided"}), 400

    overview = None
    if ticker:
        try:
            overview = get_stock_overview(ticker)
        except Exception:
            pass

    analysis_name = ""
    analysis_summary = ""
    if analysis_id and analysis_id in ANALYSIS_MAP and ticker:
        analysis_name = ANALYSIS_MAP[analysis_id][0]
        try:
            result = ANALYSIS_MAP[analysis_id][1].invoke(ticker)
            summary_lines = result.get("summary", [])
            analysis_summary = "\n".join(f"- {s}" for s in summary_lines) if summary_lines else "No summary available."
        except Exception:
            analysis_summary = "Analysis data unavailable."

    if overview:
        if analysis_name:
            analysis_block = (
                f"Current analysis: {analysis_name}\n"
                f"Analysis findings:\n{analysis_summary or 'No summary available.'}"
            )
        else:
            analysis_block = (
                "No specific analysis tab is open \u2014 use the tools to fetch whatever "
                "data the user asks about."
            )
        system_prompt = f"""You are a financial analysis assistant for the Indian stock market.
You are currently analyzing **{ticker}** \u2014 {overview['name']}.
Sector: {overview['sector']} | Industry: {overview['industry']}

Key stock metrics:
{chr(10).join(f"- {m['label']}: {m['value']}" for m in overview['metrics'])}

{analysis_block}

{TICKER_GUIDANCE}

{TOOL_GUIDE}

{CONVERSATION_GUIDE}"""
    else:
        system_prompt = f"""You are a financial analysis assistant for the Indian stock market.
You answer questions about Indian stocks, financial markets, financial statements and
investment analysis \u2014 whether or not a stock is selected in the UI.

{TICKER_GUIDANCE}

{TOOL_GUIDE}

{CONVERSATION_GUIDE}"""

    langchain_messages = [SystemMessage(content=system_prompt)]
    # Replay the conversation so follow-up questions keep their context.
    # Older turns are trimmed, but a turn is never split (we always start on a
    # user message) and older history is preserved in the browser.
    for msg in messages_raw[-30:]:
        role = msg.get("role", "user")
        content = _history_text(msg.get("content", ""))
        if not content:
            continue
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
    while len(langchain_messages) > 1 and isinstance(langchain_messages[1], AIMessage):
        langchain_messages.pop(1)

    try:
        bound_llm = llm.bind_tools(all_tools)
        response = bound_llm.invoke(langchain_messages)

        MAX_TOOL_ITERATIONS = 5
        iteration = 0
        tool_messages = []
        while response.tool_calls and iteration < MAX_TOOL_ITERATIONS:
            iteration += 1
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                print(f"{tool_name}")
                if tool_name == "screen_nifty500":
                    print("calling screen_nifty500 with args:", tool_args)
                    try:
                        tool_result = nifty500_screener_tool.invoke(tool_args)
                    except Exception as e:
                        _tool_reply(
                            tool_messages, tc,
                            f"Screener error: {e}. Fix the pandas query and call "
                            "the tool again, or explain the problem to the user.",
                        )
                        continue
                    total = tool_result.get("total_matches", 0)
                    results = tool_result.get("results", [])
                    if results:
                        DISPLAY_CHAT_COLS = [
                            "Ticker", "shortName", "currentPrice", "trailingPE",
                            "returnOnEquity", "profitMargins", "dividendYield",
                            "marketCap", "fiftyDayAverage", "52WeekChange",
                        ]
                        # Always show the indicators the scan filtered/sorted on
                        query_cols = tool_result.get("query_columns", []) or []
                        headers = []
                        for h in DISPLAY_CHAT_COLS + list(query_cols):
                            if h in results[0] and h not in headers:
                                headers.append(h)
                        display_results = results

                        # Units verified against the CSVs:
                        #   percent-as-is columns already hold e.g. 1.64 for 1.64%
                        #   ratio columns hold 0.19 for 19% and must be x100
                        _PCT_AS_IS = {
                            "dividendYield", "regularMarketChangePercent",
                            "fulldayChangePercent", "debtToEquity",
                            "price_position_52w", "dist_from_50dma",
                            "dist_from_200dma", "bb_width", "donchian_width",
                            "kc_width",
                        }
                        _PCT_RATIO = {
                            "profitMargins", "revenueGrowth", "earningsGrowth",
                            "returnOnEquity", "returnOnAssets", "payoutRatio",
                            "52WeekChange",
                        }
                        _MONEY_COLS = {
                            "previousClose", "regularMarketChange", "regularMarketOpen",
                            "regularMarketPreviousClose", "fiftyTwoWeekLow",
                            "fiftyTwoWeekHigh", "trailingEps", "forwardEps", "eps",
                            "VWAP", "Open", "High", "Low", "Close",
                            "SMA_20", "SMA_50", "SMA_100", "SMA_200",
                            "EMA_20", "EMA_50", "EMA_100", "EMA_200",
                        }

                        def _fmt_val(v, h):
                            if v is None:
                                return "--"
                            if isinstance(v, float):
                                if h in _PCT_AS_IS:
                                    return f"{v:.2f}%"
                                if (h in _PCT_RATIO or h.startswith(("RETURN_", "QOQ_"))
                                        or "margin" in h.lower() or "growth" in h.lower()):
                                    return f"{v*100:.2f}%"
                                if ("cap" in h.lower() or h.endswith("Price")
                                        or h.endswith("Average") or h in _MONEY_COLS):
                                    return f"₹{v:,.2f}" if v < 1e12 else f"₹{v/1e9:.1f}B"
                                return f"{v:.2f}"
                            return str(v)

                        header_html = "".join(f"<th>{h}</th>" for h in headers)
                        rows_html = ""
                        for idx, r in enumerate(display_results):
                            row_cls = "screener-row-even" if idx % 2 == 0 else "screener-row-odd"
                            cells = "".join(f"<td>{_fmt_val(r.get(h), h)}</td>" for h in headers)
                            rows_html += f'<tr class="{row_cls}"><td class="screener-idx">{idx}</td>{cells}</tr>'

                        table = (
                            '<div class="screener-table-wrap"><table class="screener-table">'
                            f'<thead><tr><th>#</th>{header_html}</tr></thead>'
                            f'<tbody>{rows_html}</tbody></table></div>'
                        )
                        reply = f"Found **{total} stocks** matching your criteria:\n\n{table}"
                    else:
                        reply = "No stocks matched the given criteria."

                    return jsonify({"reply": reply})
                elif tool_name in TOOL_NAME_MAP:
                    tool_ticker = tool_args.get("ticker", ticker or "TCS.NS")
                    try:
                        tool_result = TOOL_NAME_MAP[tool_name].invoke(tool_ticker)
                    except Exception as e:
                        _tool_reply(
                            tool_messages, tc,
                            f"Tool '{tool_name}' failed for '{tool_ticker}': {e}. "
                            "Try a different ticker from the dataset, or tell the "
                            "user that data is unavailable.",
                        )
                        continue
                    summary_text = "\n".join(tool_result.get("summary", []))
                    metrics_text = ", ".join(
                        f"{m['label']}: {m['value']}"
                        for m in tool_result.get("metrics", [])
                        if m.get("value") != "--"
                    )
                    _tool_reply(
                        tool_messages, tc,
                        f"Metrics: {metrics_text}\nSummary:\n{summary_text}",
                    )
                else:
                    known = ", ".join(sorted(set(TOOL_NAME_MAP) | {"screen_nifty500"}))
                    _tool_reply(
                        tool_messages, tc,
                        f"Unknown tool '{tool_name}'. Available tools: {known}. "
                        "Call one of those instead.",
                    )

            langchain_messages.extend(tool_messages)
            tool_messages = []
            response = bound_llm.invoke(langchain_messages)

        reply = _content_text(getattr(response, "content", "")).strip()
        if not reply or getattr(response, "tool_calls", None):
            # The model either returned nothing, or stopped on an unanswered tool
            # call. Nudge it once — this time without tools bound — so it must
            # answer from the conversation history instead of failing silently.
            nudge = list(tool_messages)
            for tc in getattr(response, "tool_calls", None) or []:
                _tool_reply(
                    nudge, tc,
                    "No new data is available for that call. Use the results "
                    "already present in this conversation and answer the "
                    "user's question directly.",
                )
            nudge.append(HumanMessage(
                content=(
                    "Give your final answer to my last question now, using the "
                    "analysis, figures and earlier answers already discussed in "
                    "this conversation. Do not call any more tools."
                )
            ))
            try:
                reply = _content_text(
                    llm.invoke(langchain_messages + nudge).content
                ).strip()
            except Exception:
                reply = ""
        if not reply:
            reply = ("I could not produce a response for that. Please rephrase "
                     "your question, or name the stock you would like me to look at.")
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "1") == "1",
    )