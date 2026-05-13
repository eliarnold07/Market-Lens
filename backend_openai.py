from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
CORS(app)

# Keep API keys on the backend only. Never put this in marketlens.html.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

STRATEGY_OPTIONS = [
    "Conservative Investor",
    "Balanced Investor",
    "Aggressive Growth",
    "Momentum Trader",
    "Long-Term Compounder",
    "Value Hunter",
]

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def fetch_news_headlines(query, limit=5, company_name=None, ticker=None):
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key or not query:
        return []

    query = query.strip()
    if not query:
        return []

    def normalize_text(text):
        if not text:
            return ""
        return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()

    company_term = normalize_text(company_name) if company_name else ""
    ticker_term = normalize_text(ticker) if ticker else ""

    from_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
    params = {
        "qInTitle": query,
        "language": "en",
        "from": from_date,
        "pageSize": limit,
        "sortBy": "publishedAt",
    }
    url = f"{NEWSAPI_ENDPOINT}?{urllib.parse.urlencode(params)}&apiKey={urllib.parse.quote(api_key)}"

    try:
        request_obj = urllib.request.Request(url, headers={"User-Agent": "MarketLens/1.0"})
        with urllib.request.urlopen(request_obj, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            if data.get("status") != "ok":
                return []
            headlines = []
            seen_titles = set()
            for article in data.get("articles", [])[:limit * 3]:
                title = article.get("title")
                description = article.get("description") or ""
                if not title or title in seen_titles:
                    continue

                combined_text = f"{title} {description}".lower()
                normalized_text = normalize_text(combined_text)
                if company_term and company_term not in normalized_text and ticker_term and ticker_term not in normalized_text:
                    continue
                if company_term and company_term not in normalized_text and not ticker_term:
                    continue
                if ticker_term and ticker_term not in normalized_text and not company_term:
                    continue

                seen_titles.add(title)
                headlines.append({
                    "title": title,
                    "description": description,
                    "source": article.get("source", {}).get("name"),
                    "published_at": article.get("publishedAt"),
                    "url": article.get("url"),
                })
                if len(headlines) >= limit:
                    break
            return headlines
    except Exception:
        return []


def format_large_number(num):
    """Format large numbers like 1.2T, 45B, 300M"""
    if num is None:
        return "N/A"

    try:
        num = float(num)
        if num >= 1e12:
            return f"${num/1e12:.2f}T"
        elif num >= 1e9:
            return f"${num/1e9:.2f}B"
        elif num >= 1e6:
            return f"${num/1e6:.2f}M"
        else:
            return f"${num:,.0f}"
    except Exception:
        return "N/A"


def get_sentiment(change_percent):
    """Determine quick sentiment based on price change"""
    if change_percent is None:
        return "Neutral"
    if change_percent > 2:
        return "Bullish"
    elif change_percent < -2:
        return "Bearish"
    else:
        return "Neutral"


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def moving_average(series, window):
    try:
        return float(series.tail(window).mean()) if len(series) >= window else None
    except Exception:
        return None


def compute_rsi(close_series, window=14):
    try:
        if len(close_series) < window + 1:
            return None
        delta = close_series.diff().dropna()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean().iloc[-1]
        avg_loss = loss.rolling(window=window, min_periods=window).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))
    except Exception:
        return None


def compute_volatility(close_series, window=30):
    try:
        if len(close_series) < window + 1:
            return None
        returns = close_series.pct_change().dropna().tail(window)
        return float(returns.std() * (252 ** 0.5) * 100)
    except Exception:
        return None


def compute_volume_trend(volume_series, short=10, long=50):
    try:
        if len(volume_series) < long:
            return None
        short_avg = float(volume_series.tail(short).mean())
        long_avg = float(volume_series.tail(long).mean())
        return {
            "short_avg": short_avg,
            "long_avg": long_avg,
            "trend": "higher" if short_avg > long_avg else "lower" if short_avg < long_avg else "flat",
            "ratio": float(short_avg / long_avg) if long_avg else None,
        }
    except Exception:
        return None


def calculate_risk_score(pe_ratio, rsi, volatility_30d, current_price, ma50, ma200, distance_from_52w_high):
    score = 0

    if volatility_30d is not None:
        if volatility_30d > 60:
            score += 4
        elif volatility_30d > 45:
            score += 3
        elif volatility_30d > 35:
            score += 2
        elif volatility_30d > 25:
            score += 1

    if rsi is not None:
        if rsi > 80 or rsi < 20:
            score += 3
        elif rsi > 70 or rsi < 30:
            score += 2
        elif rsi > 65 or rsi < 35:
            score += 1

    if ma200 is not None and current_price is not None and current_price < ma200:
        score += 3
    elif ma50 is not None and current_price is not None and current_price < ma50:
        score += 1

    if pe_ratio is not None:
        if pe_ratio > 80:
            score += 3
        elif pe_ratio > 50:
            score += 2
        elif pe_ratio > 30:
            score += 1

    if distance_from_52w_high is not None:
        if distance_from_52w_high < -40:
            score += 2
        elif distance_from_52w_high < -20:
            score += 1

    return min(score, 10)


def calculate_momentum_score(current_price, price_change_pct, ma50, ma200, rsi, distance_from_52w_high, volume_trend=None):
    score = 0

    if ma50 is not None and current_price is not None and current_price > ma50:
        score += 2

    if ma200 is not None and current_price is not None and current_price > ma200:
        score += 3

    if ma50 is not None and ma200 is not None and current_price is not None and current_price > ma50 and current_price > ma200:
        score += 1

    if distance_from_52w_high is not None:
        if distance_from_52w_high > -5:
            score += 2
        elif distance_from_52w_high > -15:
            score += 1

    if price_change_pct is not None:
        if price_change_pct > 3:
            score += 1
        elif price_change_pct > 0:
            score += 0.5

    if rsi is not None:
        if rsi > 70:
            score += 3
        elif 55 <= rsi <= 70:
            score += 2
        elif 45 <= rsi < 55:
            score += 1

    if volume_trend is not None and isinstance(volume_trend, dict):
        if volume_trend.get('trend') == 'higher':
            score += 1
        elif volume_trend.get('trend') == 'lower':
            score -= 1

    return max(0, min(score, 10))


def calculate_valuation_score(pe_ratio, target_price, current_price):
    score = 5

    if pe_ratio is not None:
        if pe_ratio < 15:
            score += 3
        elif pe_ratio < 25:
            score += 2
        elif pe_ratio < 40:
            score += 1
        elif pe_ratio > 80:
            score -= 3
        elif pe_ratio > 50:
            score -= 2
        elif pe_ratio > 35:
            score -= 1

    if target_price is not None and current_price is not None and current_price != 0:
        upside = ((target_price - current_price) / current_price) * 100

        if upside > 30:
            score += 2
        elif upside > 15:
            score += 1
        elif upside < -10:
            score -= 2
        elif upside < 0:
            score -= 1

    return max(0, min(score, 10))


def classify_market_profile(current_price, ma50, ma200, rsi, volatility_30d, distance_from_52w_high):
    above_ma50 = ma50 is not None and current_price is not None and current_price > ma50
    above_ma200 = ma200 is not None and current_price is not None and current_price > ma200
    near_ma50 = ma50 is not None and current_price is not None and abs(current_price - ma50) / ma50 < 0.03
    near_ma200 = ma200 is not None and current_price is not None and abs(current_price - ma200) / ma200 < 0.05

    if above_ma50 and above_ma200 and rsi is not None and rsi > 70:
        return "Extended Momentum"

    if above_ma50 and above_ma200 and rsi is not None and 50 <= rsi <= 70:
        return "Healthy Uptrend"

    if rsi is not None and rsi < 35 and volatility_30d is not None and volatility_30d > 40:
        return "Oversold Bounce Setup"

    if not above_ma50 and not above_ma200 and rsi is not None and rsi < 40:
        return "Bearish Breakdown"

    if near_ma50 and near_ma200 and rsi is not None and 40 <= rsi <= 60:
        return "Sideways Consolidation"

    if not above_ma50 and not above_ma200:
        return "Bearish Breakdown"

    if volatility_30d is not None and volatility_30d > 55:
        return "Extended Momentum"

    if distance_from_52w_high is not None and distance_from_52w_high > -5:
        return "Healthy Uptrend"

    return "Mixed / Neutral Setup"


def conservative_fit(risk_score, momentum_score, valuation_score, volatility_30d, rsi, current_price, ma200):
    score = 6
    score -= risk_score * 0.5

    if volatility_30d is not None and volatility_30d > 40:
        score -= 2

    if current_price is not None and ma200 is not None and current_price < ma200:
        score -= 2

    if rsi is not None and (rsi > 70 or rsi < 30):
        score -= 1

    if valuation_score is not None and valuation_score >= 6:
        score += 1

    if momentum_score >= 5 and current_price is not None and ma200 is not None and current_price > ma200:
        score += 1

    return round(max(0, min(score, 10)), 1)


def balanced_fit(risk_score, momentum_score, valuation_score):
    score = 4

    if risk_score <= 4:
        score += 2
    elif risk_score >= 7:
        score -= 1

    if momentum_score >= 5:
        score += 2
    elif momentum_score < 3:
        score -= 1

    if valuation_score is not None and valuation_score >= 5:
        score += 1

    return round(max(0, min(score, 10)), 1)


def aggressive_growth_fit(risk_score, momentum_score, valuation_score, rsi, volatility_30d, distance_from_52w_high, current_price, ma50, ma200):
    score = 4

    if momentum_score >= 6:
        score += 3

    if current_price is not None and ma50 is not None and ma200 is not None and current_price > ma50 and current_price > ma200:
        score += 1

    if distance_from_52w_high is not None and distance_from_52w_high > -15:
        score += 1

    if valuation_score is not None and valuation_score <= 4:
        score += 1

    if rsi is not None and rsi > 80:
        score -= 1

    if volatility_30d is not None and volatility_30d > 60:
        score -= 1

    if risk_score > 7:
        score -= 1

    return round(max(0, min(score, 10)), 1)


def momentum_trader_fit(risk_score, momentum_score, rsi, current_price, ma50, ma200, volume_trend, distance_from_52w_high):
    score = momentum_score

    if current_price is not None and ma50 is not None and current_price > ma50:
        score += 1

    if current_price is not None and ma200 is not None and current_price > ma200:
        score += 1

    if distance_from_52w_high is not None and distance_from_52w_high > -10:
        score += 1

    if rsi is not None and 55 <= rsi <= 70:
        score += 1
    elif rsi is not None and rsi > 75:
        score -= 1

    if volume_trend is not None and isinstance(volume_trend, dict):
        if volume_trend.get('trend') == 'higher':
            score += 1
        elif volume_trend.get('trend') == 'lower':
            score -= 1

    if risk_score > 7:
        score -= 1

    return round(max(0, min(score, 10)), 1)


def long_term_compounder_fit(risk_score, momentum_score, valuation_score, current_price, ma200, volatility_30d):
    score = 5

    if ma200 is not None and current_price is not None and current_price > ma200:
        score += 2

    if risk_score <= 5:
        score += 2

    if valuation_score is not None and valuation_score >= 5:
        score += 1

    if volatility_30d is not None and volatility_30d > 50:
        score -= 2

    if momentum_score < 3:
        score -= 1

    return round(max(0, min(score, 10)), 1)


def value_hunter_fit(risk_score, valuation_score, rsi, target_price, current_price, ma200):
    score = valuation_score if valuation_score is not None else 4

    if current_price is not None and ma200 is not None and current_price < ma200:
        score += 1

    if rsi is not None and rsi < 35:
        score += 1
    elif rsi is not None and rsi > 70:
        score -= 1

    if target_price is not None and current_price is not None and current_price != 0:
        upside = ((target_price - current_price) / current_price) * 100
        if upside > 20:
            score += 1
        elif upside > 10:
            score += 0.5

    if risk_score > 7:
        score -= 2

    if current_price is not None and ma200 is not None and current_price > ma200:
        score -= 1

    return round(max(0, min(score, 10)), 1)


def calculate_strategy_engine(
    selected_strategy,
    current_price,
    price_change_pct,
    pe_ratio,
    target_price,
    ma50,
    ma200,
    rsi,
    volatility_30d,
    distance_from_52w_high,
    volume_trend=None
):
    risk_score = calculate_risk_score(
        pe_ratio, rsi, volatility_30d, current_price, ma50, ma200, distance_from_52w_high
    )

    momentum_score = calculate_momentum_score(
        current_price, price_change_pct, ma50, ma200, rsi, distance_from_52w_high, volume_trend
    )

    valuation_score = calculate_valuation_score(
        pe_ratio, target_price, current_price
    )

    market_profile = classify_market_profile(
        current_price, ma50, ma200, rsi, volatility_30d, distance_from_52w_high
    )

    if selected_strategy == "Conservative Investor":
        strategy_fit_score = conservative_fit(risk_score, momentum_score, valuation_score, volatility_30d, rsi, current_price, ma200)
    elif selected_strategy == "Balanced Investor":
        strategy_fit_score = balanced_fit(risk_score, momentum_score, valuation_score)
    elif selected_strategy == "Aggressive Growth":
        strategy_fit_score = aggressive_growth_fit(risk_score, momentum_score, valuation_score, rsi, volatility_30d, distance_from_52w_high, current_price, ma50, ma200)
    elif selected_strategy == "Momentum Trader":
        strategy_fit_score = momentum_trader_fit(risk_score, momentum_score, rsi, current_price, ma50, ma200, volume_trend, distance_from_52w_high)
    elif selected_strategy == "Long-Term Compounder":
        strategy_fit_score = long_term_compounder_fit(risk_score, momentum_score, valuation_score, current_price, ma200, volatility_30d)
    elif selected_strategy == "Value Hunter":
        strategy_fit_score = value_hunter_fit(risk_score, valuation_score, rsi, target_price, current_price, ma200)
    else:
        strategy_fit_score = balanced_fit(risk_score, momentum_score, valuation_score)

    return {
        "risk_score": risk_score,
        "momentum_score": momentum_score,
        "valuation_score": valuation_score,
        "strategy_fit_score": strategy_fit_score,
        "market_profile": market_profile,
    }


def build_market_context(ticker, info, current_price, price_change_pct, market_cap, pe_ratio, revenue, revenue_growth, target_price, recommendation,
                         rsi=None, ma_50=None, ma_200=None, volume_trend=None, volatility_30d=None,
                         distance_from_52_week_high_pct=None, distance_from_52_week_low_pct=None, news_headlines=None):
    """Create compact context to send to OpenAI."""
    return {
        "ticker": ticker,
        "company": info.get("longName", ticker),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "current_price": current_price,
        "one_day_price_change_percent": price_change_pct,
        "market_cap": market_cap,
        "trailing_pe": pe_ratio,
        "forward_pe": info.get("forwardPE"),
        "total_revenue": revenue,
        "revenue_growth": revenue_growth,
        "profit_margins": info.get("profitMargins"),
        "gross_margins": info.get("grossMargins"),
        "debt_to_equity": info.get("debtToEquity"),
        "free_cashflow": info.get("freeCashflow"),
        "analyst_target_mean_price": target_price,
        "recommendation": recommendation,
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "distance_from_52_week_high_pct": distance_from_52_week_high_pct,
        "distance_from_52_week_low_pct": distance_from_52_week_low_pct,
        "rsi": rsi,
        "ma_50": ma_50,
        "ma_200": ma_200,
        "volume_trend": volume_trend,
        "volatility_30d": volatility_30d,
        "news_headlines": news_headlines or [],
        "business_summary": info.get("longBusinessSummary"),
    }


def fallback_analysis(ticker, company_name, data):
    """Used if OpenAI is unavailable, so the app still works."""
    price_change = data.get("price_change_pct", 0)
    pe_ratio = data.get("pe_ratio", None)
    market_cap = data.get("market_cap", None)
    sentiment = get_sentiment(price_change)

    if sentiment == "Bullish":
        opener = f"{company_name} ({ticker}) is showing positive short-term momentum with a {price_change:.2f}% move from the previous close."
    elif sentiment == "Bearish":
        opener = f"{company_name} ({ticker}) is under short-term pressure with a {abs(price_change):.2f}% decline from the previous close."
    else:
        opener = f"{company_name} ({ticker}) is trading relatively flat, suggesting a more balanced near-term setup."

    analysis = []
    analysis.append(f"Market Setup Summary: {opener}")
    analysis.append("Strategy Alignment: The current profile is neutral, with only basic momentum and valuation data available in fallback mode.")
    analysis.append("Tactical Considerations: Confirm the trend with live volume, moving averages, and volatility before taking a position.")
    analysis.append("Risk Assessment: The lack of OpenAI detail means focus on downside protection, especially if price remains below the 200-day average.")

    return {
        "sentiment": sentiment,
        "analysis": "\n\n".join(analysis),
        "takeaways": [
            "Verify trend direction with live volume and moving-average alignment.",
            "Prioritize capital preservation until a cleaner setup emerges.",
            "Use the report only for educational research, not as a transaction signal."
        ],
        "risk_score": "5/10",
        "momentum_score": "5/10",
        "valuation_score": "5/10",
        "strategy_fit_score": "5/10",
        "market_profile": "Mixed / Neutral Setup",
    }


def generate_openai_report(ticker, focus, market_context):
    """
    Generate a structured stock report with OpenAI.
    Returns a dict with sentiment, analysis, and takeaways.
    """
    if client is None:
        report = fallback_analysis(
            ticker,
            market_context.get("company", ticker),
            {
                "price_change_pct": market_context.get("one_day_price_change_percent") or 0,
                "pe_ratio": market_context.get("trailing_pe") or "N/A",
                "market_cap": market_context.get("market_cap"),
            },
        )
        return report

    strategy_prompts = {
        "Conservative Investor": "Prioritize downside protection, stable trend alignment, lower volatility, and disciplined capital preservation.",
        "Balanced Investor": "Balance trend quality, valuation, and risk control for a measured exposure.",
        "Aggressive Growth": "Prioritize momentum, high-upside continuation, and tactical risk management while accepting elevated volatility.",
        "Momentum Trader": "Prioritize trend continuation, moving average alignment, strong RSI, volume confirmation, and price strength near highs.",
        "Long-Term Compounder": "Prioritize durable earnings quality, cash flow stability, and constructive long-term trend behavior.",
        "Value Hunter": "Prioritize relative valuation, mean reversion potential, cash flow stability, and lower downside risk.",
    }

    strategy_guidance = strategy_prompts.get(focus, "Focus on the current market setup with a balanced view of valuation, momentum, and risk.")

    system_prompt = """
You are an institutional equity research analyst writing a concise, structured strategy note.
The report should be professional, evidence-based, and tactical.

Guidelines:
- Do NOT say buy, sell, or short.
- Avoid repetitive language and generic filler.
- Use clear section headings in the analysis text: Market Setup Summary, Strategy Alignment, Tactical Considerations, Risk Assessment.
- Keep each section short and easy to scan.
- Reference only the provided data.
- Do not invent price levels, support, or resistance unless the data clearly supports them.
- When indicators conflict, explain the conflict directly.
- Prefer specific market commentary over broad narrative.
- Return ONLY valid JSON with no markdown.
"""

    user_prompt = f"""
Write a concise equity note on {ticker} for near-term trading.

Focus area: {focus}
Strategy guidance: {strategy_guidance}

Market context:
{json.dumps(market_context, indent=2)}

Your report must follow this structure:
1. Market Setup Summary: 2-3 concise sentences covering trend condition, momentum condition, volatility, and market profile classification.
2. Strategy Alignment: explain why the stock fits or does not fit the selected strategy, referencing RSI, moving averages, volatility, proximity to the 52-week high, momentum, and trend strength.
3. Tactical Considerations: provide 2-3 specific observations on entry conditions, overextension, confirmation signals, caution areas, or ideal trader behavior.
4. Risk Assessment: describe downside risks, volatility concerns, trend weakness, overbought/oversold conditions, and key technical threats.

If news is relevant, mention the most material headline and why it matters; otherwise say news is not a material factor.

Return this exact JSON:
{{
  "sentiment": "Bullish" | "Bearish" | "Neutral",
  "analysis": "A four-section note with the headings above, separated by newlines.",
  "takeaways": [
    "Takeaway 1 - specific and actionable",
    "Takeaway 2 - specific and actionable",
    "Takeaway 3 - specific and actionable"
  ]
}}
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
    )

    raw = response.choices[0].message.content
    report = json.loads(raw)

    # Defensive cleanup so frontend always receives what it expects.
    sentiment = report.get("sentiment", "Neutral")
    if sentiment not in ["Bullish", "Bearish", "Neutral"]:
        sentiment = "Neutral"

    takeaways = report.get("takeaways", [])
    if not isinstance(takeaways, list):
        takeaways = [str(takeaways)]

    return {
        "sentiment": sentiment,
        "analysis": report.get("analysis", "No analysis returned."),
        "takeaways": takeaways[:5],
    }


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json or {}
        ticker = data.get("ticker", "").upper().strip()
        focus = data.get("focus", "General Overview")

        if not ticker:
            return jsonify({"error": "No ticker provided"}), 400

        # Fetch stock data from Yahoo Finance for now.
        stock = yf.Ticker(ticker)
        info = stock.info

        # Validate ticker exists.
        if not info.get("longName"):
            return jsonify({"error": f'Ticker "{ticker}" not found'}), 404

        current_price = safe_float(info.get("currentPrice")) or safe_float(info.get("regularMarketPrice")) or 0
        previous_close = safe_float(info.get("previousClose")) or current_price
        price_change = current_price - previous_close if previous_close else 0
        price_change_pct = (price_change / previous_close * 100) if previous_close else 0

        market_cap = info.get("marketCap")
        pe_ratio = info.get("trailingPE")
        revenue = info.get("totalRevenue")
        revenue_growth = info.get("revenueGrowth")
        target_price = info.get("targetMeanPrice")
        recommendation = (info.get("recommendationKey") or "hold").upper()

        history = stock.history(period="1y", interval="1d")
        close_series = history["Close"] if "Close" in history else history["close"] if "close" in history else None
        volume_series = history["Volume"] if "Volume" in history else history["volume"] if "volume" in history else None

        rsi = compute_rsi(close_series) if close_series is not None else None
        ma_50 = moving_average(close_series, 50) if close_series is not None else None
        ma_200 = moving_average(close_series, 200) if close_series is not None else None
        volatility_30d = compute_volatility(close_series, 30) if close_series is not None else None
        volume_trend = compute_volume_trend(volume_series) if volume_series is not None else None

        chart_dates = []
        chart_prices = []
        chart_ma50 = []
        chart_ma200 = []
        if close_series is not None and not close_series.empty:
            close_clean = close_series.dropna()
            ma50_values = close_clean.rolling(window=50, min_periods=50).mean()
            ma200_values = close_clean.rolling(window=200, min_periods=200).mean()
            for idx, price in close_clean.items():
                chart_dates.append(idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx))
                chart_prices.append(round(float(price), 2))
                ma50_val = ma50_values.loc[idx]
                ma200_val = ma200_values.loc[idx]
                chart_ma50.append(None if pd.isna(ma50_val) else round(float(ma50_val), 2))
                chart_ma200.append(None if pd.isna(ma200_val) else round(float(ma200_val), 2))

        fifty_two_week_high = safe_float(info.get("fiftyTwoWeekHigh"))
        fifty_two_week_low = safe_float(info.get("fiftyTwoWeekLow"))
        distance_from_52_week_high_pct = None
        distance_from_52_week_low_pct = None
        if fifty_two_week_high and current_price:
            distance_from_52_week_high_pct = round((fifty_two_week_high - current_price) / fifty_two_week_high * 100, 2)
        if fifty_two_week_low and current_price:
            distance_from_52_week_low_pct = round((current_price - fifty_two_week_low) / fifty_two_week_low * 100, 2)

        company_name = info.get('longName', ticker)
        if isinstance(company_name, str):
            company_name = company_name.replace('"', '')

        search_terms = [f'"{company_name}"', ticker]
        news_query = ' OR '.join(term for term in search_terms if term)
        news_headlines = fetch_news_headlines(news_query, limit=5, company_name=company_name, ticker=ticker)

        market_context = build_market_context(
            ticker=ticker,
            info=info,
            current_price=current_price,
            price_change_pct=price_change_pct,
            market_cap=market_cap,
            pe_ratio=pe_ratio,
            revenue=revenue,
            revenue_growth=revenue_growth,
            target_price=target_price,
            recommendation=recommendation,
            rsi=rsi,
            ma_50=ma_50,
            ma_200=ma_200,
            volume_trend=volume_trend,
            volatility_30d=volatility_30d,
            distance_from_52_week_high_pct=distance_from_52_week_high_pct,
            distance_from_52_week_low_pct=distance_from_52_week_low_pct,
            news_headlines=news_headlines,
        )

        ai_report = generate_openai_report(ticker, focus, market_context)

        selected_strategy = focus if focus in STRATEGY_OPTIONS else "Balanced Investor"
        strategy_engine = calculate_strategy_engine(
            selected_strategy,
            current_price,
            price_change_pct,
            pe_ratio,
            target_price,
            ma_50,
            ma_200,
            rsi,
            volatility_30d,
            distance_from_52_week_high_pct,
            volume_trend,
        )

        analysis_data = {
            "ticker": ticker,
            "company": info.get("longName", ticker),
            "sentiment": ai_report["sentiment"],
            "risk_score": f"{strategy_engine['risk_score']}/10",
            "momentum_score": f"{strategy_engine['momentum_score']}/10",
            "valuation_score": f"{strategy_engine['valuation_score']}/10",
            "strategy_fit_score": f"{strategy_engine['strategy_fit_score']}/10",
            "market_profile": strategy_engine["market_profile"],
            "metrics": [
                {"label": "Current Price", "value": f"${current_price:.2f}"},
                {"label": "Market Cap", "value": format_large_number(market_cap)},
                {"label": "P/E Ratio", "value": f"{pe_ratio:.2f}" if pe_ratio else "N/A"},
                {"label": "Revenue", "value": format_large_number(revenue)},
                {"label": "Price Change (1D)", "value": f"{price_change_pct:+.2f}%"},
                {"label": "50-Day MA", "value": f"${ma_50:.2f}" if ma_50 else "N/A"},
                {"label": "200-Day MA", "value": f"${ma_200:.2f}" if ma_200 else "N/A"},
                {"label": "RSI", "value": f"{rsi:.1f}" if rsi else "N/A"},
                {"label": "30d Volatility", "value": f"{volatility_30d:.2f}%" if volatility_30d else "N/A"},
                {"label": "Distance from 52w High", "value": f"{distance_from_52_week_high_pct:+.2f}%" if distance_from_52_week_high_pct is not None else "N/A"},
                {"label": "Price Target", "value": f"${target_price:.2f}" if target_price else "N/A"},
            ],
            "chart": {
                "dates": chart_dates,
                "prices": chart_prices,
                "ma50": chart_ma50,
                "ma200": chart_ma200,
            },
            "news_headlines": news_headlines,
            "analysis": ai_report["analysis"],
            "takeaways": ai_report["takeaways"],
            "source": {
                "market_data": "Yahoo Finance via yfinance",
                "analysis": "OpenAI" if client else "Fallback local template",
                "focus": focus,
            },
        }

        return jsonify(analysis_data)

    except json.JSONDecodeError:
        return jsonify({"error": "OpenAI returned invalid JSON. Try again."}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai-test", methods=["GET"])
def ai_test():
    """Quick route to confirm your OpenAI key works."""
    if client is None:
        return jsonify({"ok": False, "error": "Missing OPENAI_API_KEY in .env"}), 500

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": "Return JSON only: {\"ok\": true, \"message\": \"OpenAI route works\"}"}
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return jsonify(json.loads(response.choices[0].message.content))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    load_dotenv(BASE_DIR / ".env")
    return jsonify({
        "status": "ok",
        "message": "Backend is running",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "news_api_configured": bool(os.getenv("NEWSAPI_KEY")),
        "model": OPENAI_MODEL,
    })


if __name__ == "__main__":
    print("=" * 60)
    print("MarketLens Backend Server")
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print("OpenAI configured:", bool(OPENAI_API_KEY))
    print("Model:", OPENAI_MODEL)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=True, port=5000)
