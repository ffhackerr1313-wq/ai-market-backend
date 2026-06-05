import re
from textblob import TextBlob
from newsapi import NewsApiClient
from config import NEWS_API_KEY

# ── Init ─────────────────────────────────────────────────────────────────────
api = NewsApiClient(api_key=NEWS_API_KEY)

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

def get_sentiment(ticker: str) -> dict:
    """
    Fetch latest news for ticker and return sentiment score.
    Returns a dict with: score, label, article_count, headlines
    """
    # Map common tickers to better search terms
    search_map = {
        "RELIANCE.NS":  "Reliance Industries",
        "TCS.NS":       "TCS Tata Consultancy",
        "HDFCBANK.NS":  "HDFC Bank",
        "INFY.NS":      "Infosys stock",
        "ICICIBANK.NS": "ICICI Bank",
    }
    query = search_map.get(ticker.upper(), ticker.replace(".NS", "") + " stock")

    try:
        response = api.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            page_size=10,
        )
        articles = response.get("articles", [])
        if not articles:
            return _neutral(ticker, "No articles found")

        scores = []
        headlines = []
        for art in articles:
            title = art.get("title") or ""
            desc  = art.get("description") or ""
            text  = clean_text(f"{title}. {desc}")
            if text:
                polarity = TextBlob(text).sentiment.polarity
                scores.append(polarity)
                headlines.append({
                    "title":     title[:100],
                    "source":    art.get("source", {}).get("name", ""),
                    "sentiment": round(polarity, 3),
                    "url":       art.get("url", ""),
                })

        if not scores:
            return _neutral(ticker, "Could not parse articles")

        avg_score = round(sum(scores) / len(scores), 4)

        if avg_score >= 0.15:
            label = "BULLISH"
        elif avg_score <= -0.15:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        return {
            "ticker":        ticker,
            "sentiment_score": avg_score,
            "sentiment_label": label,
            "article_count": len(scores),
            "headlines":     headlines[:5],
            "error":         None,
        }

    except Exception as e:
        return _neutral(ticker, str(e))

def _neutral(ticker, reason):
    return {
        "ticker":          ticker,
        "sentiment_score": 0.0,
        "sentiment_label": "NEUTRAL",
        "article_count":   0,
        "headlines":       [],
        "error":           reason,
    }

def combined_signal(lstm_signal: str, lstm_conf: int, sentiment: dict) -> dict:
    """
    Combine LSTM signal with news sentiment for a stronger final signal.
    """
    s_label = sentiment["sentiment_label"]
    s_score = sentiment["sentiment_score"]

    # Agreement = stronger signal
    if lstm_signal == "BUY" and s_label == "BULLISH":
        final = "STRONG BUY"
        reason = "LSTM + News both bullish"
    elif lstm_signal == "SELL" and s_label == "BEARISH":
        final = "STRONG SELL"
        reason = "LSTM + News both bearish"
    # Disagreement = downgrade to HOLD
    elif lstm_signal == "BUY" and s_label == "BEARISH":
        final = "HOLD"
        reason = "LSTM bullish but news bearish — wait"
    elif lstm_signal == "SELL" and s_label == "BULLISH":
        final = "HOLD"
        reason = "LSTM bearish but news bullish — wait"
    # Neutral news = keep LSTM signal
    else:
        final = lstm_signal
        reason = f"LSTM signal with neutral news (score: {s_score})"

    return {
        "final_signal": final,
        "lstm_signal":  lstm_signal,
        "news_sentiment": s_label,
        "reason": reason,
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for ticker in ["RELIANCE.NS", "TCS.NS"]:
        print(f"\n{'='*45}")
        print(f"Ticker: {ticker}")
        result = get_sentiment(ticker)
        print(f"  Sentiment : {result['sentiment_label']} ({result['sentiment_score']})")
        print(f"  Articles  : {result['article_count']}")
        if result["error"]:
            print(f"  Error     : {result['error']}")
        print("  Top headlines:")
        for h in result["headlines"][:3]:
            print(f"    [{h['sentiment']:+.2f}] {h['title']} — {h['source']}")

        # Test combined signal
        combined = combined_signal("BUY", 65, result)
        print(f"  Combined  : {combined['final_signal']} — {combined['reason']}")
