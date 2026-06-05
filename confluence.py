"""
Multi-timeframe confluence scoring for JARVIS.

For each ticker, fetches data at three timeframes (1D, 4H, 1H) and assigns a
directional bias to each based on trend, momentum, and ICT structure.
A confluence score of 3/3 (all TFs bullish) is a strong signal; 1/3 or 0/3
is a reason to stay flat even if the ML model fires.

Bias per TF is determined by majority vote of five sub-signals:
  1. Price vs MA50
  2. RSI > 50 (bullish) / < 50 (bearish)
  3. MACD above its signal line
  4. HTF_Bullish flag (from add_features)
  5. BOS_up flag (break of structure)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from lstm_model import add_features

# ── Timeframes to analyse ──────────────────────────────────────────────────────
# Each entry: (label, yfinance interval, fetch period, optional resample target)
TIMEFRAMES = [
    ("1D",  "1d",  "1y",  None),
    ("4H",  "60m", "6mo", "4h"),
    ("1H",  "60m", "3mo", None),
]


def _fetch(ticker: str, interval: str, period: str, resample: str | None) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if resample:
            df = df.resample(resample).agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum",
            }).dropna()
        if len(df) < 30:
            return None
        return df
    except Exception:
        return None


def _bias(df: pd.DataFrame) -> dict:
    """Return bias dict for a single timeframe dataframe."""
    try:
        df = add_features(df.copy())
    except Exception:
        pass

    last = df.iloc[-1]
    close = float(last["Close"])

    scores = []

    # 1. Price vs MA50
    ma50 = df["Close"].rolling(50).mean().iloc[-1]
    if not pd.isna(ma50):
        scores.append(1 if close > float(ma50) else -1)

    # 2. RSI
    rsi = last.get("RSI", float("nan"))
    if not pd.isna(rsi):
        scores.append(1 if rsi > 52 else (-1 if rsi < 48 else 0))

    # 3. MACD vs signal
    macd = last.get("MACD", float("nan"))
    macd_sig = last.get("MACD_Signal", float("nan"))
    if not pd.isna(macd) and not pd.isna(macd_sig):
        scores.append(1 if macd > macd_sig else -1)

    # 4. HTF bullish flag
    htf = last.get("HTF_Bullish", float("nan"))
    if not pd.isna(htf):
        scores.append(1 if htf == 1 else -1)

    # 5. BOS up
    bos = last.get("bos_up", float("nan"))
    if not pd.isna(bos):
        scores.append(1 if bos == 1 else 0)

    if not scores:
        return {"bias": "NEUTRAL", "score": 0, "signals": 0}

    total = sum(scores)
    n = len(scores)
    ratio = total / n

    if ratio >= 0.4:
        bias = "BULLISH"
    elif ratio <= -0.4:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return {
        "bias": bias,
        "score": round(ratio, 2),
        "signals": n,
        "rsi": round(float(last.get("RSI", 0) or 0), 1),
        "macd_cross": bool(not pd.isna(macd) and not pd.isna(macd_sig) and macd > macd_sig),
        "above_ma50": bool(not pd.isna(ma50) and close > float(ma50)),
        "close": round(close, 2),
    }


def analyse(ticker: str) -> dict:
    """
    Returns multi-TF confluence dict for ticker.
    Always returns a dict — errors are surfaced inside, never raised.
    """
    details = {}
    bull = 0
    bear = 0
    neutral = 0

    for label, interval, period, resample in TIMEFRAMES:
        df = _fetch(ticker, interval, period, resample)
        if df is None:
            details[label] = {"bias": "NEUTRAL", "score": 0, "signals": 0, "error": "insufficient data"}
            neutral += 1
            continue
        b = _bias(df)
        details[label] = b
        if b["bias"] == "BULLISH":
            bull += 1
        elif b["bias"] == "BEARISH":
            bear += 1
        else:
            neutral += 1

    total_tfs = len(TIMEFRAMES)
    bull_pct = round(bull / total_tfs * 100)
    bear_pct = round(bear / total_tfs * 100)

    # Overall confluence verdict
    if bull == 3:
        confluence = "STRONG BULL"
        strength = 100
    elif bull == 2:
        confluence = "BULL"
        strength = 67
    elif bear == 3:
        confluence = "STRONG BEAR"
        strength = 100
    elif bear == 2:
        confluence = "BEAR"
        strength = 67
    else:
        confluence = "MIXED"
        strength = 33

    # Trade recommendation
    if bull >= 2:
        recommendation = "LOOK FOR LONGS"
    elif bear >= 2:
        recommendation = "LOOK FOR SHORTS"
    else:
        recommendation = "STAY FLAT — mixed TFs"

    return {
        "ticker": ticker,
        "confluence": confluence,
        "strength": strength,
        "bull_tfs": bull,
        "bear_tfs": bear,
        "neutral_tfs": neutral,
        "bull_pct": bull_pct,
        "bear_pct": bear_pct,
        "recommendation": recommendation,
        "details": details,
    }
