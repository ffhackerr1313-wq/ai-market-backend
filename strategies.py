"""
Trading strategy engine for JARVIS.
Each strategy takes a DataFrame (daily OR hourly OHLCV) and returns a
standardised signal dict:

  {
    "signal":           "BUY" | "SELL" | "HOLD",
    "confidence":       int  0-100,
    "reason":           str,
    "scanner_signal":   str,
    "breakout_strength": str,
    # optional extras (MOMENTUM strategy):
    "breakout_up":   bool,
    "breakout_down": bool,
    "volume_spike":  bool,
    "volume_ratio":  float,
  }
"""

import numpy as np
import pandas as pd

# ── catalogue metadata ────────────────────────────────────────────────────────

STRATEGY_META: dict[str, dict] = {
    "EMA_CROSSOVER": {
        "name": "EMA Crossover",
        "description": "9/21 EMA cross on 1 h data with 50 EMA trend filter + RSI gate. "
                       "Catches intraday momentum shifts cleanly.",
        "timeframe": "1h",
        "expected_trades": "2-5 / day",
        "why_better": "Removes random breakouts; only trades confirmed trend turns.",
    },
    "SUPERTREND": {
        "name": "Supertrend (7,3)",
        "description": "ATR-based Supertrend on 1 h data. Direction flips = high-quality "
                       "trend-change entries; fewer but more reliable signals.",
        "timeframe": "1h",
        "expected_trades": "1-3 / day",
        "why_better": "Dynamic stop built-in; naturally rides winners.",
    },
    "VWAP_RSI": {
        "name": "VWAP + RSI",
        "description": "Price vs rolling VWAP with RSI(14) confirmation on 1 h data. "
                       "Blends trend-following and mean-reversion.",
        "timeframe": "1h",
        "expected_trades": "2-4 / day",
        "why_better": "VWAP is the institutional reference; aligning with it improves fill quality.",
    },
    "MOMENTUM": {
        "name": "Momentum Breakout",
        "description": "Improved 20-bar high/low breakout + volume-spike filter on daily data. "
                       "Better RSI / EMA50 guard than the original version.",
        "timeframe": "1d",
        "expected_trades": "0-2 swing",
        "why_better": "Added EMA50 trend filter reduces false breakouts.",
    },
    "ICT_SMC": {
        "name": "ICT / Smart Money",
        "description": "BOS + FVG + Order Block confluence on daily data. "
                       "Institutional structure-based entries.",
        "timeframe": "1d",
        "expected_trades": "0-1 swing",
        "why_better": "Aligned with institutional order flow logic.",
    },
}

DEFAULT_STRATEGY = "EMA_CROSSOVER"

# ── shared helpers ────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    series = 100 - (100 / (1 + rs))
    return float(series.iloc[-1]) if not pd.isna(series.iloc[-1]) else 50.0


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── public entry point ────────────────────────────────────────────────────────

def run(df: pd.DataFrame, strategy: str) -> dict:
    """Generate a signal for the given DataFrame using the named strategy."""
    fn = _DISPATCH.get(strategy, _ema_crossover)
    try:
        return fn(df)
    except Exception as exc:
        return {
            "signal": "HOLD", "confidence": 40,
            "reason": f"strategy error: {exc}",
            "scanner_signal": "NEUTRAL", "breakout_strength": "WEAK",
        }


# ── EMA CROSSOVER ─────────────────────────────────────────────────────────────

def _ema_crossover(df: pd.DataFrame) -> dict:
    close = df["Close"]
    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    rsi   = _rsi(close)

    e9   = float(ema9.iloc[-1]);  e9p  = float(ema9.iloc[-2])
    e21  = float(ema21.iloc[-1]); e21p = float(ema21.iloc[-2])
    price = float(close.iloc[-1])
    above50 = price > float(ema50.iloc[-1])
    spread  = abs(e9 - e21) / price * 100   # % separation

    cross_up = (e9 > e21) and (e9p <= e21p)
    cross_dn = (e9 < e21) and (e9p >= e21p)

    if cross_up and above50 and rsi < 72:
        sig, conf = "BUY",  min(88, 65 + spread * 18)
        scanner, strength = "EMA CROSS BUY", "STRONG" if spread > 0.15 else "MEDIUM"
        reason = f"EMA9 crossed above EMA21 |above EMA50 |RSI {rsi:.0f}"
    elif cross_dn and not above50 and rsi > 28:
        sig, conf = "SELL", min(88, 65 + spread * 18)
        scanner, strength = "EMA CROSS SELL", "STRONG" if spread > 0.15 else "MEDIUM"
        reason = f"EMA9 crossed below EMA21 |below EMA50 |RSI {rsi:.0f}"
    elif e9 > e21 and above50 and rsi < 65:
        sig, conf = "BUY",  57
        scanner, strength = "BULLISH TREND", "MEDIUM"
        reason = f"Bullish EMA stack |above EMA50 |RSI {rsi:.0f}"
    elif e9 < e21 and not above50 and rsi > 35:
        sig, conf = "SELL", 57
        scanner, strength = "BEARISH TREND", "MEDIUM"
        reason = f"Bearish EMA stack |below EMA50 |RSI {rsi:.0f}"
    else:
        sig, conf = "HOLD", 38
        scanner, strength = "NEUTRAL", "WEAK"
        reason = f"Mixed EMAs |await crossover |RSI {rsi:.0f}"

    return {"signal": sig, "confidence": round(conf), "reason": reason,
            "scanner_signal": scanner, "breakout_strength": strength}


# ── SUPERTREND ────────────────────────────────────────────────────────────────

def _supertrend(df: pd.DataFrame, period: int = 7, mult: float = 3.0) -> dict:
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n     = len(close)

    # ATR (simple rolling)
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low  - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.zeros(n)
    for i in range(n):
        if i < period:
            atr[i] = tr[:i+1].mean()
        else:
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    hl2   = (high + low) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    direction = np.ones(n, dtype=int)

    for i in range(1, n):
        # tighten bands
        upper[i] = min(upper[i], upper[i-1]) if close[i-1] <= upper[i-1] else upper[i]
        lower[i] = max(lower[i], lower[i-1]) if close[i-1] >= lower[i-1] else lower[i]
        if close[i] > upper[i-1]:
            direction[i] = 1
        elif close[i] < lower[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

    rsi  = _rsi(df["Close"])
    bull = direction[-1] == 1
    flip = direction[-1] != direction[-2]

    if flip and bull:
        sig, conf, scanner, strength = "BUY",  86, "SUPERTREND BULL FLIP", "STRONG"
        reason = f"Supertrend flipped bullish |RSI {rsi:.0f}"
    elif flip and not bull:
        sig, conf, scanner, strength = "SELL", 86, "SUPERTREND BEAR FLIP", "STRONG"
        reason = f"Supertrend flipped bearish |RSI {rsi:.0f}"
    elif bull:
        sig, conf, scanner, strength = "BUY",  62, "SUPERTREND BULL",      "MEDIUM"
        reason = f"Supertrend bullish |trend intact |RSI {rsi:.0f}"
    else:
        sig, conf, scanner, strength = "SELL", 62, "SUPERTREND BEAR",      "MEDIUM"
        reason = f"Supertrend bearish |trend intact |RSI {rsi:.0f}"

    return {"signal": sig, "confidence": conf, "reason": reason,
            "scanner_signal": scanner, "breakout_strength": strength}


# ── VWAP + RSI ────────────────────────────────────────────────────────────────

def _vwap_rsi(df: pd.DataFrame) -> dict:
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"].replace(0, np.nan).ffill()

    typical = (high + low + close) / 3
    window  = 20
    vwap    = (typical * vol).rolling(window).sum() / vol.rolling(window).sum()

    price = float(close.iloc[-1])
    vw    = float(vwap.iloc[-1]) if not pd.isna(vwap.iloc[-1]) else price
    rsi_s = _rsi_series(close)
    rsi   = float(rsi_s.iloc[-1])  if not pd.isna(rsi_s.iloc[-1])  else 50.0
    rsi_p = float(rsi_s.iloc[-2]) if not pd.isna(rsi_s.iloc[-2]) else 50.0

    above    = price > vw
    gap_pct  = (price - vw) / vw * 100
    rsi_xu50 = (rsi > 50) and (rsi_p <= 50)   # RSI crossed above 50
    rsi_xd50 = (rsi < 50) and (rsi_p >= 50)   # RSI crossed below 50

    if above and rsi_xu50 and rsi < 70:
        sig, conf = "BUY",  83
        scanner, strength = "VWAP BULL", "STRONG"
        reason = f"Above VWAP |RSI crossed 50 ({rsi:.0f}) |momentum building"
    elif not above and rsi_xd50 and rsi > 30:
        sig, conf = "SELL", 83
        scanner, strength = "VWAP BEAR", "STRONG"
        reason = f"Below VWAP |RSI crossed below 50 ({rsi:.0f}) |selling pressure"
    elif above and rsi > 55 and gap_pct > 0.2:
        sig, conf = "BUY",  63
        scanner, strength = "VWAP TREND", "MEDIUM"
        reason = f"Above VWAP +{gap_pct:.1f}% |RSI {rsi:.0f} bullish"
    elif not above and rsi < 45 and gap_pct < -0.2:
        sig, conf = "SELL", 63
        scanner, strength = "VWAP TREND", "MEDIUM"
        reason = f"Below VWAP {gap_pct:.1f}% |RSI {rsi:.0f} bearish"
    elif above and rsi < 38:
        sig, conf = "BUY",  70
        scanner, strength = "VWAP REVERT", "MEDIUM"
        reason = f"Above VWAP |RSI oversold {rsi:.0f} — pullback buy"
    else:
        sig, conf = "HOLD", 38
        scanner, strength = "NEUTRAL", "WEAK"
        reason = f"Price near VWAP ({gap_pct:+.1f}%) |RSI {rsi:.0f} neutral"

    return {"signal": sig, "confidence": conf, "reason": reason,
            "scanner_signal": scanner, "breakout_strength": strength}


# ── MOMENTUM BREAKOUT (improved) ──────────────────────────────────────────────

def _momentum_breakout(df: pd.DataFrame) -> dict:
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    price     = float(close.iloc[-1])
    high_20   = float(high.rolling(20).max().iloc[-2])
    low_20    = float(low.rolling(20).min().iloc[-2])
    avg_vol   = float(vol.rolling(20).mean().iloc[-1])
    cur_vol   = float(vol.iloc[-1])
    vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
    vol_spike = vol_ratio > 1.8
    rsi       = _rsi(close)
    ema50     = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    bk_up     = price > high_20
    bk_dn     = price < low_20

    if bk_up and vol_spike and rsi < 76:
        sig, conf = "BUY",  81
        scanner, strength = "MOMENTUM BUY", "STRONG"
        reason = f"20-bar breakout |vol {vol_ratio:.1f}× |RSI {rsi:.0f}"
    elif bk_up and price > ema50:
        sig, conf = "BUY",  63
        scanner, strength = "BREAKOUT BUY", "MEDIUM"
        reason = f"20-bar breakout |above EMA50 |RSI {rsi:.0f}"
    elif bk_dn and vol_spike and rsi > 24:
        sig, conf = "SELL", 81
        scanner, strength = "BREAKDOWN SELL", "STRONG"
        reason = f"20-bar breakdown |vol {vol_ratio:.1f}× |RSI {rsi:.0f}"
    elif bk_dn and price < ema50:
        sig, conf = "SELL", 63
        scanner, strength = "BREAKDOWN", "MEDIUM"
        reason = f"20-bar breakdown |below EMA50 |RSI {rsi:.0f}"
    else:
        sig, conf = "HOLD", 40
        scanner, strength = "NEUTRAL", "WEAK"
        reason = f"No breakout |RSI {rsi:.0f} |vol {vol_ratio:.1f}×"

    return {
        "signal": sig, "confidence": conf, "reason": reason,
        "scanner_signal": scanner, "breakout_strength": strength,
        "breakout_up": bk_up, "breakout_down": bk_dn,
        "volume_spike": vol_spike, "volume_ratio": round(vol_ratio, 2),
    }


# ── ICT / SMC ─────────────────────────────────────────────────────────────────

def _ict_smc(df: pd.DataFrame) -> dict:
    import ict as ict_mod

    close  = df["Close"]
    rsi    = _rsi(close)
    events = ict_mod.detect_events(df)

    if not events:
        return {
            "signal": "HOLD", "confidence": 38,
            "reason": "No ICT events detected recently",
            "scanner_signal": "NEUTRAL", "breakout_strength": "WEAK",
        }

    recent     = events[-6:]
    bull_evts  = [e for e in recent if e["type"] in ("BOS_UP",  "FVG_BULL", "OB_BULL")]
    bear_evts  = [e for e in recent if e["type"] in ("BOS_DOWN","FVG_BEAR", "OB_BEAR")]

    if len(bull_evts) >= 2 and rsi < 72:
        sig, conf  = "BUY",  min(88, 62 + len(bull_evts) * 7)
        scanner, strength = "ICT BULL CONFLUENCE", "STRONG"
        reason = " |".join(e["label"] for e in bull_evts[-2:])
    elif len(bear_evts) >= 2 and rsi > 28:
        sig, conf  = "SELL", min(88, 62 + len(bear_evts) * 7)
        scanner, strength = "ICT BEAR CONFLUENCE", "STRONG"
        reason = " |".join(e["label"] for e in bear_evts[-2:])
    elif bull_evts:
        sig, conf  = "BUY",  59
        scanner, strength = "ICT BULL", "MEDIUM"
        reason = bull_evts[-1]["label"] + f" |RSI {rsi:.0f}"
    elif bear_evts:
        sig, conf  = "SELL", 59
        scanner, strength = "ICT BEAR", "MEDIUM"
        reason = bear_evts[-1]["label"] + f" |RSI {rsi:.0f}"
    else:
        sig, conf  = "HOLD", 38
        scanner, strength = "NEUTRAL", "WEAK"
        reason = "ICT sweep/neutral events only"

    return {"signal": sig, "confidence": conf, "reason": reason,
            "scanner_signal": scanner, "breakout_strength": strength}


# ── dispatch ──────────────────────────────────────────────────────────────────

_DISPATCH: dict = {
    "EMA_CROSSOVER": _ema_crossover,
    "SUPERTREND":    _supertrend,
    "VWAP_RSI":      _vwap_rsi,
    "MOMENTUM":      _momentum_breakout,
    "ICT_SMC":       _ict_smc,
}
