"""
Walk-forward validation for JARVIS.

A single 80/20 train-test split overfits the split point — the model "knows"
the future indirectly through feature engineering on the full dataset. Walk-
forward is the honest alternative: train on a fixed rolling window, test on
the immediately following period, slide forward, repeat. No lookahead.

Default: 252-bar training window (~1 year), 21-bar test window (~1 month),
         step = test_bars (non-overlapping test periods).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from lstm_model import get_data, add_features

FEATURE_COLS = [
    "Returns", "Volatility", "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Width", "BB_Pos", "Momentum", "ROC", "Volume_Ratio",
    "MA10", "MA20", "MA50", "HTF_Bullish", "sweep_low", "bos_up",
]

SL_PCT = 0.03   # 3 % stop-loss
TP_PCT = 0.06   # 6 % take-profit


def _trade_window(prices: np.ndarray, signals: np.ndarray) -> float:
    """Simulate simple long-only trading on one test window. Returns % return."""
    capital = 100.0
    cash = capital
    shares = 0.0
    entry = 0.0
    for i in range(len(signals)):
        p = float(prices[i])
        s = int(signals[i])
        if s == 1 and shares == 0:
            shares = cash / p
            entry = p
            cash = 0.0
        elif shares > 0:
            change = (p - entry) / entry
            if s == -1 or change >= TP_PCT or change <= -SL_PCT:
                cash = shares * p
                shares = 0.0
    final = cash + shares * float(prices[-1])
    return (final - capital) / capital * 100.0


def run(ticker: str, train_bars: int = 252, test_bars: int = 21) -> dict:
    """
    Returns walk-forward stats dict. Raises no exceptions — errors are
    returned as {"error": "..."} so the API can forward them cleanly.
    """
    try:
        df = get_data(ticker)
        df = add_features(df)
    except Exception as e:
        return {"error": f"Data fetch failed: {e}"}

    df = df.dropna(subset=FEATURE_COLS + ["Target", "Close"])
    n = len(df)
    min_needed = train_bars + test_bars
    if n < min_needed:
        return {"error": f"Not enough data: need {min_needed} bars, have {n}"}

    X      = df[FEATURE_COLS].values
    y      = df["Target"].values
    closes = df["Close"].values
    dates  = df.index

    windows = []
    start = 0
    while start + train_bars + test_bars <= n:
        te = start + train_bars          # train end
        ts = te                          # test start
        tx = min(ts + test_bars, n)      # test end

        X_tr, y_tr = X[start:te], y[start:te]
        X_te, y_te = X[ts:tx],   y[ts:tx]

        if len(np.unique(y_tr)) < 2:
            start += test_bars
            continue

        model = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            random_state=42, n_jobs=1,
        )
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
        acc   = float(np.mean(preds == y_te) * 100)

        signals   = np.where(proba > 0.55, 1, np.where(proba < 0.45, -1, 0))
        prices_te = closes[ts:tx]
        strat_ret = _trade_window(prices_te, signals)
        bh_ret    = float((prices_te[-1] - prices_te[0]) / prices_te[0] * 100)
        n_sigs    = int(np.sum(signals != 0))

        windows.append({
            "window":      len(windows) + 1,
            "train_start": str(dates[start])[:10],
            "train_end":   str(dates[te - 1])[:10],
            "test_start":  str(dates[ts])[:10],
            "test_end":    str(dates[tx - 1])[:10],
            "accuracy":    round(acc, 1),
            "return_pct":  round(strat_ret, 2),
            "bh_return":   round(bh_ret, 2),
            "beat_bh":     strat_ret > bh_ret,
            "n_signals":   n_sigs,
        })
        start += test_bars

    if not windows:
        return {"error": "No complete windows — try a ticker with more history"}

    accs      = [w["accuracy"]   for w in windows]
    rets      = [w["return_pct"] for w in windows]
    bh_rets   = [w["bh_return"]  for w in windows]
    beat_cnt  = sum(1 for w in windows if w["beat_bh"])
    above_55  = sum(1 for a in accs if a >= 55)

    avg_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))

    if std_acc < 4:
        stability = "STABLE"
    elif std_acc < 9:
        stability = "MODERATE"
    else:
        stability = "UNSTABLE"

    if avg_acc >= 58:
        verdict = "STRONG EDGE"
    elif avg_acc >= 54:
        verdict = "MARGINAL EDGE"
    elif avg_acc >= 50:
        verdict = "WEAK — near random"
    else:
        verdict = "NO EDGE — worse than random"

    return {
        "ticker":        ticker,
        "windows":       windows,
        "n_windows":     len(windows),
        "avg_accuracy":  round(avg_acc, 1),
        "std_accuracy":  round(std_acc, 1),
        "min_accuracy":  round(float(np.min(accs)), 1),
        "max_accuracy":  round(float(np.max(accs)), 1),
        "above_55_pct":  round(above_55 / len(windows) * 100, 1),
        "avg_return":    round(float(np.mean(rets)), 2),
        "avg_bh_return": round(float(np.mean(bh_rets)), 2),
        "beat_bh_count": beat_cnt,
        "beat_bh_pct":   round(beat_cnt / len(windows) * 100, 1),
        "stability":     stability,
        "verdict":       verdict,
        "train_bars":    train_bars,
        "test_bars":     test_bars,
    }
