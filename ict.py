"""
ict.py — lightweight ICT / Smart-Money-Concept event detection for chart markup.

Pure pandas / numpy. Operates on an OHLC(V) DataFrame with a DatetimeIndex and
columns Open, High, Low, Close (Volume optional). Returns a list of event dicts
to render as markers / shaded zones on a candlestick chart.

Event shapes:
  point  (BOS / SWEEP):  {"type", "t", "price", "label"}
  zone   (FVG / OB):     {"type", "t", "high", "low", "label"}

`t` is UNIX epoch seconds (UTC), matching the /api/ohlc candle timestamps so the
frontend can place every marker by time without a second lookup.

These are deliberately *swing-based* (pivot highs/lows) rather than the per-bar
flags used inside the model (lstm_model.add_features → sweep_low/bos_up), which
fire almost every other bar and would clutter the chart. The model keeps its
noisy per-bar features; the chart shows the structural events a trader cares
about.
"""
import numpy as np
import pandas as pd


def _epoch(ts) -> int:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


def _swings(high, low, k=2):
    """Boolean arrays (is_swing_high, is_swing_low) via a +/-k bar pivot."""
    n = len(high)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        win_h = high[i - k:i + k + 1]
        win_l = low[i - k:i + k + 1]
        if high[i] == win_h.max() and win_h.argmax() == k:
            sh[i] = True
        if low[i] == win_l.min() and win_l.argmin() == k:
            sl[i] = True
    return sh, sl


def detect_events(df: pd.DataFrame, k: int = 2, cap_per_type: int = 8) -> list:
    """Detect BOS, liquidity sweeps, FVGs and order blocks on an OHLC frame."""
    if df is None or len(df) < (2 * k + 3):
        return []

    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    idx = df.index
    n = len(df)

    sh, sl = _swings(h, l, k)
    events = []

    # ── BOS (break of structure) + SWEEP (liquidity grab) ──────────────────────
    last_sh_val = last_sl_val = None
    broke_up = broke_dn = False
    for i in range(n):
        # confirm a pivot only once its k future bars exist (i.e. at bar i, the
        # pivot sitting at i-k is now fully formed)
        j = i - k
        if j >= 0:
            if sh[j]:
                last_sh_val, broke_up = h[j], False
            if sl[j]:
                last_sl_val, broke_dn = l[j], False

        if last_sh_val is not None and c[i] > last_sh_val and not broke_up:
            events.append({"type": "BOS_UP", "t": _epoch(idx[i]),
                           "price": round(float(last_sh_val), 2),
                           "label": "Break of Structure ↑"})
            broke_up = True
        if last_sl_val is not None and c[i] < last_sl_val and not broke_dn:
            events.append({"type": "BOS_DOWN", "t": _epoch(idx[i]),
                           "price": round(float(last_sl_val), 2),
                           "label": "Break of Structure ↓"})
            broke_dn = True

        # sweep = wick takes the prior swing but price closes back inside
        if last_sl_val is not None and l[i] < last_sl_val and c[i] > last_sl_val:
            events.append({"type": "SWEEP_LOW", "t": _epoch(idx[i]),
                           "price": round(float(l[i]), 2),
                           "label": "Liquidity swept (low)"})
        if last_sh_val is not None and h[i] > last_sh_val and c[i] < last_sh_val:
            events.append({"type": "SWEEP_HIGH", "t": _epoch(idx[i]),
                           "price": round(float(h[i]), 2),
                           "label": "Liquidity swept (high)"})

    # ── FVG (fair value gap) — 3-bar imbalance ─────────────────────────────────
    for i in range(2, n):
        if l[i] > h[i - 2]:        # bullish gap between bar i-2 high and bar i low
            events.append({"type": "FVG_BULL", "t": _epoch(idx[i - 1]),
                           "high": round(float(l[i]), 2),
                           "low": round(float(h[i - 2]), 2),
                           "label": "Fair Value Gap ↑"})
        elif h[i] < l[i - 2]:      # bearish gap
            events.append({"type": "FVG_BEAR", "t": _epoch(idx[i - 1]),
                           "high": round(float(l[i - 2]), 2),
                           "low": round(float(h[i]), 2),
                           "label": "Fair Value Gap ↓"})

    # ── Order blocks: last opposite-colour candle before a BOS ─────────────────
    pos_by_epoch = {_epoch(idx[i]): i for i in range(n)}

    def _order_block(bos_events, bullish):
        for e in bos_events:
            bi = pos_by_epoch.get(e["t"])
            if bi is None:
                continue
            for j in range(bi - 1, max(bi - 6, 0) - 1, -1):
                down = c[j] < o[j]
                if (bullish and down) or (not bullish and not down):
                    if bullish:
                        events.append({"type": "OB_BULL", "t": _epoch(idx[j]),
                                       "high": round(float(max(o[j], c[j])), 2),
                                       "low": round(float(l[j]), 2),
                                       "label": "Bullish Order Block"})
                    else:
                        events.append({"type": "OB_BEAR", "t": _epoch(idx[j]),
                                       "high": round(float(h[j]), 2),
                                       "low": round(float(min(o[j], c[j])), 2),
                                       "label": "Bearish Order Block"})
                    break

    _order_block([e for e in events if e["type"] == "BOS_UP"], bullish=True)
    _order_block([e for e in events if e["type"] == "BOS_DOWN"], bullish=False)

    # ── cap each type to most recent N, return chronological ───────────────────
    by_type: dict = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)
    capped = []
    for evs in by_type.values():
        capped.extend(evs[-cap_per_type:])
    capped.sort(key=lambda e: e["t"])
    return capped


if __name__ == "__main__":
    import yfinance as yf
    d = yf.download("RELIANCE.NS", period="6mo", interval="1d", progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.dropna()
    evs = detect_events(d)
    from collections import Counter
    print("total events:", len(evs))
    print(Counter(e["type"] for e in evs))
    for e in evs[-8:]:
        print(e)
