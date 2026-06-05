"""
Kelly Criterion position sizing for JARVIS trade plans.

Kelly formula:  f* = (b·p - q) / b
  where  p = probability of winning  (LSTM confidence)
         q = 1 - p
         b = net odds (risk/reward ratio, i.e. reward per unit risked)

We always return Half-Kelly (f*/2) as the recommended size — full Kelly
maximises long-run growth in theory but the variance is brutal in practice.
Hard cap: 25% of capital (full-Kelly), 12.5% (half-Kelly) so a single bad
trade never wipes a significant portion of the book.
"""

MAX_FULL = 0.25   # never bet more than 25 % on a single trade
CAPITAL  = 100_000.0   # ₹ — default portfolio size


def fraction(p: float, b: float) -> float:
    """
    Raw Kelly fraction (0–MAX_FULL).
    p: win probability 0–1  (use confidence/100)
    b: reward-to-risk ratio  (risk_reward from trade plan)
    Returns 0 if the trade has negative expectation.
    """
    if b <= 0 or not (0 < p < 1):
        return 0.0
    f = (b * p - (1.0 - p)) / b
    return float(max(0.0, min(f, MAX_FULL)))


def size(p: float, b: float, capital: float = CAPITAL) -> dict:
    """
    Full sizing dict for a single trade signal.
    Returns full-Kelly %, half-Kelly %, and INR position size (half-Kelly).
    """
    f_full = fraction(p, b)
    f_half = f_full / 2.0

    # Qualitative risk label
    if f_half >= 0.10:
        sizing_label = "AGGRESSIVE"
    elif f_half >= 0.05:
        sizing_label = "MODERATE"
    elif f_half > 0.0:
        sizing_label = "CONSERVATIVE"
    else:
        sizing_label = "NO TRADE"

    return {
        "kelly_full_pct":   round(f_full * 100, 1),
        "kelly_half_pct":   round(f_half * 100, 1),
        "position_size_inr": round(f_half * capital),
        "sizing_label":     sizing_label,
        "capital":          capital,
    }
