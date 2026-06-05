"""
Telegram alert engine for JARVIS.

Sends a formatted message when a BUY/SELL signal fires at >= ALERT_THRESHOLD
confidence. Deduplicates: same symbol+signal won't re-alert within
COOLDOWN_HOURS. Called in a daemon thread from /api/signals so it never
blocks the API response.

Setup:
  1. Create a bot via @BotFather on Telegram → copy the token
  2. Start a chat with your bot, then visit:
     https://api.telegram.org/bot<TOKEN>/getUpdates
     to find your chat_id
  3. Fill in TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in config.py
"""
import time
import threading
from datetime import datetime

import requests as _req

try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ALERT_THRESHOLD
except ImportError:
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID   = ""
    ALERT_THRESHOLD    = 60

COOLDOWN_HOURS = 4

_cache: dict = {}       # {symbol: {"signal": str, "sent_at": float}}
_lock = threading.Lock()


def is_configured() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram(text: str) -> bool:
    """Low-level: POST a message to the Telegram Bot API. Returns True on success."""
    if not is_configured():
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = _req.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        return False


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt(price: float | None) -> str:
    if price is None:
        return "—"
    return f"₹{price:,.2f}"


def _pct(price: float | None, ref: float | None) -> str:
    if price is None or ref is None or ref == 0:
        return ""
    p = (price - ref) / ref * 100
    sign = "+" if p >= 0 else ""
    return f"({sign}{p:.1f}%)"


def _build_message(d: dict) -> str:
    sig    = d["signal"]
    sym    = d["symbol"].replace(".NS", "")
    conf   = d["confidence"]
    acc    = d["accuracy"]
    entry  = d.get("entry")
    sl     = d.get("stop_loss")
    tp1    = d.get("target1")
    tp2    = d.get("target2")
    rr     = d.get("risk_reward", 0)
    reason = d.get("reason", "")
    now    = datetime.now().strftime("%H:%M:%S")
    icon   = "📈" if sig == "BUY" else "📉"

    return "\n".join([
        "🚨 <b>JARVIS SIGNAL ALERT</b>",
        "",
        f"{icon} <b>{sig} — {sym}</b>",
        f"Confidence: <b>{conf}%</b>  ·  Accuracy: <b>{acc}%</b>",
        "",
        f"💰 Entry:      <b>{_fmt(entry)}</b>",
        f"🛑 Stop Loss:  <b>{_fmt(sl)}</b>  {_pct(sl, entry)}",
        f"🎯 Target 1:   <b>{_fmt(tp1)}</b>  {_pct(tp1, entry)}",
        f"🎯 Target 2:   <b>{_fmt(tp2)}</b>  {_pct(tp2, entry)}",
        f"⚖️  Risk/Reward: <b>1 : {rr:.1f}</b>",
        "",
        f"📊 <i>{reason}</i>",
        f"🕐 {now} IST",
    ])


# ── Dedup logic ───────────────────────────────────────────────────────────────

def _should_send(symbol: str, signal: str, confidence: int) -> bool:
    if signal == "HOLD":
        return False
    if confidence < ALERT_THRESHOLD:
        return False
    with _lock:
        prev = _cache.get(symbol)
        if prev is None:
            return True
        if prev["signal"] != signal:
            return True
        elapsed = time.time() - prev["sent_at"]
        return elapsed >= COOLDOWN_HOURS * 3600


def _record(symbol: str, signal: str) -> None:
    with _lock:
        _cache[symbol] = {"signal": signal, "sent_at": time.time()}


# ── Public API ────────────────────────────────────────────────────────────────

def maybe_send(signal_data: dict) -> None:
    """Check dedup rules and send if appropriate. Safe to call in a daemon thread."""
    sym  = signal_data.get("symbol", "")
    sig  = signal_data.get("signal", "HOLD")
    conf = signal_data.get("confidence", 0)
    if not _should_send(sym, sig, conf):
        return
    msg = _build_message(signal_data)
    ok  = send_telegram(msg)
    if ok:
        _record(sym, sig)


def alert_status() -> dict:
    """Returns public status info — never exposes the actual token."""
    with _lock:
        recent = {
            sym: {"signal": v["signal"], "sent_ago_min": round((time.time() - v["sent_at"]) / 60)}
            for sym, v in _cache.items()
        }
    return {
        "configured": is_configured(),
        "threshold": ALERT_THRESHOLD,
        "cooldown_hours": COOLDOWN_HOURS,
        "recent_alerts": recent,
    }
