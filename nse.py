"""
NSE live data module for JARVIS.

Uses only NSE endpoints that don't require Akamai session cookies:
  - /api/marketStatus        → market open/closed status
  - /api/market-data-pre-open?key=ALL  → live prices for all EQ stocks
  - /api/allIndices          → Nifty 50 / Bank Nifty index levels

The quote-equity endpoint requires a WAF-bypassed browser session,
so we use the pre-open feed which returns the same price data in bulk.
"""

import time
import threading
import requests

NSE_BASE   = "https://www.nseindia.com"
QUOTE_TTL  = 30    # seconds — cache during market hours
CLOSED_TTL = 300   # seconds — cache when market closed

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

_lock = threading.Lock()
_session: requests.Session | None = None

# bulk cache: {ts, data: {SYMBOL: quote_dict}}
_bulk_cache: dict = {"ts": 0.0, "data": {}}
_market_cache: dict = {"ts": 0.0, "data": {}}
_index_cache: dict  = {"ts": 0.0, "data": []}


def _get_session() -> requests.Session:
    global _session
    with _lock:
        if _session is None:
            s = requests.Session()
            s.headers.update(_HEADERS)
            _session = s
        return _session


def _get(path: str, params: dict | None = None) -> dict | list:
    try:
        r = _get_session().get(f"{NSE_BASE}{path}", params=params, timeout=8)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}"}
        return r.json()
    except requests.exceptions.Timeout:
        return {"error": "NSE timeout"}
    except Exception as e:
        return {"error": str(e)}


# ── Market status ─────────────────────────────────────────────────────────────

def market_status() -> dict:
    now = time.time()
    if now - _market_cache["ts"] < 60 and _market_cache["data"]:
        return _market_cache["data"]

    raw = _get("/api/marketStatus")
    if isinstance(raw, dict) and "error" in raw:
        result = {"status": "UNKNOWN", "is_open": False, "error": raw["error"]}
    else:
        markets = raw.get("marketState", []) if isinstance(raw, dict) else []
        eq = next(
            (m for m in markets if "Capital" in m.get("market", "")),
            markets[0] if markets else {}
        )
        status = eq.get("marketStatus", "Closed")
        result = {
            "status": status,
            "is_open": status.lower() == "open",
            "trade_date": eq.get("tradeDate", ""),
            "index": eq.get("index", "NIFTY 50"),
            "nifty_last": eq.get("last"),
            "nifty_change": eq.get("variation"),
            "nifty_pct": eq.get("percentChange"),
        }

    _market_cache["ts"] = now
    _market_cache["data"] = result
    return result


# ── Index data ────────────────────────────────────────────────────────────────

def get_indices() -> list:
    now = time.time()
    if now - _index_cache["ts"] < 60 and _index_cache["data"]:
        return _index_cache["data"]

    raw = _get("/api/allIndices")
    indices = []
    if isinstance(raw, dict) and "data" in raw:
        for item in raw["data"]:
            indices.append({
                "name":        item.get("index"),
                "symbol":      item.get("indexSymbol"),
                "last":        item.get("last"),
                "change":      item.get("variation"),
                "pct_change":  item.get("percentChange"),
                "open":        item.get("open"),
                "high":        item.get("high"),
                "low":         item.get("low"),
                "prev_close":  item.get("previousClose"),
                "year_high":   item.get("yearHigh"),
                "year_low":    item.get("yearLow"),
                "advances":    item.get("advances"),
                "declines":    item.get("declines"),
            })
    _index_cache["ts"] = now
    _index_cache["data"] = indices
    return indices


# ── Individual stock quotes (via bulk pre-open feed) ─────────────────────────

def _fetch_bulk() -> dict:
    """Fetch all stock quotes from the pre-open bulk endpoint. Returns {SYMBOL: quote}."""
    raw = _get("/api/market-data-pre-open", {"key": "ALL"})
    if isinstance(raw, dict) and "error" in raw:
        return {}

    result = {}
    for item in raw.get("data", []):
        meta = item.get("metadata", {})
        sym  = meta.get("symbol", "")
        if not sym:
            continue
        prev  = meta.get("previousClose") or meta.get("prevClose")
        last  = meta.get("lastPrice") or meta.get("iep")
        chng  = meta.get("change")
        pchng = meta.get("pChange")
        if last and prev and chng is None:
            chng  = round(last - prev, 2)
            pchng = round(chng / prev * 100, 2) if prev else 0

        result[sym] = {
            "symbol":       sym,
            "name":         sym,
            "last_price":   last,
            "change":       chng,
            "pct_change":   pchng,
            "open":         None,
            "high":         None,
            "low":          None,
            "prev_close":   prev,
            "week52_high":  meta.get("yearHigh"),
            "week52_low":   meta.get("yearLow"),
            "total_volume": meta.get("finalQuantity") or meta.get("totalTradedVolume"),
            "total_value":  meta.get("totalTurnover"),
            "source":       "NSE",
        }
    return result


def _get_bulk_cached() -> dict:
    now = time.time()
    is_open = market_status().get("is_open", False)
    ttl = QUOTE_TTL if is_open else CLOSED_TTL

    if now - _bulk_cache["ts"] < ttl and _bulk_cache["data"]:
        return _bulk_cache["data"]

    data = _fetch_bulk()
    if data:
        _bulk_cache["ts"] = now
        _bulk_cache["data"] = data
    return _bulk_cache.get("data", {})


def get_quote(ticker: str) -> dict:
    """Live quote for one ticker (accepts 'RELIANCE' or 'RELIANCE.NS')."""
    symbol = ticker.upper().replace(".NS", "")
    bulk = _get_bulk_cached()
    if symbol in bulk:
        return bulk[symbol]
    return {"symbol": symbol, "error": "Symbol not found in NSE pre-open feed", "source": "NSE"}


def get_quotes(tickers: list[str]) -> list[dict]:
    """Live quotes for a list of tickers. Single bulk fetch, no per-ticker delay."""
    bulk = _get_bulk_cached()
    results = []
    for t in tickers:
        symbol = t.upper().replace(".NS", "")
        if symbol in bulk:
            results.append(bulk[symbol])
        else:
            results.append({"symbol": symbol, "error": "not found", "source": "NSE"})
    return results
