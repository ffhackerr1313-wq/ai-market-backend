from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from lstm_model import train_lstm, get_data, add_features
from sentiment import get_sentiment, combined_signal
from config import FRONTEND_URL
import time
import os
import json
import asyncio
import datetime as _dt
import threading
import warnings
warnings.filterwarnings("ignore")

import db          # local SQLite helper (watchlist + config + positions)
import ict         # ICT/SMC event detection for chart markup
import alerts      # Telegram alert engine
import walkforward # Walk-forward validation
import kelly       # Kelly criterion position sizing
import confluence  # Multi-timeframe confluence
import nse         # NSE live data
import strategies  # Multi-strategy engine

# ── WebSocket connection manager ──────────────────────────────────────────────

class _WsManager:
    def __init__(self):
        self.connections: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.add(ws)

    def disconnect(self, ws: WebSocket):
        self.connections.discard(ws)

    async def broadcast(self, payload: dict):
        if not self.connections:
            return
        msg = json.dumps(payload)
        dead = []
        for ws in list(self.connections):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections.discard(ws)

_ws_manager = _WsManager()

_IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))

async def _price_broadcaster():
    """Push NSE quotes + market status to all WS clients every 10 seconds."""
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(10)
        if not _ws_manager.connections:
            continue
        try:
            watchlist = db.list_watchlist()
            symbols = [w["symbol"] for w in watchlist] if watchlist else [
                s["symbol"] for s in STOCK_UNIVERSE.get("stocks", [])[:20]
            ]
            quotes, market = await asyncio.gather(
                loop.run_in_executor(None, lambda: nse.get_quotes(symbols)),
                loop.run_in_executor(None, nse.market_status),
            )
            ts = _dt.datetime.now(tz=_IST).strftime("%H:%M:%S")
            await _ws_manager.broadcast({"type": "prices", "quotes": quotes, "market": market, "ts": ts})
        except Exception as exc:
            print(f"[ws broadcaster] {exc}")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Restore persisted Telegram config from SQLite on every startup
    saved_token     = db.get_config("telegram_token",   "")
    saved_chat_id   = db.get_config("telegram_chat_id", "")
    saved_threshold = db.get_config("alert_threshold",  "60")
    if saved_token:
        alerts.TELEGRAM_BOT_TOKEN = saved_token
    if saved_chat_id:
        alerts.TELEGRAM_CHAT_ID = saved_chat_id
    try:
        alerts.ALERT_THRESHOLD = int(saved_threshold)
    except ValueError:
        pass

    task = asyncio.create_task(_price_broadcaster())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
_cache: dict = {}
_cache_ts: dict = {}
CACHE_TTL = 1800  # 30 minutes

# ── Stock universe (Nifty 50 + Bank Nifty) loaded from stocks.json ──────────────
_STOCKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stocks.json")
try:
    with open(_STOCKS_PATH, encoding="utf-8") as _f:
        STOCK_UNIVERSE = json.load(_f)
except Exception as _e:
    print(f"[warn] could not load stocks.json: {_e}")
    STOCK_UNIVERSE = {"indices": [], "stocks": []}

# symbol -> {symbol, name, sector, cap} for quick lookups (stocks + indices for search)
_STOCK_BY_SYMBOL = {
    **{s["symbol"].upper(): s for s in STOCK_UNIVERSE.get("stocks", [])},
    **{s["symbol"].upper(): {**s, "sector": "Index"} for s in STOCK_UNIVERSE.get("indices", [])},
}

def _lookup_stock(symbol):
    return _STOCK_BY_SYMBOL.get(symbol.strip().upper())

def _normalize_symbol(symbol):
    """Uppercase and auto-append .NS for plain Indian tickers (leave indices alone)."""
    symbol = symbol.strip().upper()
    if not symbol.endswith(".NS") and not symbol.startswith("^"):
        symbol = f"{symbol}.NS"
    return symbol

def get_cached(ticker):
    now = time.time()
    if ticker in _cache and now - _cache_ts.get(ticker, 0) < CACHE_TTL:
        return _cache[ticker]
    result = train_lstm(ticker)
    _cache[ticker] = result
    _cache_ts[ticker] = now
    return result

@app.get("/")
def root():
    return {"status": "AI Market Predictor API running"}

# ── Stock universe ─────────────────────────────────────────────────────────────

@app.get("/api/stocks")
def get_stocks(q: str | None = None, sector: str | None = None):
    """Searchable stock universe (Nifty 50 + Bank Nifty). Optional q / sector filter."""
    stocks = STOCK_UNIVERSE.get("stocks", [])
    if sector:
        stocks = [s for s in stocks if s.get("sector", "").lower() == sector.lower()]
    if q:
        ql = q.strip().lower()
        stocks = [s for s in stocks
                  if ql in s["symbol"].lower() or ql in s.get("name", "").lower()]
    sectors = sorted({s.get("sector", "") for s in STOCK_UNIVERSE.get("stocks", []) if s.get("sector")})
    return {
        "indices": STOCK_UNIVERSE.get("indices", []),
        "stocks": stocks,
        "count": len(stocks),
        "sectors": sectors,
    }

# ── Watchlist (#1) ─────────────────────────────────────────────────────────────

@app.get("/api/watchlist")
def get_watchlist():
    """User's watchlist enriched with latest price + day change per symbol."""
    items = db.list_watchlist()
    for it in items:
        try:
            hist = yf.Ticker(it["symbol"]).history(period="2d")
            if len(hist) >= 2:
                price = float(hist["Close"].iloc[-1])
                prev  = float(hist["Close"].iloc[-2])
                it["price"]  = round(price, 2)
                it["change"] = round((price - prev) / prev * 100, 2)
            else:
                it["price"], it["change"] = None, 0
        except Exception:
            it["price"], it["change"] = None, 0
        it["in_watchlist"] = True
    return items

@app.post("/api/watchlist/{symbol}")
def add_to_watchlist(symbol: str):
    symbol = _normalize_symbol(symbol)
    meta = _lookup_stock(symbol)
    db.add_watchlist(
        symbol,
        name=meta["name"] if meta else None,
        sector=meta.get("sector") if meta else None,
    )
    return {"ok": True, "symbol": symbol, "watchlist": db.list_watchlist()}

@app.delete("/api/watchlist/{symbol}")
def remove_from_watchlist(symbol: str):
    symbol = _normalize_symbol(symbol)
    db.remove_watchlist(symbol)
    return {"ok": True, "symbol": symbol, "watchlist": db.list_watchlist()}

# ── Tickers ────────────────────────────────────────────────────────────────────

@app.get("/api/tickers")
def get_tickers():
    result = []
    for t in TICKERS:
        try:
            hist = yf.Ticker(t).history(period="2d")
            if len(hist) < 2:
                continue
            price  = float(hist["Close"].iloc[-1])
            prev   = float(hist["Close"].iloc[-2])
            change = round(((price - prev) / prev) * 100, 2)
            sym    = "₹" if ".NS" in t else "$"
            result.append({"symbol": t, "price": f"{sym}{price:.2f}", "change": change})
        except Exception:
            result.append({"symbol": t, "price": "N/A", "change": 0})
    return result

# ── Signals — multi-strategy, all 60 Nifty stocks (#17) ───────────────────────

@app.get("/api/signals")
def get_signals(strategy: str | None = None):
    active = strategy or db.get_config("active_strategy", strategies.DEFAULT_STRATEGY)
    if active not in strategies.STRATEGY_META:
        active = strategies.DEFAULT_STRATEGY

    meta = strategies.STRATEGY_META[active]
    tf = meta["timeframe"]
    interval, period = ("60m", "1mo") if tf == "1h" else ("1d", "3mo")

    all_stocks = STOCK_UNIVERSE.get("stocks", [])
    symbols = [s["symbol"] for s in all_stocks if s.get("symbol")]

    # bulk download — yfinance fetches all tickers in one request
    try:
        raw = yf.download(symbols, interval=interval, period=period,
                          progress=False, group_by="ticker")
    except Exception:
        raw = None

    capital = float(db.get_config("capital", "100000"))
    result = []

    for stock in all_stocks:
        sym = stock.get("symbol", "")
        if not sym:
            continue
        try:
            if raw is not None and len(symbols) > 1:
                try:
                    df = raw[sym].dropna(subset=["Open", "High", "Low", "Close"])
                except (KeyError, TypeError):
                    df = pd.DataFrame()
                if df.empty:
                    df = yf.download(sym, interval=interval, period=period, progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.dropna(subset=["Open", "High", "Low", "Close"])
            else:
                df = yf.download(sym, interval=interval, period=period, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna(subset=["Open", "High", "Low", "Close"])

            if df is None or len(df) < 30:
                continue

            sig_data = strategies.run(df, active)
            price = float(df["Close"].iloc[-1])

            # ATR(14) trade plan
            h, l, c = df["High"], df["Low"], df["Close"]
            tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            if not np.isfinite(atr) or atr <= 0:
                atr = price * 0.01
            entry = round(price, 2)
            sl    = round(price - 1.5 * atr, 2)
            tp1   = round(price + 2.0 * atr, 2)
            tp2   = round(price + 4.0 * atr, 2)
            risk  = entry - sl
            rr    = round((tp1 - entry) / risk, 2) if risk > 0 else 0

            kelly_data = kelly.size(sig_data.get("confidence", 50) / 100.0, rr, capital)

            item = {
                "symbol":            sym,
                "name":              stock.get("name", sym.replace(".NS", "")),
                "sector":            stock.get("sector", ""),
                "signal":            sig_data["signal"],
                "confidence":        sig_data.get("confidence", 50),
                "accuracy":          sig_data.get("accuracy", 50),
                "reason":            sig_data.get("reason", ""),
                "scanner_signal":    sig_data.get("scanner_signal", "NEUTRAL"),
                "breakout_strength": sig_data.get("breakout_strength", "WEAK"),
                "breakout_up":       sig_data.get("breakout_up", False),
                "breakout_down":     sig_data.get("breakout_down", False),
                "volume_spike":      sig_data.get("volume_spike", False),
                "volume_ratio":      sig_data.get("volume_ratio", 1.0),
                "model":             active,
                "strategy":          active,
                "entry":             entry,
                "stop_loss":         sl,
                "target1":           tp1,
                "target2":           tp2,
                "risk_reward":       rr,
                "atr":               round(atr, 2),
                **kelly_data,
            }

            if item["signal"] in ("BUY", "SELL"):
                threading.Thread(target=alerts.maybe_send, args=(item,), daemon=True).start()

            result.append(item)
        except Exception:
            continue

    order = {"BUY": 0, "SELL": 1, "HOLD": 2}
    result.sort(key=lambda x: (order.get(x["signal"], 3), -x.get("confidence", 0)))
    return result

# ── Legacy chart endpoint ──────────────────────────────────────────────────────

@app.get("/api/chart/{ticker}")
def get_chart(ticker: str):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.ffill().dropna()
        df["MA50"] = df["Close"].rolling(50).mean()
        result = [{"date": idx.strftime("%b %d"), "price": round(float(row["Close"]), 2),
                   "ma50": round(float(row["MA50"]), 2) if not pd.isna(row["MA50"]) else None,
                   "pred": None} for idx, row in df.iterrows()]
        last = float(df["Close"].iloc[-1])
        avg_ret = float(df["Close"].pct_change().iloc[-10:].mean())
        for i in range(1, 6):
            result.append({"date": f"+{i}d", "price": None, "ma50": None,
                           "pred": round(last * (1 + avg_ret * i), 2)})
        return result
    except Exception:
        return []

# ── Candlestick chart — OHLC + ICT events + LSTM forecast (#2/#3/#4/#8) ───────

TF_CONFIG = {
    "1m":  {"interval": "1m",  "period": "5d"},
    "5m":  {"interval": "5m",  "period": "1mo"},
    "15m": {"interval": "15m", "period": "1mo"},
    "30m": {"interval": "30m", "period": "2mo"},
    "1h":  {"interval": "60m", "period": "3mo"},
    "4h":  {"interval": "60m", "period": "6mo", "resample": "4h"},
    "1d":  {"interval": "1d",  "period": "1y"},
    "1wk": {"interval": "1wk", "period": "5y"},
}

def _epoch(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())

def _build_prediction(df, p_up):
    closes = df["Close"]
    last = float(closes.iloc[-1])
    daily_vol = float(closes.pct_change().dropna().std())
    if not np.isfinite(daily_vol) or daily_vol <= 0:
        daily_vol = 0.01
    edge = (p_up - 0.5) * 2.0
    mu   = edge * daily_vol
    future = pd.bdate_range(pd.Timestamp(df.index[-1]) + pd.Timedelta(days=1), periods=5)
    pts = []
    for step in range(1, 6):
        price = last * ((1 + mu) ** step)
        sigma = daily_vol * np.sqrt(step)
        pts.append({
            "t": _epoch(future[step - 1]),
            "c": round(price, 2),
            "high": round(price * (1 + sigma), 2),
            "low":  round(price * (1 - sigma), 2),
        })
    return {"basis": "lstm", "p_up": round(p_up, 3),
            "daily_vol": round(daily_vol, 4), "points": pts}

@app.get("/api/ohlc/{ticker}")
def get_ohlc(ticker: str, timeframe: str = "1d", period: str | None = None):
    tf  = timeframe if timeframe in TF_CONFIG else "1d"
    cfg = TF_CONFIG[tf]
    interval   = cfg["interval"]
    use_period = period or cfg["period"]
    try:
        df = yf.download(ticker, period=use_period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        if df.empty:
            return {"symbol": ticker, "timeframe": tf, "candles": [],
                    "error": "No data returned for this symbol/timeframe."}

        if cfg.get("resample"):
            df = (df.resample(cfg["resample"], label="left", closed="left")
                    .agg({"Open": "first", "High": "max", "Low": "min",
                          "Close": "last", "Volume": "sum"})
                    .dropna(subset=["Open", "High", "Low", "Close"]))

        df = df.tail(2000)
        has_vol = "Volume" in df.columns

        close = df["Close"]
        ind = {
            "ma20":  close.rolling(20).mean(),
            "ma50":  close.rolling(50).mean(),
            "ma200": close.rolling(200).mean(),
            "ema9":  close.ewm(span=9).mean(),
        }
        def _series(s):
            return [round(float(v), 2) if pd.notna(v) else None for v in s]

        candles = [{
            "t": _epoch(idx),
            "o": round(float(r["Open"]), 2),
            "h": round(float(r["High"]), 2),
            "l": round(float(r["Low"]), 2),
            "c": round(float(r["Close"]), 2),
            "v": int(r["Volume"]) if has_vol and pd.notna(r["Volume"]) else 0,
        } for idx, r in df.iterrows()]

        events = ict.detect_events(df)

        prediction = None
        if tf == "1d" and ticker in _cache:
            try:
                p_up = float(_cache[ticker].get("confidence", 50)) / 100.0
                prediction = _build_prediction(df, p_up)
            except Exception:
                prediction = None

        return {
            "symbol": ticker, "timeframe": tf, "interval": interval,
            "period": use_period, "count": len(candles),
            "candles": candles,
            "indicators": {k: _series(v) for k, v in ind.items()},
            "events": events,
            "prediction": prediction,
        }
    except Exception as e:
        return {"symbol": ticker, "timeframe": tf, "candles": [], "error": str(e)}

# ── Portfolio ──────────────────────────────────────────────────────────────────

@app.get("/api/portfolio")
def get_portfolio():
    allocs = {"RELIANCE.NS": 0.25, "TCS.NS": 0.20, "HDFCBANK.NS": 0.20,
              "INFY.NS": 0.20, "ICICIBANK.NS": 0.15}
    result = []
    for t, alloc in allocs.items():
        try:
            df = yf.download(t, period="6mo", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df  = df.dropna()
            ret = float((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100)
            result.append({"symbol": t, "allocation": round(alloc * 100),
                           "ret": round(ret, 2), "value": round(100000 * alloc * (1 + ret / 100), 2)})
        except Exception:
            result.append({"symbol": t, "allocation": round(alloc * 100),
                           "ret": 0, "value": round(100000 * alloc, 2)})
    return result

@app.get("/api/portfolio/history")
def get_portfolio_history():
    try:
        tlist  = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
        allocs = [0.25, 0.20, 0.20, 0.20, 0.15]
        data   = yf.download(tlist, period="10mo", interval="1mo", progress=False)["Close"].ffill().dropna()
        result = []
        for idx, row in data.iterrows():
            val = sum(100000 * allocs[i] * float(row[t]) / float(data[t].iloc[0])
                      for i, t in enumerate(tlist) if t in data.columns)
            result.append({"m": idx.strftime("%b"), "v": round(val, 2)})
        return result
    except Exception:
        return [{"m": "Jan", "v": 100000}]

# ── Positions (#6) ─────────────────────────────────────────────────────────────

class PositionIn(BaseModel):
    symbol: str
    shares: float
    avg_cost: float
    entry_date: str
    notes: str | None = None

@app.get("/api/positions")
def get_positions():
    rows = db.list_positions()
    result = []
    for r in rows:
        try:
            hist = yf.Ticker(r["symbol"]).history(period="2d")
            price = float(hist["Close"].iloc[-1]) if len(hist) >= 1 else r["avg_cost"]
        except Exception:
            price = r["avg_cost"]
        invested      = r["shares"] * r["avg_cost"]
        current_value = r["shares"] * price
        pnl           = current_value - invested
        pnl_pct       = (pnl / invested * 100) if invested > 0 else 0
        result.append({
            **r,
            "current_price": round(price, 2),
            "invested":      round(invested, 2),
            "current_value": round(current_value, 2),
            "pnl":           round(pnl, 2),
            "pnl_pct":       round(pnl_pct, 2),
        })
    return result

@app.post("/api/positions")
def create_position(pos: PositionIn):
    symbol = _normalize_symbol(pos.symbol)
    id_ = db.add_position(symbol, pos.shares, pos.avg_cost, pos.entry_date, pos.notes)
    return {"ok": True, "id": id_}

@app.put("/api/positions/{position_id}")
def edit_position(position_id: int, pos: PositionIn):
    db.update_position(position_id, pos.shares, pos.avg_cost, pos.entry_date, pos.notes)
    return {"ok": True}

@app.delete("/api/positions/{position_id}")
def remove_position(position_id: int):
    db.delete_position(position_id)
    return {"ok": True}

# ── Risk ───────────────────────────────────────────────────────────────────────

@app.get("/api/risk/{ticker}")
def get_risk(ticker: str):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df  = df.dropna()
        ret = df["Close"].pct_change().dropna()
        vol    = float(ret.std() * np.sqrt(252) * 100)
        sharpe = float((ret.mean() / ret.std()) * np.sqrt(252)) if ret.std() > 0 else 0
        mdd    = float(((df["Close"] / df["Close"].cummax()) - 1).min() * 100)
        wr     = float((ret > 0).mean() * 100)
        return {"volatility": round(abs(vol), 2), "sharpe": round(sharpe, 2),
                "max_drawdown": round(abs(mdd), 2), "win_rate": round(wr, 1)}
    except Exception:
        return {"volatility": 0, "sharpe": 0, "max_drawdown": 0, "win_rate": 0}

# ── Model info ─────────────────────────────────────────────────────────────────

@app.get("/api/model/info")
def model_info():
    try:
        import tensorflow as tf
        return {"model": "LSTM", "tensorflow": tf.__version__, "features": 17, "lookback": 30}
    except Exception:
        return {"model": "RandomForest", "tensorflow": None, "features": 17, "lookback": 0}

# ── Backtest (#14) ─────────────────────────────────────────────────────────────

@app.get("/api/backtest/{ticker}")
def get_backtest(ticker: str):
    try:
        df = get_data(ticker)
        df = add_features(df)

        FEATURE_COLS = [
            "Returns", "Volatility", "RSI", "MACD", "MACD_Signal", "MACD_Hist",
            "BB_Width", "BB_Pos", "Momentum", "ROC", "Volume_Ratio",
            "MA10", "MA20", "MA50", "HTF_Bullish", "sweep_low", "bos_up",
        ]

        from sklearn.ensemble import RandomForestClassifier
        X = df[FEATURE_COLS].values
        y = df["Target"].values
        split = int(len(df) * 0.8)

        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X[:split], y[:split])
        proba = model.predict_proba(X[split:])

        predictions = np.where(proba[:, 1] > 0.55, 1, np.where(proba[:, 1] < 0.45, -1, 0))
        test_df = df.iloc[split:].copy()
        test_df["Signal"] = predictions

        capital = 100000
        cash = capital
        shares = 0
        entry_price = 0
        portfolio_values = []
        trades = []
        sl_pct = 0.03
        tp_pct = 0.06

        for i in range(len(test_df)):
            price  = float(test_df["Close"].iloc[i])
            signal = int(test_df["Signal"].iloc[i])
            date   = str(test_df.index[i])[:10]

            if signal == 1 and shares == 0:
                shares = (cash * 0.95) / price
                entry_price = price
                cash -= shares * price
                trades.append({"date": date, "type": "BUY", "price": round(price, 2), "pnl": 0})
            elif shares > 0:
                pnl_pct_cur = (price - entry_price) / entry_price
                if pnl_pct_cur <= -sl_pct or pnl_pct_cur >= tp_pct or signal == -1:
                    pnl   = (price - entry_price) * shares
                    cash += shares * price
                    trades.append({"date": date, "type": "SELL", "price": round(price, 2), "pnl": round(pnl, 2)})
                    shares = 0

            total = cash + shares * price
            portfolio_values.append({"date": date, "value": round(total, 2), "price": round(price, 2)})

        final_value  = portfolio_values[-1]["value"] if portfolio_values else capital
        total_return = (final_value - capital) / capital * 100
        buy_hold     = float((test_df["Close"].iloc[-1] - test_df["Close"].iloc[0]) / test_df["Close"].iloc[0] * 100)

        sell_trades   = [t for t in trades if t["type"] == "SELL"]
        wins          = [t for t in sell_trades if t["pnl"] > 0]
        losses        = [t for t in sell_trades if t["pnl"] <= 0]
        win_rate      = len(wins) / len(sell_trades) * 100 if sell_trades else 0
        avg_win       = np.mean([t["pnl"] for t in wins])   if wins   else 0
        avg_loss      = np.mean([t["pnl"] for t in losses]) if losses else 0
        loss_sum      = sum(t["pnl"] for t in losses)
        profit_factor = abs(sum(t["pnl"] for t in wins) / loss_sum) if losses and loss_sum != 0 else 0

        values = [p["value"] for p in portfolio_values]
        peak = capital
        max_dd = 0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd

        monthly: dict = {}
        for p in portfolio_values:
            monthly[p["date"][:7]] = p["value"]
        monthly_returns = []
        months = sorted(monthly.keys())
        for i in range(1, len(months)):
            prev = monthly[months[i - 1]]
            curr = monthly[months[i]]
            monthly_returns.append({"month": months[i], "return": round((curr - prev) / prev * 100, 2)})

        return {
            "ticker":          ticker,
            "initial_capital": capital,
            "final_value":     round(final_value, 2),
            "total_return":    round(total_return, 2),
            "buy_hold_return": round(buy_hold, 2),
            "total_trades":    len(sell_trades),
            "win_rate":        round(win_rate, 1),
            "avg_win":         round(avg_win, 2),
            "avg_loss":        round(avg_loss, 2),
            "profit_factor":   round(profit_factor, 2),
            "max_drawdown":    round(max_dd, 2),
            "sharpe":          round(float(
                np.mean(np.diff(values) / np.array(values[:-1])) /
                (np.std(np.diff(values) / np.array(values[:-1])) + 1e-9) * np.sqrt(252)
            ), 2),
            "portfolio_curve": portfolio_values[-100:],
            "monthly_returns": monthly_returns,
            "trades":          trades[-20:],
        }
    except Exception as e:
        return {"error": str(e)}

# ── Search ─────────────────────────────────────────────────────────────────────

@app.get("/api/search/{query}")
def search_stock(query: str):
    try:
        symbol = query.upper()
        if not symbol.endswith(".NS") and not symbol.startswith("^"):
            symbol = f"{symbol}.NS"

        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y")
        if len(df) < 50:
            return {"error": "Not enough data"}

        price = float(df["Close"].iloc[-1])
        df["EMA50"]  = df["Close"].ewm(span=50).mean()
        df["EMA200"] = df["Close"].ewm(span=200).mean()
        ema50  = float(df["EMA50"].iloc[-1])
        ema200 = float(df["EMA200"].iloc[-1])

        delta = df["Close"].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi   = float(100 - (100 / (1 + gain / loss)).iloc[-1])

        momentum = float((df["Close"].iloc[-1] - df["Close"].iloc[-6]) / df["Close"].iloc[-6] * 100)

        df["H-L"]  = df["High"] - df["Low"]
        df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
        df["L-PC"] = abs(df["Low"]  - df["Close"].shift(1))
        df["TR"]   = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
        df["ATR"]  = df["TR"].rolling(14).mean()
        atr        = float(df["ATR"].iloc[-1])

        entry     = round(price, 2)
        stop_loss = round(price - 1.5 * atr, 2)
        target1   = round(price + 2 * atr, 2)
        target2   = round(price + 4 * atr, 2)
        risk      = entry - stop_loss
        reward    = target1 - entry
        rr_ratio  = round(reward / risk, 2) if risk > 0 else 0

        high20      = float(df["High"].rolling(20).max().iloc[-2])
        breakout    = price > high20
        current_vol = float(df["Volume"].iloc[-1])
        avg_vol     = float(df["Volume"].rolling(20).mean().iloc[-1])
        volume_spike = current_vol > avg_vol * 1.8

        # Signal Engine
        score = 0
        if ema50 > ema200: score += 2
        if rsi < 35:       score += 2
        elif rsi > 65:     score -= 2
        if momentum > 2:   score += 1
        if breakout:       score += 2
        if volume_spike:   score += 1

        signal = "BUY" if score >= 4 else "SELL" if score <= -2 else "HOLD"

        # Kelly sizing for the search result
        rr_for_kelly = rr_ratio if rr_ratio > 0 else 1.0
        p_win = 0.60 if signal == "BUY" else 0.40 if signal == "SELL" else 0.50
        capital = float(db.get_config("capital", "100000"))
        kelly_data = kelly.size(p_win, rr_for_kelly, capital)

        return {
            "symbol": symbol, "price": round(price, 2), "signal": signal,
            "rsi": round(rsi, 2), "momentum": round(momentum, 2),
            "ema50": round(ema50, 2), "ema200": round(ema200, 2),
            "atr": round(atr, 2), "entry": entry,
            "stop_loss": stop_loss, "target1": target1, "target2": target2,
            "risk_reward": rr_ratio, "breakout": breakout,
            "volume_spike": volume_spike, "score": score,
            **kelly_data,
        }
    except Exception as e:
        return {"error": str(e)}

# ── News / Sentiment ───────────────────────────────────────────────────────────

@app.get("/api/news/{ticker}")
def get_news(ticker: str):
    try:
        sentiment = get_sentiment(ticker)
        pred      = get_cached(ticker)
        combined  = combined_signal(pred["signal"], pred["confidence"], sentiment)
        return {
            "ticker":           ticker,
            "sentiment_label":  sentiment["sentiment_label"],
            "sentiment_score":  sentiment["sentiment_score"],
            "article_count":    sentiment["article_count"],
            "headlines":        sentiment["headlines"],
            "final_signal":     combined["final_signal"],
            "reason":           combined["reason"],
            "error":            sentiment["error"],
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e), "headlines": []}

@app.get("/api/news")
def get_all_news():
    results = []
    for t in TICKERS:
        try:
            sentiment = get_sentiment(t)
            pred      = get_cached(t)
            combined  = combined_signal(pred["signal"], pred["confidence"], sentiment)
            results.append({
                "ticker":          t,
                "sentiment_label": sentiment["sentiment_label"],
                "sentiment_score": sentiment["sentiment_score"],
                "final_signal":    combined["final_signal"],
                "headlines":       sentiment["headlines"][:3],
            })
        except Exception as e:
            results.append({"ticker": t, "error": str(e), "headlines": []})
    return results

# ── Telegram Alerts (#10) ─────────────────────────────────────────────────────

class AlertConfig(BaseModel):
    bot_token: str
    chat_id: str
    threshold: int = 60

@app.get("/api/alerts/config")
def get_alert_config():
    return alerts.alert_status()

@app.post("/api/alerts/config")
def set_alert_config(cfg: AlertConfig):
    alerts.TELEGRAM_BOT_TOKEN = cfg.bot_token
    alerts.TELEGRAM_CHAT_ID   = cfg.chat_id
    alerts.ALERT_THRESHOLD    = cfg.threshold
    db.set_config("telegram_token",   cfg.bot_token)
    db.set_config("telegram_chat_id", cfg.chat_id)
    db.set_config("alert_threshold",  str(cfg.threshold))
    return {"ok": True, "configured": bool(cfg.bot_token and cfg.chat_id)}

@app.post("/api/alerts/test")
def test_alert():
    ok = alerts.send_telegram("🔔 JARVIS test alert — Telegram is configured correctly!")
    return {"ok": ok, "configured": alerts.is_configured()}

# ── Walk-forward validation (#11) ─────────────────────────────────────────────

@app.get("/api/backtest/walkforward/{ticker}")
def get_walkforward(ticker: str, train_bars: int = 252, test_bars: int = 21):
    return walkforward.run(ticker, train_bars=train_bars, test_bars=test_bars)

# ── Multi-TF Confluence (#13) ─────────────────────────────────────────────────

@app.get("/api/confluence")
def get_confluence_all():
    watchlist = db.list_watchlist()
    symbols = ([w["symbol"] for w in watchlist] if watchlist
               else [s["symbol"] for s in STOCK_UNIVERSE.get("stocks", [])[:5]])
    results = []
    for sym in symbols[:10]:
        results.append(confluence.analyse(sym))
    return results

@app.get("/api/confluence/{ticker}")
def get_confluence(ticker: str):
    return confluence.analyse(_normalize_symbol(ticker))

# ── NSE Live Data (#15) ───────────────────────────────────────────────────────

@app.get("/api/nse/market")
def nse_market():
    return nse.market_status()

@app.get("/api/nse/indices")
def nse_indices():
    return nse.get_indices()

@app.get("/api/nse/quote/{ticker}")
def nse_quote(ticker: str):
    return nse.get_quote(ticker)

@app.get("/api/nse/quotes")
def nse_quotes():
    watchlist = db.list_watchlist()
    symbols = ([w["symbol"] for w in watchlist] if watchlist
               else [s["symbol"] for s in STOCK_UNIVERSE.get("stocks", [])[:20]])
    return nse.get_quotes(symbols)

# ── Strategy selector (#17) ───────────────────────────────────────────────────

class StrategyIn(BaseModel):
    strategy: str

@app.get("/api/strategy")
def get_strategy():
    active = db.get_config("active_strategy", strategies.DEFAULT_STRATEGY)
    if active not in strategies.STRATEGY_META:
        active = strategies.DEFAULT_STRATEGY
    return {
        "active":    active,
        "meta":      strategies.STRATEGY_META[active],
        "available": [{"key": k, **v} for k, v in strategies.STRATEGY_META.items()],
        "default":   strategies.DEFAULT_STRATEGY,
    }

@app.post("/api/strategy")
def set_strategy(body: StrategyIn):
    if body.strategy not in strategies.STRATEGY_META:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {body.strategy}")
    db.set_config("active_strategy", body.strategy)
    return {"ok": True, "active": body.strategy}

@app.post("/api/strategy/{key}")
def set_strategy_by_key(key: str):
    """Path-param variant used by the signals page switcher."""
    if key not in strategies.STRATEGY_META:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {key}")
    db.set_config("active_strategy", key)
    return {"ok": True, "active": key}

# ── WebSocket: live prices (#18) ──────────────────────────────────────────────

@app.websocket("/ws/prices")
async def ws_prices(websocket: WebSocket):
    await _ws_manager.connect(websocket)
    try:
        loop = asyncio.get_running_loop()
        watchlist = db.list_watchlist()
        symbols = ([w["symbol"] for w in watchlist] if watchlist
                   else [s["symbol"] for s in STOCK_UNIVERSE.get("stocks", [])[:20]])
        quotes, market = await asyncio.gather(
            loop.run_in_executor(None, lambda: nse.get_quotes(symbols)),
            loop.run_in_executor(None, nse.market_status),
        )
        ts = _dt.datetime.now(tz=_IST).strftime("%H:%M:%S")
        await websocket.send_text(json.dumps({
            "type": "prices", "quotes": quotes, "market": market, "ts": ts,
        }))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _ws_manager.disconnect(websocket)
    except Exception:
        _ws_manager.disconnect(websocket)
