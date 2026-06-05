import os
import time
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# ── Cache settings ──────────────────────────────────────────────────────────
MODEL_DIR        = os.path.join(os.path.dirname(__file__), "models")
RETRAIN_AFTER_H  = 24          # retrain if model is older than this many hours
os.makedirs(MODEL_DIR, exist_ok=True)

def _model_path(ticker: str)  -> str:
    return os.path.join(MODEL_DIR, f"{ticker.replace('.','_')}_lstm.keras")

def _scaler_path(ticker: str) -> str:
    return os.path.join(MODEL_DIR, f"{ticker.replace('.','_')}_scaler.pkl")

def _meta_path(ticker: str)   -> str:
    return os.path.join(MODEL_DIR, f"{ticker.replace('.','_')}_meta.pkl")

def _xgb_path(ticker: str) -> str:
    return os.path.join(MODEL_DIR, f"{ticker.replace('.','_')}_xgb.pkl")

def _xgb_scaler_path(ticker: str) -> str:
    return os.path.join(MODEL_DIR, f"{ticker.replace('.','_')}_xgb_scaler.pkl")

def _model_is_fresh(ticker: str) -> bool:
    """Return True if a saved model exists and is younger than RETRAIN_AFTER_H."""
    mp = _model_path(ticker)
    if not os.path.exists(mp):
        return False
    age_hours = (time.time() - os.path.getmtime(mp)) / 3600
    return age_hours < RETRAIN_AFTER_H

def _save_artifacts(ticker, model, scaler, meta):
    model.save(_model_path(ticker))
    with open(_scaler_path(ticker), "wb") as f:
        pickle.dump(scaler, f)
    with open(_meta_path(ticker), "wb") as f:
        pickle.dump(meta, f)
    print(f"  [cache] saved model + scaler for {ticker}")

def _load_artifacts(ticker):
    model  = load_model(_model_path(ticker))
    with open(_scaler_path(ticker), "rb") as f:
        scaler = pickle.load(f)
    with open(_meta_path(ticker), "rb") as f:
        meta = pickle.load(f)
    print(f"  [cache] loaded saved model for {ticker} (skipping retrain)")
    return model, scaler, meta

# ── Data helpers ─────────────────────────────────────────────────────────────
def get_data(ticker: str, period: str = "2y"):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.sort_index().ffill().dropna(subset=["Close"])

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA10"]  = df["Close"].rolling(10).mean()
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()

    df["H-L"]  = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"]  - df["Close"].shift(1))
    df["TR"]   = df[["H-L","H-PC","L-PC"]].max(axis=1)
    df["ATR"]  = df["TR"].rolling(14).mean()

    df["Returns"]    = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(10).std()

    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_mid
    df["BB_Pos"]   = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    df["Momentum"] = df["Close"] - df["Close"].shift(5)
    df["ROC"]      = df["Close"].pct_change(10) * 100

    df["Volume_MA"]    = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]

    df["EMA50"]       = df["Close"].ewm(span=50).mean()
    df["EMA200"]      = df["Close"].ewm(span=200).mean()
    df["HTF_Bullish"] = (df["EMA50"] > df["EMA200"]).astype(int)
    df["sweep_low"]   = (df["Low"] < df["Low"].rolling(5).min().shift(1)).astype(int)
    df["bos_up"]      = (df["Close"] > df["High"].shift(1)).astype(int)

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df.dropna()


def build_sequences(data, labels, lookback=30):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(labels[i])
    return np.array(X), np.array(y)

FEATURE_COLS = [
    "Returns","Volatility","RSI","MACD","MACD_Signal","MACD_Hist",
    "BB_Width","BB_Pos","Momentum","ROC","Volume_Ratio",
    "MA10","MA20","MA50","HTF_Bullish","sweep_low","bos_up"
]

XGB_EXTRA_COLS   = ["Ret_3d", "Ret_5d", "Ret_20d", "RSI_slope", "MACD_slope"]
XGB_FEATURE_COLS = FEATURE_COLS + XGB_EXTRA_COLS


def _add_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag/slope features used only by XGBoost (not LSTM sequences)."""
    df = df.copy()
    df["Ret_3d"]     = df["Close"].pct_change(3)
    df["Ret_5d"]     = df["Close"].pct_change(5)
    df["Ret_20d"]    = df["Close"].pct_change(20)
    df["RSI_slope"]  = df["RSI"].diff(5)
    df["MACD_slope"] = df["MACD"].diff(3)
    return df.dropna()


def _train_or_load_xgb(ticker: str, df: pd.DataFrame) -> float | None:
    """Train or load XGBoost model. Returns probability for the latest bar (0-1)."""
    if not XGBOOST_AVAILABLE:
        return None

    xp  = _xgb_path(ticker)
    xsp = _xgb_scaler_path(ticker)

    df_xgb = _add_xgb_features(df)
    X = df_xgb[XGB_FEATURE_COLS].values
    y = df_xgb["Target"].values
    split = int(len(X) * 0.8)
    if split < 30:
        return None

    # Load from disk if fresh
    if os.path.exists(xp):
        age_h = (time.time() - os.path.getmtime(xp)) / 3600
        if age_h < RETRAIN_AFTER_H:
            try:
                with open(xp,  "rb") as f: xgb_model = pickle.load(f)
                with open(xsp, "rb") as f: scaler    = pickle.load(f)
                X_scaled = scaler.transform(X)
                prob = float(xgb_model.predict_proba(X_scaled[-1:])[0][1])
                print(f"  [cache] loaded XGB for {ticker}")
                return prob
            except Exception:
                pass

    try:
        scaler   = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        pos      = int(y[:split].sum())
        neg      = split - pos
        spw      = neg / pos if pos else 1.0

        xgb_model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss", random_state=42,
            verbosity=0,
        )
        xgb_model.fit(X_scaled[:split], y[:split])

        with open(xp,  "wb") as f: pickle.dump(xgb_model, f)
        with open(xsp, "wb") as f: pickle.dump(scaler,    f)
        print(f"  [train] XGB trained for {ticker}")

        prob = float(xgb_model.predict_proba(X_scaled[-1:])[0][1])
        return prob
    except Exception as e:
        print(f"  [warn] XGB failed for {ticker}: {e}")
        return None


# ── Trade plan ────────────────────────────────────────────────────────────────
def build_trade_plan(df: pd.DataFrame, signal: str, confidence: float) -> dict:
    entry = float(df["Close"].iloc[-1])
    atr   = float(df["ATR"].iloc[-1])

    if signal == "SELL":
        stop_loss = entry + 1.5 * atr
        target1   = entry - 2   * atr
        target2   = entry - 4   * atr
    else:
        stop_loss = entry - 1.5 * atr
        target1   = entry + 2   * atr
        target2   = entry + 4   * atr

    risk   = abs(entry - stop_loss)
    reward = abs(target1 - entry)
    rr_ratio = round(reward / risk, 2) if risk else 0

    trade_quality = (
        "EXCELLENT" if rr_ratio >= 2 else
        "GOOD"      if rr_ratio >= 1.5 else
        "AVERAGE"   if rr_ratio >= 1 else
        "LOW"
    )
    return {
        "entry": round(entry, 2), "stop_loss": round(stop_loss, 2),
        "target1": round(target1, 2), "target2": round(target2, 2),
        "risk_reward": rr_ratio, "trade_quality": trade_quality,
        "atr": round(atr, 2),
    }

# ── Main prediction function ──────────────────────────────────────────────────
def train_lstm(ticker: str):
    if not TENSORFLOW_AVAILABLE:
        return _rf_fallback(ticker)

    try:
        # ── LOAD from disk if fresh, skip training entirely ──
        if _model_is_fresh(ticker):
            model, scaler, meta = _load_artifacts(ticker)
            df = get_data(ticker)
            df = add_features(df)
            X_raw    = df[FEATURE_COLS].values
            X_scaled = scaler.transform(X_raw)        # use SAVED scaler, not refit
            X_last   = X_scaled[-30:].reshape(1, 30, len(FEATURE_COLS))
            lstm_conf = float(model.predict(X_last, verbose=0)[0][0])
            xgb_prob  = _train_or_load_xgb(ticker, df)
            if xgb_prob is not None:
                last_conf  = 0.55 * lstm_conf + 0.45 * xgb_prob
                model_name = "LSTM+XGBoost"
            else:
                last_conf  = lstm_conf
                model_name = "LSTM"
            signal     = "BUY" if last_conf > 0.57 else "SELL" if last_conf < 0.43 else "HOLD"
            trade_plan = build_trade_plan(df, signal, last_conf)
            return {
                **meta,
                "signal":     signal,
                "confidence": round(last_conf * 100),
                "model":      model_name,
                "lstm_conf":  round(lstm_conf * 100, 1),
                "xgb_conf":   round(xgb_prob * 100, 1) if xgb_prob is not None else None,
                **trade_plan,
            }

        # ── TRAIN from scratch (first run or model expired) ──
        print(f"  [train] training new model for {ticker}...")
        df = get_data(ticker)
        df = add_features(df)

        X_raw = df[FEATURE_COLS].values
        y_raw = df["Target"].values

        scaler   = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_raw)

        X_seq, y_seq = build_sequences(X_scaled, y_raw, 30)
        split = int(len(X_seq) * 0.8)
        if split < 50:
            return _rf_fallback(ticker)

        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(30, len(FEATURE_COLS))),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1,  activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(
            X_train, y_train, epochs=50, batch_size=32,
            validation_split=0.1,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
            verbose=0
        )

        proba    = model.predict(X_test, verbose=0).flatten()
        accuracy = float(accuracy_score(y_test, (proba > 0.5).astype(int)))

        lstm_conf = float(proba[-1])

        # ── XGBoost blend ──────────────────────────────────────────────
        xgb_prob = _train_or_load_xgb(ticker, df)
        if xgb_prob is not None:
            last_conf  = 0.55 * lstm_conf + 0.45 * xgb_prob
            model_name = "LSTM+XGBoost"
        else:
            last_conf  = lstm_conf
            model_name = "LSTM"

        signal    = "BUY" if last_conf > 0.57 else "SELL" if last_conf < 0.43 else "HOLD"
        trade_plan = build_trade_plan(df, signal, last_conf)

        meta = {
            "accuracy":   round(accuracy * 100, 1),
            "model":      model_name,
            "lstm_conf":  round(lstm_conf * 100, 1),
            "xgb_conf":   round(xgb_prob * 100, 1) if xgb_prob is not None else None,
        }

        # ── SAVE model + scaler + meta to disk ──
        _save_artifacts(ticker, model, scaler, meta)

        return {
            "signal":     signal,
            "confidence": round(last_conf * 100),
            **meta, **trade_plan
        }

    except Exception as e:
        print(f"  [error] LSTM failed for {ticker}: {e}")
        return _rf_fallback(ticker)


# ── Random Forest fallback (also saved to disk) ───────────────────────────────
def _rf_fallback(ticker: str):
    from sklearn.ensemble import RandomForestClassifier

    rf_path = os.path.join(MODEL_DIR, f"{ticker.replace('.','_')}_rf.pkl")
    sc_path = os.path.join(MODEL_DIR, f"{ticker.replace('.','_')}_rf_scaler.pkl")

    try:
        df = get_data(ticker)
        df = add_features(df)
        X  = df[FEATURE_COLS].values
        y  = df["Target"].values
        split = int(len(df) * 0.8)

        # Load saved RF if fresh
        if os.path.exists(rf_path):
            age_h = (time.time() - os.path.getmtime(rf_path)) / 3600
            if age_h < RETRAIN_AFTER_H:
                with open(rf_path, "rb") as f:
                    rf_model = pickle.load(f)
                with open(sc_path, "rb") as f:
                    scaler = pickle.load(f)
                print(f"  [cache] loaded saved RF for {ticker}")
                X_scaled  = scaler.transform(X)
                last_conf = float(rf_model.predict_proba(X_scaled[-1:])[0][1])
                proba_all = rf_model.predict_proba(X_scaled[split:])
                accuracy  = float(np.mean((proba_all[:,1] > 0.5).astype(int) == y[split:]))
                signal    = "BUY" if last_conf > 0.55 else "SELL" if last_conf < 0.45 else "HOLD"
                trade_plan = build_trade_plan(df, signal, last_conf)
                return {"signal": signal, "confidence": round(last_conf*100),
                        "accuracy": round(accuracy*100,1), "model": "RandomForest", **trade_plan}

        # Train new RF
        scaler   = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        rf_model.fit(X_scaled[:split], y[:split])

        with open(rf_path, "wb") as f:
            pickle.dump(rf_model, f)
        with open(sc_path, "wb") as f:
            pickle.dump(scaler, f)

        proba     = rf_model.predict_proba(X_scaled[split:])
        last_conf = float(proba[-1][1])
        accuracy  = float(np.mean((proba[:,1] > 0.5).astype(int) == y[split:]))
        signal    = "BUY" if last_conf > 0.55 else "SELL" if last_conf < 0.45 else "HOLD"
        trade_plan = build_trade_plan(df, signal, last_conf)
        return {"signal": signal, "confidence": round(last_conf*100),
                "accuracy": round(accuracy*100,1), "model": "RandomForest", **trade_plan}

    except Exception as e:
        print(f"  [error] RF fallback failed: {e}")
        return {"signal": "HOLD", "confidence": 50, "accuracy": 50, "model": "error"}


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for t in ["RELIANCE.NS", "TCS.NS"]:
        print(f"\n{'='*40}")
        print(f"Ticker: {t}")
        r = train_lstm(t)
        print(f"  Model     : {r['model']}")
        print(f"  Signal    : {r['signal']}")
        print(f"  Confidence: {r['confidence']}%")
        print(f"  Accuracy  : {r['accuracy']}%")
        print(f"  Entry     : {r.get('entry')}  SL: {r.get('stop_loss')}  T1: {r.get('target1')}")
        print(f"  R:R       : {r.get('risk_reward')}  Quality: {r.get('trade_quality')}")

    print("\n--- Second call (should load from cache, not retrain) ---")
    r2 = train_lstm("RELIANCE.NS")
    print(f"  Signal: {r2['signal']}  Confidence: {r2['confidence']}%")