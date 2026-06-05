import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

def get_data(ticker: str, period: str = "2y"):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.sort_index().ffill().dropna(subset=["Close"])

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["MA10"]  = df["Close"].rolling(10).mean()
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
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

def train_lstm(ticker: str):
    if not TENSORFLOW_AVAILABLE:
        return _rf_fallback(ticker)
    try:
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
        model.fit(X_train, y_train, epochs=50, batch_size=32,
                  validation_split=0.1,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
                  verbose=0)
        proba     = model.predict(X_test, verbose=0).flatten()
        accuracy  = float(accuracy_score(y_test, (proba > 0.5).astype(int)))
        last_conf = float(proba[-1])
        signal    = "BUY" if last_conf > 0.55 else "SELL" if last_conf < 0.45 else "HOLD"
        return {"signal": signal, "confidence": round(last_conf*100), "accuracy": round(accuracy*100,1), "model": "LSTM"}
    except Exception as e:
        return _rf_fallback(ticker)

def _rf_fallback(ticker: str):
    from sklearn.ensemble import RandomForestClassifier
    try:
        df = get_data(ticker)
        df = add_features(df)
        X  = df[FEATURE_COLS].values
        y  = df["Target"].values
        split = int(len(df) * 0.8)
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X[:split], y[:split])
        proba     = model.predict_proba(X[split:])
        last_conf = float(proba[-1][1])
        accuracy  = float(np.mean((proba[:,1]>0.5).astype(int) == y[split:]))
        signal    = "BUY" if last_conf > 0.55 else "SELL" if last_conf < 0.45 else "HOLD"
        return {"signal": signal, "confidence": round(last_conf*100), "accuracy": round(accuracy*100,1), "model": "RandomForest"}
    except:
        return {"signal": "HOLD", "confidence": 50, "accuracy": 50, "model": "error"}

if __name__ == "__main__":
    for t in ["AAPL", "TCS.NS"]:
        print(f"\nTraining {t}...")
        r = train_lstm(t)
        print(f"  {r['model']} | {r['signal']} | {r['confidence']}% conf | {r['accuracy']}% acc")
