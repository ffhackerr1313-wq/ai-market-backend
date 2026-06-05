# =========================================
# 📊 ICT / SMC STRATEGY (V3 - ACTIVE)
# =========================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------
# SETTINGS
# -----------------------------------------
ticker = "RELIANCE.NS"   # You can change this
initial_capital = 100000

# -----------------------------------------
# FETCH DATA
# -----------------------------------------
df = yf.download(ticker, start="2024-01-01", end=None)

# Fix MultiIndex issue
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.sort_index()
df = df.asfreq('B')
df = df.ffill()

# -----------------------------------------
# TECHNICAL ANALYSIS
# -----------------------------------------
price = float(df["Close"].iloc[-1])
returns = df["Close"].pct_change().dropna()
mom = float((df["Close"].iloc[-1] - df["Close"].iloc[-6]) / df["Close"].iloc[-6] * 100)

# VOLUME ANALYSIS
current_volume = float(df["Volume"].iloc[-1])
avg_volume = float(df["Volume"].rolling(20).mean().iloc[-1])

volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

volume_spike = volume_ratio > 1.8

# BREAKOUT DETECTION
high_20 = float(df["High"].rolling(20).max().iloc[-2])
low_20  = float(df["Low"].rolling(20).min().iloc[-2])

breakout_up = price > high_20
breakout_down = price < low_20

# SCANNER ENGINE
scanner_signal = "NEUTRAL"
breakout_strength = "WEAK"

if breakout_up and volume_spike:
    scanner_signal = "MOMENTUM BUY"
    breakout_strength = "STRONG"

elif breakout_up:
    scanner_signal = "BREAKOUT BUY"
    breakout_strength = "MEDIUM"

elif breakout_down and volume_spike:
    scanner_signal = "BREAKDOWN SELL"
    breakout_strength = "STRONG"

elif breakout_down:
    scanner_signal = "BREAKDOWN"
    breakout_strength = "MEDIUM"

# -----------------------------------------
# INDICATORS (TREND)
# -----------------------------------------
df['EMA50'] = df['Close'].ewm(span=50).mean()
df['EMA200'] = df['Close'].ewm(span=200).mean()

# -----------------------------------------
# SIGNAL LOGIC (ICT V3)
# -----------------------------------------
df['Signal'] = 0

for i in range(1, len(df)):

    row = df.iloc[i]
    prev = df.iloc[i-1]

    # -------------------------
    # BULLISH TREND
    # -------------------------
    if row['EMA50'] > row['EMA200']:

        # Break of Structure
        if row['Close'] > prev['High']:

            # Pullback near EMA50 (2% zone)
            if abs(row['Close'] - row['EMA50']) / row['Close'] < 0.02:
                df.at[df.index[i], 'Signal'] = 1

    # -------------------------
    # BEARISH TREND
    # -------------------------
    elif row['EMA50'] < row['EMA200']:

        if row['Close'] < prev['Low']:

            if abs(row['Close'] - row['EMA50']) / row['Close'] < 0.02:
                df.at[df.index[i], 'Signal'] = -1

# -----------------------------------------
# BACKTEST (REALISTIC SIMPLE)
# -----------------------------------------
cash = initial_capital
position = 0
entry_price = 0

portfolio = []

for i in range(len(df)):
    row = df.iloc[i]

    price = float(row['Close'])
    signal = int(row['Signal'])

    # BUY
    if signal == 1 and position == 0:
        position = cash / price
        entry_price = price
        cash = 0

    # SELL
    elif signal == -1 and position > 0:
        cash = position * price
        position = 0

    total_value = cash + position * price
    portfolio.append(total_value)

df['Portfolio'] = portfolio

# -----------------------------------------
# RESULTS
# -----------------------------------------
final_value = df['Portfolio'].iloc[-1]
strategy_return = (final_value - initial_capital) / initial_capital
buy_hold = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1

if __name__ == "__main__":
    print(f"\n🚀 Running for {ticker}")
    print(f"💵 Final Value: {final_value:.2f}")
    print(f"📈 Strategy Return: {strategy_return:.2%}")
    print(f"📊 Buy & Hold Return: {buy_hold:.2%}")

# -----------------------------------------
# PLOT PORTFOLIO
# -----------------------------------------
plt.figure(figsize=(12,6))

plt.plot(df.index, df['Portfolio'], label='Strategy')

normalized = df['Close'] / df['Close'].iloc[0] * initial_capital
plt.plot(df.index, normalized, label='Buy & Hold')

plt.legend()
plt.title("ICT Strategy V3 (Active Trading)")
plt.grid()
plt.show()

# -----------------------------------------
# PLOT SIGNALS
# -----------------------------------------
plt.figure(figsize=(12,6))

plt.plot(df.index, df['Close'], label='Price')

buy = df[df['Signal'] == 1]
sell = df[df['Signal'] == -1]

plt.scatter(buy.index, buy['Close'], marker='^', label='BUY')
plt.scatter(sell.index, sell['Close'], marker='v', label='SELL')

plt.legend()
plt.title("ICT Signals (V3)")
plt.grid()
plt.show()
