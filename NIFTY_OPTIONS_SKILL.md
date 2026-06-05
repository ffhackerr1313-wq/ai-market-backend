# Nifty 50 Options Trading System — Master Skill
> Copy this file to: C:\Users\atul\.claude\skills\nifty-options.md
> Then it loads automatically in every Claude Code session for this project.

---

## Project Identity
- **Backend:** `C:\major project` (FastAPI + Python)
- **Frontend:** `~/ai-market-ui` (Next.js + React)
- **Goal:** Build automated Nifty 50 options trading engine targeting ₹20,000–50,000/day
- **Phase 2 equity features done** — items #0-17 shipped. See memory/priorities-done.md before touching existing code.

---

## Current System Gaps (as of 2026-06-01)
The existing system trades Nifty 50 **equity stocks** with delayed yfinance data. For options:
- No option chain data, no Greeks, no IV, no strikes
- No broker API connection
- No real-time data (15-30 min delay = fatal for options)
- 0DTE/intraday signals need <60s latency minimum

---

## 5 Strategies to Implement (Priority Order)

### 1. Iron Condor / Short Strangle (PRIMARY — most consistent)
```
Sell OTM Call + Put 100 pts from ATM → buy wings 200 pts away
Win rate: 65-72% | Weekly: ₹1,500-3,000/lot | 5 lots = ₹30k-60k/month
Rule: Only when India VIX < 16. Always buy wings. Never naked sell.
```

### 2. Thursday 0DTE Straddle Sell
```
Sell ATM CE + PE at 9:30 AM Thursday → close at 2:00-2:30 PM
Win rate: 58-65% | Premium collected: ₹80-120 → decay to ₹10-25
Rule: SKIP on RBI/Budget/election Thursdays
```

### 3. Opening Range Breakout (ORB) — feeds from existing EMA signal
```
Mark 9:15-9:45 AM Nifty high/low
Buy ATM CE on upside break, ATM PE on downside break
Win rate: 50-55% with 1:2 R/R = positive expectancy
SL: 40% of premium. Target: 80-100% gain.
EMA Crossover on ^NSEI 15m → directly feeds this strategy
```

### 4. VWAP + PCR Directional (adapts existing VWAP_RSI)
```
Price > VWAP + RSI > 55 + PCR > 1.2 → buy ATM CE
Price < VWAP + RSI < 45 + PCR < 0.8 → buy ATM PE
PCR from NSE option chain API (free)
Win rate: 48-52%, needs 1:2 R/R strictly
```

### 5. IV Crush after Events
```
Sell straddle before RBI/Budget when VIX = 18-25
Close morning after event when IV collapses 5-8 pts
Win rate: 60-70% on event-specific entries
```

---

## Capital → Daily P&L Map

| Capital | Lots | Realistic Daily Avg | Monthly Net |
|---------|------|-------------------|-------------|
| ₹2 lakh | 1 lot | ₹2,000-3,000 | ₹44k-66k |
| ₹5 lakh | 3 lots | ₹5,000-8,000 | ₹1.1L-1.7L |
| ₹10 lakh | 5-7 lots | ₹10,000-15,000 | ₹2.2L-3.3L |
| ₹15 lakh | 10-12 lots | ₹20,000-30,000 | ₹4.4L-6.6L |
| ₹25 lakh | 20+ lots | ₹35,000-50,000 | ₹7.7L-11L |

---

## Broker: Angel One SmartAPI (recommended)
- **Cost:** Free API
- **Auth:** TOTP via `pyotp` — fully automatable, no daily browser login
- **Data:** Real-time tick-by-tick WebSocket
- **Option chain:** Full chain + IV + Greeks in API response
- **Alternative:** Shoonya (Finvasia) for zero brokerage if 10+ trades/day

---

## Hardcoded Risk Rules (never skip these)
```python
DAILY_LOSS_LIMIT     = -8_000   # Hard stop ALL trading
MAX_TRADES_PER_DAY   = 4
MAX_OPEN_POSITIONS   = 2
OPTION_BUY_SL_PCT    = 0.40     # Exit if premium drops 40%
OPTION_SELL_SL_MULT  = 2.0      # Exit if premium doubles
FORCE_EXIT_TIME      = "15:15"  # All positions squared off
NO_TRADE_BEFORE      = "09:30"  # Skip 9:15-9:30 (IV too high)
VIX_SELL_MAX         = 18.0     # No selling premium above this
VIX_BUY_MIN          = 11.0     # No buying options below this
MIN_DTE_FOR_BUYING   = 2
```

---

## What Carries Forward From Current Codebase

| Existing File | Role in Options System |
|--------------|----------------------|
| `strategies.py` EMA_CROSSOVER | Nifty 15m directional bias → ORB signal |
| `strategies.py` VWAP_RSI | VWAP + PCR gate for directional entries |
| `ict.py` BOS/FVG | Trend confirmation for ORB direction |
| `kelly.py` | Adapt to lot-count sizing |
| `nse.py` | Extend for option-chain-indices endpoint |
| `alerts.py` Telegram | Options trade alerts (strike, premium, Greeks) |
| `db.py` | Add trades + daily_pnl + kill_switch tables |
| React signals page | Becomes `/options` dashboard |

---

## New Files to Build (in order)

### Phase 0 — Foundation (Weeks 1-2, paper mode)
```
options_chain.py    NSE option chain: OI, IV, LTP, PCR, max pain, ATM strike
greeks.py           Black-Scholes via mibian: Delta/Gamma/Theta/Vega/IV rank
db.py               Add: trades table, daily_pnl table, kill_switch flag
/api/options/chain  New FastAPI endpoint
/options page       Frontend: chain viewer + trade recommendation card
```

### Phase 1 — Signals (Weeks 3-6, paper trade)
```
options_signal.py   EMA+VWAP on ^NSEI + PCR gate → CE/PE entry signal
options_sizing.py   Lot count calculator
/api/options/signal New endpoint
```
**Milestone: 55%+ win rate on paper before any real money**

### Phase 2 — Live small (Months 2-3, ₹1-2L, 1 lot)
```
Angel One SmartAPI WebSocket → replaces yfinance for ^NSEI
Max ₹3,000 risk/day. Every trade logged to DB.
```

### Phase 3 — Scale (Months 4-6, ₹5-8L, 3-5 lots)
```
execution.py        Auto-order placement via SmartAPI
risk_manager.py     Daily loss circuit breaker + VIX gate
```

### Phase 4 — Full scale (Month 7+, ₹15-20L, 8-12 lots, ₹20k+/day)
```
position_risk.py    Portfolio delta/theta roll-up
expiry_manager.py   Weekly roll + expiry calendar
```

---

## NSE Option Chain API
```
URL: https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY
Auth: Cookie session — nse.py already handles this
Returns: All strikes, CE/PE OI, change in OI, IV, LTP, delta, theta, gamma
Poll: Every 3 minutes during market hours (9:15 AM – 3:30 PM IST)
```

---

## Monthly Cost (Phase 3+)
| Item | Cost |
|------|------|
| Angel SmartAPI | ₹0 |
| Brokerage (~150 orders) | ₹3,000 |
| AWS Mumbai VPS | ₹1,500 |
| STT + NSE + SEBI fees | ₹6,000-8,000 |
| **Total** | **~₹11,000/month** |

---

## Start Here
The single first step: **Open Angel One account → enable SmartAPI → get TOTP secret.**
Then build `options_chain.py` using the NSE endpoint above (nse.py session already works).
Everything else builds on that.
