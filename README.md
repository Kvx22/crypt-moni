# Crypto Risk Board: a 24/7 crypto scanner with five learning advisors

The Risk Board, rebuilt for crypto. Crypto never closes, so the board never adjourns: a scanner reads the market every
minute, the same five advisors grade their own calls every hour and learn from what actually happened, and the
Control Room dashboard shows all of it live: the regime, the votes, the picks, the risk on every coin, and how
each advisor's thinking is drifting.

```
Coinbase (Kraken fallback) ─► SCANNER: 12 coins, every 60 s, 24/7
                               metrics · fixed risk score · alerts
                                        │
  BTC daily closes ─► HMM REGIME BRAIN (crash · bear · neutral · bull · euphoria)
                                        │
             ┌──────────────────────────┴───────────────────────────┐
     EXPOSURE BOARD (daily)                                 PICK DESK (hourly)
     how much risk to run                                   which coins
     Kelly per regime + Hedge weights                       ridge-to-instinct learning + Hedge weights
     chair caps the board at her vote + 35 pts              chair vetoes BUY on EXTREME risk
             └──────────────────────────┬───────────────────────────┘
                                 paper allocation
                                        │
                SAFETY LAYER (deterministic, no model, paper-only)
                kill switch · circuit breaker · order gate · watchdog
                                        │
                     paper broker (fees + slippage), nothing real
everything ─► SQLite ─► Control Room dashboard at http://127.0.0.1:8765
```

**Paper by default.** Out of the box it trades a paper book only. **Live mode** (see [EXECUTION.md](EXECUTION.md))
turns the advisors' verdict into real orders on Coinbase or your Robinhood agentic account, and **every order waits
for your approval** in the dashboard, on your phone, or in the terminal. There is no auto-approve. Live mode also needs
two arming keys (a config flag and an environment variable) and a per-trade cap you set yourself.

## Quick start

Python 3.10+ and numpy. From inside the `crypt-moni` folder (Windows: use `py` instead of `python`):

```bash
pip install -r requirements.txt
python -m unittest discover -s tests          # 43 tests, ~40 s

# See it work offline first: three weeks of SIMULATED market in ~45 s
python run.py demo
python run.py dashboard --config config.sim.json     # http://127.0.0.1:8766

# The real thing: live Coinbase data, scanner + dashboard in one process
python run.py doctor                          # READY?
python run.py up                              # http://127.0.0.1:8765
```

On first start the scanner pulls ~12 days of hourly bars per coin, replays the last 365 days of BTC through the
brain (about 15 s) so the exposure board has learned something before it goes live, and the first board convenes
at the next closed hourly bar.

Watch it learn in fast-forward: `python run.py up --config config.sim.json` runs the simulator at one market hour
every 2 seconds with the dashboard open on port 8766.

### Trade it for real, with your approval

Read [EXECUTION.md](EXECUTION.md). The short version: Coinbase connects through its official API and sends what you
approve within a minute, day or night; Robinhood's agentic account is only reachable through Claude's connector, so
approved Robinhood orders wait for a Claude session. For a true 24/7 setup: Coinbase, on an always-on machine,
dashboard on your phone over Tailscale, push notifications from ntfy when the board wants your approval.

### Stocks too (Robinhood board)

`config.robinhood.json` adds a **stock sleeve** (40% of the account by default) beside the 12 coins: SPY, IBIT, GLD,
XLE, XLV, GDX, COPX, ITA, URA and VTIP. It is the same five advisors on a stock clock: a regime brain on SPY, a
separate stock exposure board and stock pick desk that learn from completed trading days (the crypto and stock desks
never mix), proposals once each trading morning from 10:00 ET, orders only while the US market is open. Stock data
comes from Yahoo Finance's free public endpoint (unofficial; if it ever blocks you, the scanner keeps running and
stock rows just go stale). The dashboard shows stocks in the risk board (filter: All / Crypto / Stocks), a stock-market
brain card, and a Crypto desk / Stock desk toggle on the learning view. Details in [EXECUTION.md](EXECUTION.md).

### Run it 24/7

| Where | How |
|---|---|
| Any machine with Docker | `docker compose up -d --build`: scanner, dashboard and watchdog as three auto-restarting services on one data volume. Dashboard on http://127.0.0.1:8765 |
| Linux server / Raspberry Pi | `deploy/crypt-moni.service` + `deploy/crypt-moni-watchdog.service` (systemd, restart on failure). Steps in the file header |
| Windows PC | double-click `deploy/windows-24-7.bat` (restarts the scanner if it ever exits); add it to Task Scheduler "At log on" to survive reboots. Keep the PC from sleeping |

A home PC that sleeps is not 24/7. A small always-on box (old laptop, Pi, $5 VPS) is the honest way to run it.
If you put the dashboard on a server, keep it behind SSH tunnelling or a VPN: it has no login.

## Commands

| | |
|---|---|
| `up` | scanner + dashboard in one process |
| `scan` / `dashboard` / `watchdog` | the three pieces separately (what Docker runs) |
| `demo [--hours N]` | offline simulation into `data/sim.db` |
| `once`, `status` | one tick and print the board / print the latest board |
| `export [--out FILE]` | the dashboard as one self-contained, read-only HTML snapshot you can share |
| `doctor` | config, data, feed reachability, safety settings |
| `backtest [--days N]` | walk-forward test of brain + exposure board on BTC daily closes |
| `kill`, `reset-kill`, `reset-breaker` | safety controls. Only a human resets |
| `outbox`, `approve ID\|all`, `reject ID` | live mode: review and decide proposed orders from the terminal |
| `export-approved`, `ack`, `rh-snapshot` | live Robinhood: the hand-off Claude uses (EXECUTION.md) |

## The five advisors

Original characters written to stand for five investing styles, the same board (and faces) as *The Risk Board*.
Not real people. Their lines in the dashboard are generated from the numbers.

| | Style | Exposure instinct (crash → euphoria) | Pick-desk instinct | Open-mindedness |
|---|---|---|---|---|
| **Vera Holt**, The Contrarian | fades crowds | 55 / 25 / 50 / 40 / 10 % | buys losers: low RSI, deep drawdown, negative 24h | 0.5 (learns fastest) |
| **Kai Arden**, The Founder | rides trends | 15 / 35 / 70 / 95 / 100 % | momentum, breakouts, volume | 0.2 (stubborn) |
| **Imani Okoro**, Chair · Risk Officer | sizes to survive | 0 / 15 / 50 / 70 / 50 % | low volatility, liquid, shallow drawdowns | 0.3 |
| **Marguerite Lowe**, The Value Compounder | patience | 40 / 55 / 60 / 60 / 30 % | liquid names in uptrends, bought on weekly dips | 0.4 |
| **Idris Calder**, The Macro Strategist | reads the regime | 5 / 25 / 55 / 80 / 60 % | high beta when the brain leans risk-on, low beta when it leans risk-off | 0.4 |

### How they adapt while the scanner scans

**Pick desk, every hour.** The scanner turns each coin into 11 signals (24h and 7-day momentum, last-hour move,
trend vs 3-day average, position in the 7-day range, RSI, volume surge, volatility, drawdown, liquidity, BTC beta ×
regime), standardised across coins so learning is about *which coin beats the others*. Each advisor scores every
coin with a weight vector over those signals. Every hour:

1. Their calls from 4 hours ago are graded against what actually happened: rank correlation (IC), hit rate of
   their top 3, and their top 3's return over the field. The calls being graded were made with the weights they
   had *then*, so the grade is honest out-of-sample.
2. **They learn.** Weights are re-solved by ridge regression on all graded outcomes, pulled toward each advisor's
   instinct. Stubborn advisors are pulled harder: once the memory is full (about three weeks), Kai's view is still
   about two-thirds instinct and Vera's under a third.
   Old evidence fades (decay 0.995/hour, about an 8-day memory), so when the market changes character, they do too.
3. **Desk weights move** by Hedge on each advisor's loss-averse pick payoff, with a fixed-share floor so nobody is
   ever silenced. The dashboard charts these weights over time and shows, per advisor, every signal's weight now vs
   their instinct (the "drift").

The desk's combined score gives each coin a verdict: BUY, WATCH, NEUTRAL, AVOID, or **VETOED** when Imani blocks a
BUY on a coin whose fixed risk score is EXTREME.

**Exposure board, every UTC day.** Ported unchanged from the Risk Board: each advisor turns the brain's regime
probabilities into an exposure vote, learns a per-regime Kelly exposure from realised BTC returns (blended in by
open-mindedness × data confidence), Hedge weights move with realised payoff, and the chair can't be out-voted past
her own vote + 35 points.

## What else is in the box

**Regime brain (`regime.py`)**: 5-state Gaussian HMM in pure numpy (EM + causal forward filter, tested for no
look-ahead) on BTC daily features: 1-day return, 10-day vol, 30-day return, drawdown from the 365-day high, stretch
over the 200-day average. States are named worst→best by health. Refits every 90 days (warm start); a regime switch
needs ≥55% two days in a row. Between daily closes the dashboard also shows a live read with today's price as a
provisional close (it never moves the official state). Seeded with real BTC-USD daily closes 2015-07-20 → 2026-10-15
(`data/BTC-USD.csv`); the scanner appends each completed UTC day.

**Risk score (`scanner.py`)**: fixed arithmetic, not learned, so nobody can talk it down. 0-100 from volatility (30%),
drawdown (20%), volatility expansion (15%), tail moves (15%), thin liquidity (10%), high beta (10%). LOW < 25 ≤
ELEVATED < 45 ≤ HIGH < 65 ≤ EXTREME.

**Alerts**: |hourly move| ≥ 3σ, breaks of the 7-day high or low, volume ≥ 3× its weekly average, RSI ≥ 80 or ≤ 20, a
coin's risk tier jumping to HIGH or EXTREME. Each alert type has a cooldown so a lasting condition doesn't spam.

**Paper book (`allocation.py`, `paper.py`)**: invested fraction = exposure-board vote → shrink toward neutral when
the regime is uncertain (doubt only lowers it) → vol-target on BTC → breaker cap → rate limit (de-risk 2× faster).
Coins: the desk's BUY/WATCH names by smoothed (~12 h) conviction, held coins get a small stickiness bonus,
per-coin caps (BTC 50%, ETH 35%, others 15%), leftovers to the BTC core (unless the desk rates BTC AVOID) or cash.
Rebalances every 8 h, 40 bp fee + 5 bp slippage.

**Safety layer (`safety.py`, `watchdog.py`)**: imports nothing from the brain (a test enforces it), no learned parameters.

| Guard | Behaviour |
|---|---|
| Human approval | live orders only leave the outbox after you approve them; approved orders are re-checked (price within ±2% of the proposal, every guard below) before sending; unapproved ones expire after 2 h |
| Two arming keys | live mode needs `safety.live_armed: true` AND env `CM_LIVE=1`; otherwise nothing is proposed |
| Kill switch | the `data/KILL` file. Dashboard button, watchdog, `run.py kill`, or 5 consecutive engine errors create it; only `reset-kill` clears it |
| Per-trade cap | unset ⇒ every order refused; over-cap orders clamped (or rejected) |
| Circuit breaker | drawdown from a slowly decaying high-water mark: 10% → cap 60% · 15% → cap 30% · 25% → HALT (sells only, latched until `reset-breaker`); day loss ≥ 5% (UTC) blocks buys |
| Order gate | allowlist · no shorting · position caps · 2% cash reserve · daily turnover 60% · 60 orders/day · ±2% price band · quotes older than 5 min refused · 3 h cooldown per coin and side |
| Watchdog | separate process; trips the kill switch if the scanner heartbeat is 15 min stale (no market-hours exemption: crypto doesn't close) or paper drawdown hits 30% |
| Feed failures | back off up to 10 min and keep scanning; never trade on stale data |

## Results so far (honest)

Brain + exposure board, walk-forward on real BTC-USD daily closes, 2019-09-03 → 2026-10-15 (`python run.py backtest
--days 2600`): refit every 90 days on past data only, board learns online, vote applied the next day, cash at 4%.

| | Board | Buy & hold BTC |
|---|---|---|
| CAGR | +29.2% | +33.3% |
| Volatility | 31% | 61% |
| Sharpe (4% cash) | 0.69 | 0.40 |
| Max drawdown | −33% | −77% |

Average exposure 50%. It is a *defensive* system: it buys a much shallower drawdown with some upside given up, and
it lags in strong bull runs. One asset, settings chosen once, history not a forecast.

The pick desk has no long real-data test yet: the sandbox this was built in could not reach exchange APIs, so the
scanner and desk were exercised on the simulator (which plants a 24h momentum signal that flips to mean reversion
every 10 days, so you can watch the advisors trade the lead). Its live record starts the day you run it, and the
dashboard grades it every hour in public.

## Known limits

- Hand-set instincts and thresholds. The brain only sees BTC price features (no funding rates, flows, on-chain data).
- Coinbase omits an hourly candle when nothing traded; such a coin skips that hour's grading.
- The 1h column on the live board is the move since the last hourly close.
- Analysis and simulation tool, not financial advice. Decisions and risk are yours.
