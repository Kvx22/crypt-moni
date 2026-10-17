# Live trading: the board proposes, you approve

Paper mode is the default and needs none of this. Live mode turns the advisors' verdict into **real orders that
wait for your approval**. Nothing is ever sent without it: there is no auto-approve setting.

```
advisors' verdict (every 8 h) ─► orders that pass the safety gate ─► OUTBOX (pending, expire after 2 h)
                                                                         │  you: Approve / Reject
                                                                         ▼  (dashboard, phone, or CLI)
       Coinbase: the scanner re-checks each approved order the next minute ─► market order via the official API
       Robinhood: waits for a Claude session to place it through the Robinhood connector, then `ack`
```

Re-checks at send time, every time: kill switch, both arming keys, per-trade cap, price moved no more than ±2%
since the proposal, cash and position caps, circuit breaker, daily order and turnover limits, quote freshness,
3-hour cooldown per coin and side. An order that fails any check is marked *skipped* with the reason, never sent.
A send whose outcome is unknown (network cut mid-request) is marked *check* and is never retried automatically.

## Which account for 24/7

| | Coinbase (recommended for 24/7) | Robinhood agentic account |
|---|---|---|
| How it connects | Coinbase's official Advanced Trade API, from the scanner itself | Only through Claude's Robinhood connector; no program on your machine can reach it |
| When approved orders go out | Within a minute, any hour, while you sleep | When you next ask Claude to place them |
| Balances | Read live from the API every minute | As of Claude's last sync (proposals stop when older than 24 h) |
| Real-time drawdown breaker | Yes, on the real account | Only as fresh as the last sync |
| Stocks | No (Coinbase is crypto only) | Yes: a stock sleeve on its own daily clock, market hours only |

**The 24/7 setup:** Coinbase on an always-on machine (mini PC, Raspberry Pi 5, or a small VPS) running Docker, the
dashboard reachable from your phone over Tailscale, and a phone push from ntfy when orders need you. Keep Robinhood
as a second board you run with Claude when you want it.

## Coinbase setup (about 15 minutes)

1. **A separate portfolio.** In Coinbase (Advanced), create a portfolio, e.g. "Risk Board", and move into it only
   the money the board may trade. The API key you make next can only see and trade this portfolio.
2. **An API key scoped to it.** Coinbase Developer Platform → API keys → create a *Secret API key* for the "Risk
   Board" portfolio with **View** and **Trade** permissions only. Never **Transfer**. Add an IP allowlist if your
   machine has a fixed IP. Download the key JSON.
3. **Put the key file at** `secrets/cdp_api_key.json` (git and Docker ignore that folder) on the always-on machine.
   Or set `COINBASE_API_KEY` and `COINBASE_API_SECRET` in `.env`.
4. **Your limits, in `config.coinbase.json`:** set `safety.max_per_trade_usd` (it ships unset, which refuses every
   order); review the position caps; set `execution.quote_currency` to `USD` or `USDC` (whichever the portfolio
   holds); set `safety.live_armed` to `true`.
5. **Second arming key and approval token**, in `.env` (copy `.env.example`): `CM_LIVE=1`, and `CM_TOKEN=` a long
   random string. With a token set, the dashboard only accepts approvals when opened as `http://host:8765/?t=<token>`.
6. **Phone push (optional, recommended):** install the ntfy app, subscribe to a long random topic, put it in
   `execution.ntfy_topic`, and your Tailscale dashboard address (with `?t=`) in `execution.dashboard_url` so the
   notification opens it. Messages list the proposed orders only, never keys or balances.
7. **Check, then run:**
   ```bash
   pip install -r requirements-live.txt
   python run.py doctor --config config.coinbase.json      # every line OK -> READY (reads balances, places nothing)
   python run.py up --config config.coinbase.json          # or: CONFIG=config.coinbase.json docker compose up -d --build
   ```
8. **Phone access:** install Tailscale on the always-on machine and your phone. Open
   `http://<machine's tailscale name>:8765/?t=<token>`. Don't open port 8765 to the internet: the dashboard has no login beyond the token.

To stop everything at once: the dashboard's Kill switch button, or `python run.py kill`. Only
`python run.py reset-kill` clears it.

## Robinhood setup (crypto and stocks)

The Robinhood board trades **both**: the same 12 coins as the crypto board, plus a stock sleeve of liquid ETFs
(default: SPY, IBIT, GLD, XLE, XLV, GDX, COPX, ITA, URA, VTIP; edit `stocks.symbols` in `config.robinhood.json`).
Stocks run on their own clock and brain, and stock orders only go out while the US market is open:

| | Crypto | Stocks |
|---|---|---|
| Data | Coinbase public candles (Kraken fallback), every minute | Yahoo Finance daily bars and quotes (unofficial, free, no key) |
| Clock | 24/7, hourly desk | 9:30-16:00 ET Mon-Fri, one decision per completed trading day |
| Regime brain | bitcoin | SPY |
| Advisors | crypto desk + crypto exposure board | a separate stock desk + stock exposure board (they learn stocks and coins separately) |
| Share of the account | 60% | 40% (`stocks.share`) |
| Proposals | every 8 h | once each trading morning from 10:00 ET, together with any crypto changes |

Stock orders are plain dollar-amount market orders on liquid ETFs and large caps. Micro-caps and limit-only
names are not supported. If you hold other stocks in the account that are not in `stocks.symbols`, the board counts
their value toward your equity but never trades them.

1. In the Robinhood app, keep **trade approvals on** for the agentic account if you want a second confirmation per order.
2. In `config.robinhood.json`: set `safety.max_per_trade_usd`, `safety.live_armed: true`, and decide `stocks.share`
   (set it to `0` or delete the `stocks` block for a crypto-only board). Set `CM_LIVE=1` in `.env`.
3. `python run.py doctor --config config.robinhood.json` (it also checks Yahoo Finance and tells you if the market is open).
4. Run it: `python run.py up --config config.robinhood.json` (dashboard on port 8767).
5. **Sync** (ask Claude, in Claude Code on the same machine): "sync the Robinhood board". Claude reads the agentic
   account's buying power, crypto holdings and stock positions with the Robinhood connector, and writes them with
   `python run.py rh-snapshot --config config.robinhood.json --cash <buying power> --pos BTC-USD=<qty>,SPY=<shares>,... --other <market value of holdings outside the board's list>`.
   Do this daily, and before you approve anything. Proposals stop when the snapshot is older than 24 h.
6. Approve in the dashboard as usual.
7. **Place** (ask Claude): "place the approved Robinhood board orders". Claude must, for each order in
   `python run.py export-approved --config config.robinhood.json`:
   - stop if `kill_switch` is true or `orders` is empty;
   - for a **stock**: only while the US market is open (9:30-16:00 ET, not a holiday); otherwise leave it approved
     and say so, since it expires after 2 hours. Get a fresh quote (`get_equity_quotes`), skip it if the price moved
     more than `max_move` from `ref_price`, run `review_equity_order`, then place a dollar-amount market order for
     `notional_usd` with `place_equity_order` on the agentic account. If the stock can't be bought by dollar amount
     (not fractional), skip it and ack it as failed with that reason;
   - for a **coin**: get a fresh quote (`get_crypto_quotes`), skip it if the price moved more than `max_move`,
     preview, then `place_crypto_order` for `notional_usd`;
   - Robinhood's own trade approval may ask you again in the app;
   - record the result: `python run.py ack <outbox_id> --config config.robinhood.json --status sent --order-id <id>`
     (or `--status failed --detail "<reason>"`).
   Never place anything that is not in that list.

## Research with Claude (opportunity scout)

The scout (see the README) brings back leads the advisors like. To have Claude research them, in Claude Code on the
machine running the board, say: **"research the board's opportunities"**. Claude should:

1. Run `python run.py scout-export --config <config>` and read the candidates and the research checklist it prints.
2. For each candidate (the top few; skip ones with `already_researched` unless asked): stocks, use the Robinhood
   tools for fundamentals, analyst ratings, the earnings calendar and recent SEC filings, plus a current news
   search; coins, use a news search (listings, unlocks, exploits, regulation) and what the project does. Open the
   pages, say which sources are aggregators, mark any date it could not confirm, check the weekday of each catalyst
   date, and flag AI-valuation exposure.
3. Record each one: `python run.py research SYMBOL --config <config> --verdict favorable|mixed|unfavorable|unverified --note "reasons and what would change it" --source URL --catalyst "date: event"`.
4. Never place an order and never put a name on the watchlist for you. Tell you which leads look best, then you
   decide: the **Watch** button on the dashboard (or `python run.py watch SYMBOL`) puts a name on the board. From
   there it is scanned and scored like every other name, and any trade is still proposed for your approval.

Watched names are adopted by the running scanner on its next tick. Removing one takes effect at the next restart.

## Commands

```
python run.py outbox  --config F          what's pending, approved, sent, skipped
python run.py approve 12 --config F       approve one (or: approve all)
python run.py reject 12 --config F
```

Analysis tool, not financial advice. Crypto can lose most of its value quickly; the board is built defensive, but
every order you approve is your decision and your risk.
