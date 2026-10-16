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

## Robinhood setup

1. In the Robinhood app, keep **trade approvals on** for the agentic account if you want a second confirmation per order.
2. In `config.robinhood.json`: set `safety.max_per_trade_usd`, `safety.live_armed: true`; set `CM_LIVE=1`.
3. Run it beside (or instead of) Coinbase: `python run.py up --config config.robinhood.json` (dashboard on port 8767).
4. **Sync** (ask Claude, in Claude Code on the same machine): "sync the Robinhood crypto board". Claude reads the
   agentic account's crypto holdings and buying power with the Robinhood connector and writes them with
   `python run.py rh-snapshot --config config.robinhood.json --cash <buying power> --pos BTC-USD=<qty>,ETH-USD=<qty>`.
5. Approve in the dashboard as usual.
6. **Place** (ask Claude): "place the approved crypto board orders". Claude must, for each order in
   `python run.py export-approved --config config.robinhood.json`:
   - stop if `kill_switch` is true or `orders` is empty;
   - get a fresh quote (`get_crypto_quotes`) and skip the order if the price moved more than `max_move` from `ref_price`;
   - preview, then place it with `place_crypto_order` for `notional_usd` on the agentic account (Robinhood's own
     trade approval may ask you again in the app);
   - record the result: `python run.py ack <outbox_id> --config config.robinhood.json --status sent --order-id <id>`
     (or `--status failed --detail "<reason>"`).
   Never place anything that is not in that list.

## Commands

```
python run.py outbox  --config F          what's pending, approved, sent, skipped
python run.py approve 12 --config F       approve one (or: approve all)
python run.py reject 12 --config F
```

Analysis tool, not financial advice. Crypto can lose most of its value quickly; the board is built defensive, but
every order you approve is your decision and your risk.
