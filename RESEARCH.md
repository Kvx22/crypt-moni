# The research service

The boards now come with a 24/7 research service. It reads news, filings, exchange status pages,
market structure and crowd chatter, turns what it finds into labelled notes, and gives the five
advisors a research view of every name they score. **Nothing in it can place an order**: research can
move a score (within limits), add a warning to a proposal, or suggest a name for the watch-only list.
Every order still waits for your approval, exactly as before.

It needs no API keys and no Claude session: it runs on your PC with the board.

## Start it

Nothing to install: `python run.py up --config config.robinhood.json` (or the Coinbase config) starts the
research service as a supervised background process. Start both boards and they share one service (the
second one stands by and takes over if the first stops). Its log is `data/research-<board>.log`.

Two things to do once, on the PC that runs the boards:

1. `python run.py research-probe --config config.robinhood.json` fetches every source once and prints what
   answered. Some sites block some networks or regions; failed primaries are covered by their standbys.
2. SEC asks every program that reads EDGAR to say who it is. Add a contact to your config so the filings
   workers are not refused:

   ```json
   "research": { "contact": "you@example.com" }
   ```

   (Put your own address there. It is sent only to sec.gov and the other sources, inside the User-Agent.)

## What runs

| Team | What it watches | Primaries | Standbys |
|---|---|---|---|
| catalysts | crypto news feeds, Google/Bing News per name, Yahoo/Nasdaq stock headlines | 4 | 4 |
| risk | exchange trading status, exchange incident pages, risk news searches, SEC press/litigation | 4 | 4 |
| filings | SEC filings (8-K items), insider Form 4s, earnings calendar, DeFi TVL | 4 | 4 |
| market | perpetual funding rates, order-book imbalance, VIX | 3 | 3 |
| sentiment | StockTwits, Reddit, Hacker News, fear & greed | 3 | 3 |
| discovery | new exchange listings, trending coins, listing news | 2 | 2 |

That is 20 primary workers and 20 standbys. A standby takes over when its primary fails three times in a row
or goes quiet, and hands back when the primary recovers. When a team's coverage drops (under 60% of its names
got a fresh note in 24 hours) the service grows: it switches on a reserve source or splits a per-name worker
into more shards, up to 60 workers. Grown workers that stop pulling their weight are retired after 48 quiet
hours.

**Forever search** runs beside them: a separate process with no keys in its environment and its own small
database. It searches continuously (Google News, Bing News, and SearXNG if you run one: set
`research.forever.searxng_url`), reads pages without a browser, and refuses private addresses (it can never
reach your dashboard or your LAN). Everything it finds is *quarantined*: shown as "unverified" and never
counted unless an ordinary source agrees.

## From item to advice

1. **Labelling.** Each item gets a catalyst (listing, earnings, security, regulatory, insider, unlock, …), a
   direction (bullish, neutral, bearish) and any risk flags (exploit, halt, delisting, bankruptcy, fraud,
   regulatory, lawsuit, dilution, outage). Rules do this first; the **extractor** (a small text model trained
   on your PC, starting from 300+ hand-written examples) takes over where it is confident. Click **fix** on a
   note in the dashboard to correct it: your corrections count three times as much in training.
2. **Corroboration.** Syndicated copies of one story count once. A positive boost needs **four independent
   sources over at least two kinds of evidence** (news plus filings, say). Thinner good news is still heard,
   at a quarter of its weight. Bad news is never discounted.
3. **Risk flags give a capped penalty, not a veto.** A credible flag (official source, a reliable source, or
   two independent ones) lowers the desk score by at most 0.35 (the BUY bar is 0.50), and the penalty fades
   with a one-day half-life. The only hard stop is the exchange itself reporting trading halted: then the
   board proposes nothing on that name until it resumes.
4. **The advisors weigh research their own way**, with weights that learn from outcomes like their market
   weights do: Kai leans on catalysts and momentum in the news, Vera fades crowd hype, Imani weighs risk flags
   hardest, Marguerite reads filings and fundamentals, Idris watches market structure.
5. **Thin evidence triggers more research.** When an advisor likes a name that fewer than two independent
   sources cover, the board files a request; the workers look at requested names first.

## Outcomes, learning, and the switches

Every note is stored with the price at the time, then scored at **1h, 4h, 8h, 12h, 1d and 7d** against
bitcoin (coins) or SPY (stocks). Stocks count market hours: 1d is one session, 7d is five. From that:

* each source earns a reliability score (sources that keep being wrong lose weight and are polled less);
* the **outcome model** learns which kinds of notes preceded real moves;
* the **extractor** and the outcome model retrain daily on your PC, and a new version replaces the current one
  only if it does better on data it never saw (champion vs challenger); the old one is kept.

Every research piece starts in **shadow**: computed, logged, shown, but not applied. It switches on by
itself when its evidence gate passes, **one at a time, in this order**, at least three days apart:

| Stage | Switches on when | What changes |
|---|---|---|
| penalties | after 7+ days, 30+ penalised names did worse than their benchmark (55%+ of the time) | capped penalties and halts apply |
| signals | after 7+ days and 300+ graded scores, research-adjusted scores ranked outcomes better | research features enter each advisor's score |
| watchlist | 10+ advisor watch-only picks beat their benchmark over 7 days | advisors add names to the watch-only list (1 each per day) |
| cadence | the 2-hour paper book beat today's cadence by 0.5%+ after fees over 14+ days | the board proposes every 2 hours (conviction bar, 6h cooldown, 6 a day) |

A stage that goes bad is switched back to shadow by itself. You can override any of it:
`python run.py research-stage` lists them, `research-stage signals on|off|shadow|auto` switches one ("off"
sticks until you change it). The dashboard's Research card has the same buttons.

## The advisors start trained

With a real feed, on a fresh start and weekly after that, the board replays up to 90 days of hourly history so the pick desk
starts from what worked rather than from instinct. The replayed desk is used only if it beat the instinct-only
desk (and, once the live desk has learned, the live desk too) on the most recent fifth of that history. (The simulator only pretrains when its config sets `"pretrain": {"enabled": true}`.)

## The 2-hour cadence, in paper first

Two paper books run beside every board, both following the same advisor targets: one rebalances the way the
board does today, the other every 2 hours with a conviction bar, a per-name cooldown and a daily cap. The
Research card compares them after fees. The cadence only reaches your real proposals through its stage gate.

## Commands

| Command | |
|---|---|
| `research-status` | workers, teams, stages, models and the latest notes |
| `research-probe [--team T]` | fetch every source once and report what answered |
| `research-stage [NAME on\|off\|shadow\|auto]` | see or switch the research pieces |
| `research-label ID --catalyst C --direction -1\|0\|1 [--flags a,b]` | correct a note |
| `research-train` | train the extractor and outcome model now |
| `pretrain` | replay history for the advisors now (stop the board first) |

`up --no-research` runs a board without the service.

## Config (all optional)

```json
"research": {
  "contact": "you@example.com",
  "enabled": true,
  "max_workers": 60,
  "weight": 0.35, "penalty_cap": 0.35, "boost_min_sources": 4, "boost_min_kinds": 2,
  "disable": ["reddit_crypto"],
  "gates": { "signals": { "min_days": 14 } },
  "forever": { "enabled": true, "searxng_url": null, "per_hour": 240 },
  "thesis_llm": null
},
"cadence": { "every_hours": 2, "min_conviction": 0.35, "cooldown_hours": 6, "max_trades_per_day": 6 },
"pretrain": { "enabled": true, "hours": 2160, "every_hours": 168 }
```

`thesis_llm` (off by default) lets a local model (for example Ollama at `http://127.0.0.1:11434/api/generate`)
write the short case shown with each proposal. It only ever sees the structured facts, its text is marked as an
AI draft, and it never affects a decision. Without it, the case is written from a template.

## Limits worth knowing

* Research improves the odds; it cannot guarantee anything. Small trades produce few outcomes, so the gates
  take days to weeks to open.
* Free sources change, rate-limit or block. The health panel and phone alerts tell you when a team is
  degraded; `research-probe` tells you which source.
* Forever search reads the open web. Its findings are quarantined for that reason, and its process holds no
  keys, but it is process-level isolation on your PC, not a separate machine.
