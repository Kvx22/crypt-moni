"""Market data feeds.

LiveFeed   public market data, no API key, tried in order until one answers:
           1. Coinbase Exchange   api.exchange.coinbase.com
           2. Coinbase Advanced   api.coinbase.com/api/v3/brokerage/market (public endpoints)
           3. Kraken              api.kraken.com
SimFeed    an offline synthetic market (regime-switching factor model) whose
           clock moves one hour per tick, for demos and tests. Its data is
           invented: everything it shows is labelled SIMULATION.

Every feed returns only COMPLETED bars (a bar whose period has ended), oldest
first. Nothing here can place an order.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np

from .models import Bar

HOUR, DAY = 3600, 86400
UA = "crypt-moni/1.0 (+https://github.com/kvx22/crypt-moni)"


class FeedError(Exception):
    """Network or parsing failure. The engine backs off and retries; it never trades on it."""


def http_json(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise FeedError(f"{urllib.parse.urlparse(url).netloc}: {e}") from e


def _iso(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s):
    s = s.replace("Z", "+00:00")
    if "." in s:  # trim fractional seconds to 6 digits for fromisoformat
        head, tail = s.split(".", 1)
        frac = "".join(ch for ch in tail if ch.isdigit())
        tz = tail[len(frac):]
        s = f"{head}.{frac[:6]}{tz}"
    return dt.datetime.fromisoformat(s).timestamp()


def completed(bars, gran, now):
    return sorted([b for b in bars if b.ts + gran <= now], key=lambda b: b.ts)


# ---------------------------------------------------------------------------
class CoinbaseExchange:
    name = "coinbase"
    base = "https://api.exchange.coinbase.com"

    @staticmethod
    def parse_candles(rows):
        # [time, low, high, open, close, volume], newest first
        return [Bar(int(r[0]), float(r[3]), float(r[2]), float(r[1]), float(r[4]), float(r[5])) for r in rows]

    def candles(self, symbol, gran, start=None, end=None):
        q = {"granularity": gran}
        if start is not None:
            end = min(end if end is not None else start + 299 * gran, start + 299 * gran, time.time())
            q.update(start=_iso(start), end=_iso(end))
        rows = http_json(f"{self.base}/products/{symbol}/candles?{urllib.parse.urlencode(q)}")
        if not isinstance(rows, list):
            raise FeedError(f"coinbase candles {symbol}: {str(rows)[:120]}")
        return self.parse_candles(rows)

    def ticker(self, symbol):
        d = http_json(f"{self.base}/products/{symbol}/ticker")
        if "price" not in d:
            raise FeedError(f"coinbase ticker {symbol}: {str(d)[:120]}")
        return float(d["price"]), _parse_iso(d["time"]) if d.get("time") else time.time()


class CoinbaseAdvanced:
    name = "coinbase-advanced"
    base = "https://api.coinbase.com/api/v3/brokerage/market"
    GRAN = {60: "ONE_MINUTE", 300: "FIVE_MINUTE", 900: "FIFTEEN_MINUTE", 3600: "ONE_HOUR", 21600: "SIX_HOUR", 86400: "ONE_DAY"}

    @staticmethod
    def parse_candles(d):
        return [Bar(int(c["start"]), float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"]), float(c["volume"]))
                for c in d.get("candles", [])]

    def candles(self, symbol, gran, start=None, end=None):
        end = int(end if end is not None else time.time())
        start = int(start if start is not None else end - 300 * gran)
        start = max(start, end - 349 * gran)  # 350 candles per request
        q = urllib.parse.urlencode({"granularity": self.GRAN[gran], "start": start, "end": end})
        return self.parse_candles(http_json(f"{self.base}/products/{symbol}/candles?{q}"))

    def ticker(self, symbol):
        d = http_json(f"{self.base}/products/{symbol}/ticker?limit=1")
        t = (d.get("trades") or [None])[0]
        if not t:
            raise FeedError(f"coinbase-advanced ticker {symbol}: no trades")
        return float(t["price"]), _parse_iso(t["time"])


class Kraken:
    name = "kraken"
    base = "https://api.kraken.com/0/public"
    ALIAS = {"BTC": "XBT", "DOGE": "XDG"}

    def pair(self, symbol):
        b, q = symbol.split("-")
        return self.ALIAS.get(b, b) + q

    @staticmethod
    def _result(d, what):
        if d.get("error"):
            raise FeedError(f"kraken {what}: {d['error']}")
        res = {k: v for k, v in d.get("result", {}).items() if k != "last"}
        if not res:
            raise FeedError(f"kraken {what}: empty result")
        return next(iter(res.values()))

    @staticmethod
    def parse_ohlc(rows):
        # [time, open, high, low, close, vwap, volume, count], oldest first
        return [Bar(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6])) for r in rows]

    def candles(self, symbol, gran, start=None, end=None):
        q = {"pair": self.pair(symbol), "interval": gran // 60}
        if start is not None:
            q["since"] = int(start) - 1
        bars = self.parse_ohlc(self._result(http_json(f"{self.base}/OHLC?{urllib.parse.urlencode(q)}"), f"OHLC {symbol}"))
        return [b for b in bars if end is None or b.ts <= end]

    def ticker(self, symbol):
        r = self._result(http_json(f"{self.base}/Ticker?pair={self.pair(symbol)}"), f"Ticker {symbol}")
        return float(r["c"][0]), time.time()


ET = ZoneInfo("America/New_York")
OPEN_T, CLOSE_T = dt.time(9, 30), dt.time(16, 0)


def stock_market_open(ts):
    """US regular session, Mon-Fri 9:30-16:00 ET. Holidays are not listed: on one, quotes stop moving,
    go stale, and the safety gate refuses to trade on them."""
    d = dt.datetime.fromtimestamp(ts, ET)
    return d.weekday() < 5 and OPEN_T <= d.time() < CLOSE_T


class Yahoo:
    """Yahoo Finance's public chart endpoint for US stocks and ETFs (unofficial, no key)."""
    name = "yahoo"
    base = "https://query1.finance.yahoo.com/v8/finance/chart"

    @staticmethod
    def parse(j):
        """-> (daily bars with ts = 00:00 UTC of the exchange date, quote price, quote time)."""
        res = (j.get("chart") or {}).get("result")
        if not res:
            raise FeedError(f"yahoo: {((j.get('chart') or {}).get('error') or {}).get('description', 'no result')}")
        r = res[0]
        meta = r.get("meta", {})
        q = ((r.get("indicators") or {}).get("quote") or [{}])[0]
        bars = []
        for i, t in enumerate(r.get("timestamp") or []):
            try:
                o, h, l, c, v = (q[k][i] for k in ("open", "high", "low", "close", "volume"))
            except (KeyError, IndexError):
                continue
            if c is None:
                continue  # a session that has not closed yet can come back null
            day = dt.datetime.fromtimestamp(t, ET).date()
            ts = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp())
            bars.append(Bar(ts, float(o if o is not None else c), float(h if h is not None else c),
                            float(l if l is not None else c), float(c), float(v or 0)))
        px, qt = meta.get("regularMarketPrice"), meta.get("regularMarketTime")
        return bars, (float(px) if px else None), (float(qt) if qt else None)

    def fetch(self, symbol, rng="3mo"):
        j = http_json(f"{self.base}/{urllib.parse.quote(symbol)}?interval=1d&range={rng}&includePrePost=false")
        return self.parse(j)


# ---------------------------------------------------------------------------
SOURCES = {"coinbase": CoinbaseExchange, "coinbase-advanced": CoinbaseAdvanced, "kraken": Kraken}


class LiveFeed:
    kind = "live"
    simulated = False

    def __init__(self, sources=("coinbase", "coinbase-advanced", "kraken"), pause=0.12):
        self.sources = [SOURCES[s]() for s in sources]
        self.pause = pause
        self.last_source = None
        self.errors = {}

    def now(self):
        return time.time()

    def tick(self):
        pass

    def _try(self, fn_name, symbol, *a):
        errs = []
        for src in self.sources:
            try:
                out = getattr(src, fn_name)(symbol, *a)
                self.last_source = src.name
                time.sleep(self.pause)  # stay well under public rate limits
                return out
            except FeedError as e:
                errs.append(str(e))
                self.errors[src.name] = self.errors.get(src.name, 0) + 1
        raise FeedError(f"all sources failed for {symbol}: " + " | ".join(errs))

    def candles(self, symbol, gran, start=None, end=None):
        return completed(self._try("candles", symbol, gran, start, end), gran, self.now())

    def ticker(self, symbol):
        return self._try("ticker", symbol)

    # stock-book API (crypto symbols pass straight through)
    def market_open(self, symbol=None):
        return True


class StockFeed:
    """US stocks/ETFs from Yahoo, cached so a one-minute scan doesn't hammer a free endpoint.
    Completed daily bars only; the quote carries Yahoo's own last-trade time, so a closed
    market (or a holiday) shows up as an old quote that the safety gate refuses."""
    kind = "yahoo"
    simulated = False

    def __init__(self, quote_seconds=90, daily_seconds=1800, pause=0.15):
        self.y, self.qs, self.ds, self.pause = Yahoo(), quote_seconds, daily_seconds, pause
        self._q, self._d = {}, {}
        self.errors = {}

    def now(self):
        return time.time()

    def tick(self):
        pass

    def market_open(self, symbol=None):
        return stock_market_open(self.now())

    def _fetch(self, symbol):
        try:
            bars, px, qt = self.y.fetch(symbol)
        except FeedError:
            self.errors[symbol] = self.errors.get(symbol, 0) + 1
            raise
        time.sleep(self.pause)
        return bars, px, qt

    def daily(self, symbol, now=None):
        """Completed daily bars (up to ~3 months)."""
        now = now or self.now()
        c = self._d.get(symbol)
        if c and now - c[0] < self.ds:
            return c[1]
        bars, px, qt = self._fetch(symbol)
        if px:
            self._q[symbol] = (now, (px, qt or now))
        d = dt.datetime.fromtimestamp(now, ET)
        today = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp())
        after_close = d.time() >= dt.time(16, 15)
        out = [b for b in bars if b.ts < today or (b.ts == today and after_close and d.weekday() < 5)]
        self._d[symbol] = (now, out)
        return out

    def ticker(self, symbol):
        now = self.now()
        c = self._q.get(symbol)
        age = self.qs if stock_market_open(now) else 900
        if c and now - c[0] < age:
            return c[1]
        bars, px, qt = self._fetch(symbol)
        if not px:
            raise FeedError(f"yahoo {symbol}: no price")
        self._q[symbol] = (now, (px, qt or now))
        return px, qt or now


class MixedFeed:
    """Crypto from the exchanges, stocks from Yahoo, behind one interface."""
    kind = "live"
    simulated = False

    def __init__(self, crypto, stocks, stock_symbols):
        self.crypto, self.stocks, self.stock_set = crypto, stocks, set(stock_symbols)

    @property
    def last_source(self):
        return self.crypto.last_source

    def now(self):
        return time.time()

    def tick(self):
        pass

    def is_stock(self, symbol):
        return symbol in self.stock_set

    def market_open(self, symbol=None):
        return self.stocks.market_open() if symbol is None or symbol in self.stock_set else True

    def candles(self, symbol, gran, start=None, end=None):
        return self.crypto.candles(symbol, gran, start, end)

    def ticker(self, symbol):
        return (self.stocks if symbol in self.stock_set else self.crypto).ticker(symbol)

    def daily(self, symbol, now=None):
        return self.stocks.daily(symbol, now)


# ---------------------------------------------------------------------------
class SimFeed:
    """Synthetic 24/7 market for demos and tests (invented data).

    Market factor follows a hidden 5-state regime chain. Each coin = beta x factor
    + idiosyncratic part. The idiosyncratic part carries a planted, slowly
    switching signal: for a stretch of days yesterday's winners keep winning
    (momentum), then the sign flips (mean reversion). That gives the advisors
    something real to learn, and something that changes under them."""

    kind = "sim"
    simulated = True
    REG_DRIFT = np.array([-0.0030, -0.0006, 0.0000, 0.0004, 0.0010])   # hourly factor drift
    REG_VOL = np.array([0.016, 0.008, 0.005, 0.0055, 0.009])          # hourly factor vol
    # starting prices for the invented series (plausible levels, not quotes)
    START = {"ETH-USD": 2460, "SOL-USD": 140, "XRP-USD": 2.2, "DOGE-USD": 0.18, "ADA-USD": 0.62, "AVAX-USD": 22,
             "LINK-USD": 15, "LTC-USD": 82, "BCH-USD": 440, "DOT-USD": 3.6, "AAVE-USD": 210}

    def __init__(self, symbols, start_ts=None, seed=7, history=420, anchors=None, step=HOUR, flip_hours=240):
        self.symbols = list(symbols)
        self.rng = np.random.default_rng(seed)
        self.step = step
        self.flip_hours = flip_hours
        t0 = int(start_ts if start_ts is not None else time.time()) // HOUR * HOUR
        self.t = t0 - history * HOUR      # bar start of the next bar to generate
        self.clock = float(t0) + 1800     # half past: ticks land mid-hour, like a real scan would
        n = len(self.symbols)
        r = self.rng
        self.vscale = np.array([1.0 if "-" in s else 0.28 for s in self.symbols])   # stocks move far less than coins
        self.beta = np.array([1.0 if s.startswith("BTC") else r.uniform(0.9, 1.6) for s in self.symbols])
        self.ivol = np.array([0.0 if s.startswith("BTC") else r.uniform(0.003, 0.007) for s in self.symbols])
        anchors = {**self.START, **(anchors or {})}
        self.price = np.array([float(anchors.get(s, 10 ** r.uniform(-1, 3))) for s in self.symbols])
        self.basevol = np.array([10 ** r.uniform(6.3, 8.3) for _ in self.symbols])  # USD per hour
        self.regime = 2
        self.idio = np.zeros((24, n))
        self.bars = {s: [] for s in self.symbols}
        self.hour_count = 0
        A = np.full((5, 5), 0.002); np.fill_diagonal(A, 0.992)
        self.A = A / A.sum(1, keepdims=True)
        for _ in range(history):
            self._gen()
        # rescale so the last pre-history close equals the anchor (continuity with real history)
        for i, s in enumerate(self.symbols):
            if s in anchors and self.bars[s]:
                k = anchors[s] / self.bars[s][-1].close
                self.bars[s] = [Bar(b.ts, b.open * k, b.high * k, b.low * k, b.close * k, b.volume / k) for b in self.bars[s]]
                self.price[i] *= k

    def phase(self):
        """+1 momentum stretch, -1 reversal stretch (the planted signal's sign)."""
        return 1.0 if (self.hour_count // self.flip_hours) % 2 == 0 else -1.0

    def _gen(self):
        r = self.rng
        self.regime = int(r.choice(5, p=self.A[self.regime]))
        f = self.REG_DRIFT[self.regime] + self.REG_VOL[self.regime] * r.standard_t(5) / math.sqrt(5 / 3)
        signal = self.phase() * 0.15 * self.idio.mean(0)          # planted 24h-return signal
        e = signal + self.ivol * r.standard_normal(len(self.symbols))
        self.idio = np.vstack([self.idio[1:], e])
        ret = (self.beta * f + e) * self.vscale
        for i, s in enumerate(self.symbols):
            o = self.price[i]
            c = o * math.exp(ret[i])
            wig = abs(r.normal(0, 0.003 + abs(ret[i]) / 2))
            hi, lo = max(o, c) * (1 + wig), min(o, c) * (1 - wig)
            z = abs(ret[i]) / max(self.beta[i] * self.REG_VOL[2] + self.ivol[i], 1e-6)
            vol = self.basevol[i] / c * math.exp(r.normal(0, 0.35)) * (1 + 1.5 * z)
            self.bars[s].append(Bar(self.t, o, hi, lo, c, vol))
            self.bars[s] = self.bars[s][-3000:]
            self.price[i] = c
        self.t += HOUR
        self.hour_count += 1

    def now(self):
        return self.clock

    def tick(self):
        """Advance the simulated clock by `step` and generate any bars that completed."""
        self.clock += self.step
        while self.t + HOUR <= self.clock:
            self._gen()

    def candles(self, symbol, gran, start=None, end=None):
        bars = [b for b in self.bars[symbol] if (start is None or b.ts >= start) and (end is None or b.ts <= end)]
        if gran == HOUR:
            return completed(bars, gran, self.clock)[-300:]
        if gran == DAY:
            days = {}
            for b in bars:
                d = b.ts // DAY * DAY
                days.setdefault(d, []).append(b)
            out = [Bar(d, v[0].open, max(x.high for x in v), min(x.low for x in v), v[-1].close, sum(x.volume for x in v))
                   for d, v in sorted(days.items()) if len(v) == 24]
            return completed(out, DAY, self.clock)
        raise FeedError(f"sim feed has no {gran}s bars")

    def market_open(self, symbol=None):
        return True   # the simulator has no closing bell

    def daily(self, symbol, now=None):
        return self.candles(symbol, DAY)

    def ticker(self, symbol):
        last = self.bars[symbol][-1]
        frac = (self.clock - (last.ts + HOUR)) / HOUR
        return last.close * math.exp(self.rng.normal(0, 0.002) * math.sqrt(max(frac, 0.0))), self.clock


def stock_symbols(cfg):
    return list((cfg.get("stocks") or {}).get("symbols", []))


def make_feed(cfg, anchors=None):
    f = cfg.get("feed", {})
    stocks = stock_symbols(cfg)
    if f.get("kind", "live") == "sim":
        return SimFeed(list(cfg["universe"]) + stocks, start_ts=f.get("start_ts"), seed=f.get("seed", 7), anchors=anchors,
                       step=f.get("step_seconds", HOUR), flip_hours=f.get("flip_hours", 240))
    crypto = LiveFeed(tuple(f.get("sources", ("coinbase", "coinbase-advanced", "kraken"))))
    if not stocks:
        return crypto
    return MixedFeed(crypto, StockFeed(f.get("stock_quote_seconds", 90)), stocks)
