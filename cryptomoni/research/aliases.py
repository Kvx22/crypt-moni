"""Which symbols a piece of text is about.

An article says "Bitcoin", "BTC", "$BTC" or "BTC-USD"; a press release says "Nvidia
(NASDAQ: NVDA)". Aliases turns those mentions into the board's symbols, so a general
news feed can be routed to the coins and stocks it actually mentions.

The hard part is not finding names, it is not finding false ones. Many tickers and coin
names are ordinary words ("LINK", "NEAR", "Polygon", "Avalanche", "Doge"). Those are in
AMBIGUOUS and only count when the mention is unmistakable (a $cashtag, "(NYSE: X)",
"X-USD") or when a market word ("token", "staking", "shares", "ETF") sits within a few
words of it. Bare tickers must be written in capitals; names match in any case. When
names overlap the longest one wins, so "Bitcoin Cash" is BCH and never BTC.

discover() does the opposite job: it lists symbols the board does NOT follow yet, for the
discovery team to look into.
"""
from __future__ import annotations

import re

CRYPTO_NAMES = {
    # config.json universe
    "BTC": ["Bitcoin"], "ETH": ["Ethereum", "Ether"], "SOL": ["Solana"], "XRP": ["XRP Ledger", "Ripple"],
    "DOGE": ["Dogecoin", "Doge"], "ADA": ["Cardano"], "AVAX": ["Avalanche"], "LINK": ["Chainlink"],
    "LTC": ["Litecoin"], "BCH": ["Bitcoin Cash"], "DOT": ["Polkadot"], "AAVE": ["Aave"],
    # universe.py CRYPTO_CANDIDATES
    "ATOM": ["Cosmos Hub", "Cosmos"], "NEAR": ["NEAR Protocol"], "APT": ["Aptos"], "ARB": ["Arbitrum"],
    "OP": ["Optimism"], "INJ": ["Injective"], "SUI": ["Sui Network", "Sui"], "TIA": ["Celestia"],
    "FIL": ["Filecoin"], "ICP": ["Internet Computer"], "HBAR": ["Hedera Hashgraph", "Hedera"],
    "XLM": ["Stellar Lumens", "Stellar"], "ETC": ["Ethereum Classic"], "UNI": ["Uniswap"], "CRV": ["Curve DAO", "Curve Finance"],
    "LDO": ["Lido DAO", "Lido"], "RENDER": ["Render Network", "Render"], "FET": ["Fetch.ai", "Artificial Superintelligence Alliance"],
    "GRT": ["The Graph"], "SHIB": ["Shiba Inu"], "PEPE": ["Pepe"], "ALGO": ["Algorand"], "SAND": ["The Sandbox"],
    "MANA": ["Decentraland"], "AXS": ["Axie Infinity"], "COMP": ["Compound"], "SNX": ["Synthetix"],
    "1INCH": ["1inch"], "ENA": ["Ethena"], "POL": ["Polygon"],
    # other well-known coins (discovery fodder)
    "TAO": ["Bittensor"], "TON": ["Toncoin"], "TRX": ["Tron"], "BNB": ["BNB Chain", "Binance Coin"],
    "WIF": ["dogwifhat"], "BONK": ["Bonk"], "SEI": ["Sei Network", "Sei"], "JUP": ["Jupiter"], "PYTH": ["Pyth Network", "Pyth"],
    "ONDO": ["Ondo Finance", "Ondo"], "XMR": ["Monero"], "STX": ["Stacks"], "IMX": ["Immutable"], "RUNE": ["THORChain"],
    "MKR": ["MakerDAO"], "KAS": ["Kaspa"], "FLOKI": ["Floki"], "WLD": ["Worldcoin"], "JTO": ["Jito"],
    "STRK": ["Starknet"], "HYPE": ["Hyperliquid"], "ENS": ["Ethereum Name Service"], "GALA": ["Gala Games"],
    "EGLD": ["MultiversX"], "QNT": ["Quant Network"], "XTZ": ["Tezos"], "EOS": ["EOS Network"],
}

STOCK_NAMES = {
    # config.robinhood.json stocks.symbols
    "SPY": ["S&P 500 ETF", "SPDR S&P 500"], "IBIT": ["iShares Bitcoin Trust"], "GLD": ["SPDR Gold Shares", "SPDR Gold Trust"],
    "XLE": ["Energy Select Sector SPDR"], "XLV": ["Health Care Select Sector SPDR"], "GDX": ["VanEck Gold Miners ETF", "Gold Miners ETF"],
    "COPX": ["Global X Copper Miners ETF", "Copper Miners ETF"], "ITA": ["iShares U.S. Aerospace & Defense ETF"],
    "URA": ["Global X Uranium ETF", "Uranium ETF"], "VTIP": ["Vanguard Short-Term Inflation-Protected Securities ETF"],
    # universe.py STOCK_CANDIDATES: ETFs
    "QQQ": ["Invesco QQQ", "Nasdaq-100 ETF"], "IWM": ["iShares Russell 2000 ETF", "Russell 2000 ETF"],
    "DIA": ["SPDR Dow Jones Industrial Average ETF", "Dow Jones ETF"], "XLK": ["Technology Select Sector SPDR"],
    "XLF": ["Financial Select Sector SPDR"], "XLI": ["Industrial Select Sector SPDR"], "XLY": ["Consumer Discretionary Select Sector SPDR"],
    "XLP": ["Consumer Staples Select Sector SPDR"], "XLU": ["Utilities Select Sector SPDR"], "XLRE": ["Real Estate Select Sector SPDR"],
    "XLB": ["Materials Select Sector SPDR"], "XLC": ["Communication Services Select Sector SPDR"],
    "SMH": ["VanEck Semiconductor ETF"], "SOXX": ["iShares Semiconductor ETF"], "ARKK": ["ARK Innovation ETF"],
    "XBI": ["SPDR S&P Biotech ETF"], "IBB": ["iShares Biotechnology ETF"], "KRE": ["SPDR S&P Regional Banking ETF", "regional bank ETF"],
    "ITB": ["iShares U.S. Home Construction ETF"], "JETS": ["U.S. Global Jets ETF"], "TAN": ["Invesco Solar ETF"],
    "ICLN": ["iShares Global Clean Energy ETF"], "LIT": ["Global X Lithium & Battery Tech ETF"],
    "TLT": ["iShares 20+ Year Treasury Bond ETF"], "IEF": ["iShares 7-10 Year Treasury Bond ETF"],
    "HYG": ["iShares iBoxx High Yield Corporate Bond ETF", "high-yield bond ETF"], "LQD": ["iShares iBoxx Investment Grade Corporate Bond ETF"],
    "EEM": ["iShares MSCI Emerging Markets ETF"], "EFA": ["iShares MSCI EAFE ETF"], "VNQ": ["Vanguard Real Estate ETF"],
    "SLV": ["iShares Silver Trust"], "USO": ["United States Oil Fund"], "UNG": ["United States Natural Gas Fund"],
    "DBC": ["Invesco DB Commodity Index Tracking Fund"], "DBA": ["Invesco DB Agriculture Fund"],
    # universe.py STOCK_CANDIDATES: large caps
    "NVDA": ["Nvidia"], "MSFT": ["Microsoft"], "AAPL": ["Apple"], "AMZN": ["Amazon"], "GOOGL": ["Alphabet", "Google"],
    "META": ["Meta Platforms", "Meta"], "TSLA": ["Tesla"], "AVGO": ["Broadcom"], "JPM": ["JPMorgan Chase", "JPMorgan"],
    "XOM": ["Exxon Mobil", "ExxonMobil", "Exxon"], "LLY": ["Eli Lilly"], "UNH": ["UnitedHealth Group", "UnitedHealth"],
    "COST": ["Costco"], "WMT": ["Walmart"], "V": ["Visa Inc"],
}

# Tickers and names that are also ordinary words or other things. They count only as a
# cashtag, an exchange-qualified ticker, a pair, or next to a market word.
AMBIGUOUS = {
    # tickers (matched exactly, capitals)
    "OP", "ARB", "UNI", "SAND", "MANA", "COMP", "GRT", "LINK", "DOT", "NEAR", "ONE", "GAS", "ATOM", "APT",
    "ALGO", "ETC", "ADA", "DOGE", "TIA", "POL", "PEPE", "RENDER", "HYPE", "TAO", "TON", "SEI", "JUP", "BONK",
    "IMX", "RUNE", "ENS", "GALA", "EOS", "STX",
    "V", "ITA", "DIA", "LIT", "TAN", "JETS", "GDX", "COST",
    # names (matched case-insensitively)
    "avalanche", "polygon", "chainlink", "ripple", "ether", "doge", "cosmos", "optimism", "sui", "stellar",
    "lido", "render", "pepe", "compound", "tron", "bonk", "sei", "jupiter", "stacks", "immutable", "floki",
    "internet computer", "the graph", "the sandbox", "apple", "amazon", "meta", "google",
}
_AMBIG_NAMES = {a.lower() for a in AMBIGUOUS if not a.isupper()}
_AMBIG_TICKERS = {a for a in AMBIGUOUS if a.isupper()}

CRYPTO_CONTEXT = {"token", "tokens", "coin", "coins", "crypto", "cryptocurrency", "cryptocurrencies", "blockchain",
                  "defi", "price", "network", "staking", "altcoin", "altcoins", "protocol", "mainnet", "testnet",
                  "airdrop", "wallet", "onchain", "on-chain", "layer-2", "l2", "memecoin", "dex", "tvl", "validators"}
STOCK_CONTEXT = {"shares", "share", "stock", "stocks", "etf", "fund", "inc", "corp", "earnings", "nasdaq", "nyse",
                 "price", "investors", "ticker", "dividend", "revenue", "analyst", "analysts", "s&p"}
CONTEXT_WORDS = 6
MAX_TEXT = 200_000   # characters scanned per call; items are ~4000, this only stops a runaway page

# never symbols: fiat, stablecoins, units and finance shorthand that shows up after a "$"
JUNK = {"USD", "USDT", "USDC", "USDE", "DAI", "FDUSD", "PYUSD", "TUSD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF",
        "CNY", "HKD", "K", "M", "MM", "B", "BN", "T", "TN", "MLN", "BLN", "TRN", "US", "CEO", "IPO", "AI", "ETF",
        "SEC", "FED", "GDP", "CPI", "API", "NFT", "DAO", "TVL", "ATH", "OTC", "AND", "THE", "FOR"}

_HOW_RANK = {"cashtag": 0, "exchange": 0, "pair": 0, "name": 1, "ticker": 2}
_WORD = re.compile(r"[\w$&.+-]+")
_CASHTAG = re.compile(r"(?<![\w$])\$([A-Za-z0-9][A-Za-z0-9]{0,9})(?![\w])")
_EXCHANGE = re.compile(r"(?<![\w])(?i:nasdaq|nyse\s?arca|nyse\s?american|nyse|amex|cboe|bats)\s*:\s*"
                       r"([A-Z]{1,5}(?:\.[A-Z])?)(?![\w])")
_PAIR = re.compile(r"(?<![\w$])([A-Z0-9]{2,10})\s?[-/]\s?(?:USDT|USDC|USD)(?![\w])")


def _base(symbol):
    return symbol.split("-")[0].upper()


def _guess_asset(symbol):
    if "-" in symbol or "/" in symbol:
        return "crypto"
    return "stock" if symbol.upper() in STOCK_NAMES else None


def _near(text, start, end, words):
    """True when one of `words` is within CONTEXT_WORDS words of text[start:end]."""
    before = _WORD.findall(text[max(0, start - 120):start])[-CONTEXT_WORDS:]
    after = _WORD.findall(text[end:end + 120])[:CONTEXT_WORDS]
    return any(w.lower().strip(".-") in words for w in before + after)


def _alternation(strings):
    alts = sorted({s for s in strings if s}, key=len, reverse=True)
    return "|".join(re.escape(s) for s in alts) if alts else r"(?!x)x"


class Aliases:
    """Mentions of the subscribed symbols in text.

    symbols: {symbol: asset}, e.g. {"BTC-USD": "crypto", "NVDA": "stock"}
    extra_names: {symbol: [names]} added to the built-in tables (e.g. from config)
    """

    def __init__(self, symbols: dict, extra_names: dict | None = None):
        self.symbols = {s: a for s, a in (symbols or {}).items() if isinstance(s, str) and s}
        self._extra = {}
        for s, names in (extra_names or {}).items():
            if isinstance(s, str) and isinstance(names, (list, tuple)):
                self._extra[s] = [n.strip() for n in names if isinstance(n, str) and n.strip()]
        # every symbol we know of, subscribed or not, so overlaps resolve the same way
        # whatever is subscribed ("Bitcoin Cash" stays BCH even when only BTC is followed)
        everything = {f"{t}-USD": "crypto" for t in CRYPTO_NAMES}
        everything.update({t: "stock" for t in STOCK_NAMES})
        everything.update(self.symbols)
        self._name_to = {}       # lowercase name -> [(symbol, asset)]
        self._ticker_to = {}     # TICKER -> [(symbol, asset)]
        for sym, asset in everything.items():
            asset = asset or _guess_asset(sym)
            for n in self._table_names(sym):
                self._name_to.setdefault(n.lower(), []).append((sym, asset))
            self._ticker_to.setdefault(_base(sym) if asset == "crypto" else sym.upper(), []).append((sym, asset))
        self._names_re = re.compile(r"(?<![\w$])(?:" + _alternation(self._name_to) + r")(?![\w])", re.I)
        self._tickers_re = re.compile(r"(?<![\w$])(?:" + _alternation(self._ticker_to) + r")(?![\w])")

    def _table_names(self, symbol):
        b = _base(symbol)
        table = CRYPTO_NAMES.get(b, []) if "-" in symbol else STOCK_NAMES.get(symbol, [])
        return list(dict.fromkeys(table + self._extra.get(symbol, [])))

    def _ok_context(self, text, start, end, asset):
        return _near(text, start, end, CRYPTO_CONTEXT if asset == "crypto" else STOCK_CONTEXT)

    def _scan(self, text):
        """All mentions as (start, end, symbol, asset, how), overlaps resolved, unfiltered."""
        text = text[:MAX_TEXT]
        found = []
        for m in _CASHTAG.finditer(text):
            for sym, asset in self._ticker_to.get(m.group(1).upper(), ()):
                found.append((m.start(), m.end(), sym, asset, "cashtag"))
        for m in _EXCHANGE.finditer(text):
            for sym, asset in self._ticker_to.get(m.group(1), ()):
                if asset == "stock":
                    found.append((m.start(), m.end(), sym, asset, "exchange"))
        for m in _PAIR.finditer(text):
            for sym, asset in self._ticker_to.get(m.group(1), ()):
                if asset == "crypto":
                    found.append((m.start(), m.end(), sym, asset, "pair"))
        for m in self._names_re.finditer(text):
            name = m.group(0).lower()
            for sym, asset in self._name_to.get(name, ()):
                if name not in _AMBIG_NAMES or self._ok_context(text, m.start(), m.end(), asset):
                    found.append((m.start(), m.end(), sym, asset, "name"))
        for m in self._tickers_re.finditer(text):
            t = m.group(0)
            for sym, asset in self._ticker_to.get(t, ()):
                if (t not in _AMBIG_TICKERS and len(t) > 2) or self._ok_context(text, m.start(), m.end(), asset):
                    found.append((m.start(), m.end(), sym, asset, "ticker"))
        # strongest form first, then the longest span; a span already taken blocks the rest
        found.sort(key=lambda f: (_HOW_RANK[f[4]], f[0] - f[1], f[0]))
        used, spans, out = bytearray(len(text)), set(), []
        for f in found:
            span = (f[0], f[1], f[4])
            # the very same mention may name two symbols (a ticker that is both a coin and a
            # stock); any other overlap means a stronger reading already claimed the text
            if span not in spans and any(used[f[0]:f[1]]):
                continue
            spans.add(span)
            used[f[0]:f[1]] = b"\x01" * (f[1] - f[0])
            out.append(f)
        return sorted(out)

    def find(self, text) -> list[tuple[str, str]]:
        """[(symbol, how)] for the subscribed symbols mentioned in text, in order of first
        mention; `how` is the strongest form seen: cashtag|exchange|pair|name|ticker."""
        if not isinstance(text, str) or not text:
            return []
        best, order = {}, []
        for _, _, sym, _, how in self._scan(text):
            if sym not in self.symbols:
                continue
            if sym not in best:
                order.append(sym)
                best[sym] = how
            elif _HOW_RANK[how] < _HOW_RANK[best[sym]]:
                best[sym] = how
        return [(s, best[s]) for s in order]

    def names(self, symbol) -> list[str]:
        """Search terms for a symbol, best first: ["Solana", "SOL"], ["Nvidia", "NVDA"]."""
        if not isinstance(symbol, str) or not symbol:
            return []
        ticker = _base(symbol) if (self.asset(symbol) == "crypto") else symbol.upper()
        return list(dict.fromkeys(self._table_names(symbol) + [ticker]))

    def asset(self, symbol) -> str | None:
        """"crypto" | "stock" for a subscribed symbol, else a best guess from the tables (or None)."""
        if symbol in self.symbols:
            return self.symbols[symbol]
        return _guess_asset(symbol) if isinstance(symbol, str) and symbol else None


_ALL = None


def _all_aliases():
    global _ALL
    if _ALL is None:
        _ALL = Aliases({})
    return _ALL


def _crypto_context(text):
    return any(w.lower() in CRYPTO_CONTEXT for w in _WORD.findall(text))


def discover(text, known: set) -> list[tuple[str, str, str]]:
    """Symbols mentioned in text that are NOT in `known`: [(symbol, asset, how)].

    $XYZ -> "XYZ-USD" when XYZ is a known coin (or the text is about crypto and XYZ is not a
    known stock), "XYZ" when it is a known stock; "(NASDAQ: XYZ)" / "NYSE:XYZ" -> stock "XYZ";
    a full coin name ("Bittensor") -> its pair. Fiat, stablecoins and amounts are never returned."""
    if not isinstance(text, str) or not text:
        return []
    known = set(known or ())
    hits = []
    crypto_ctx = None
    for m in _CASHTAG.finditer(text):
        t = m.group(1).upper()
        if t in JUNK or not (t[0].isalpha() or t in CRYPTO_NAMES):
            continue
        if t in CRYPTO_NAMES:
            hits.append((m.start(), f"{t}-USD", "crypto", "cashtag"))
        elif t in STOCK_NAMES:
            hits.append((m.start(), t, "stock", "cashtag"))
        elif t.isalpha() and 2 <= len(t) <= 6:
            if crypto_ctx is None:
                crypto_ctx = _crypto_context(text)
            if crypto_ctx:
                hits.append((m.start(), f"{t}-USD", "crypto", "cashtag"))
    for m in _EXCHANGE.finditer(text):
        if m.group(1) not in JUNK:
            hits.append((m.start(), m.group(1), "stock", "exchange"))
    for start, _, sym, asset, how in _all_aliases()._scan(text):
        if how == "name" and asset == "crypto":
            hits.append((start, sym, asset, how))
    out, seen = [], set()
    for _, sym, asset, how in sorted(hits):
        if sym not in known and sym not in seen:
            seen.add(sym)
            out.append((sym, asset, how))
    return out
