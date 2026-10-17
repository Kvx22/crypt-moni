"""Where the opportunity scout looks beyond the board's own list. Edit in config under "scout".
A symbol the exchange or Yahoo doesn't have is skipped, never fatal."""

CRYPTO_CANDIDATES = [
    "ATOM-USD", "NEAR-USD", "APT-USD", "ARB-USD", "OP-USD", "INJ-USD", "SUI-USD", "TIA-USD", "FIL-USD", "ICP-USD",
    "HBAR-USD", "XLM-USD", "ETC-USD", "UNI-USD", "CRV-USD", "LDO-USD", "RENDER-USD", "FET-USD", "GRT-USD", "SHIB-USD",
    "PEPE-USD", "ALGO-USD", "SAND-USD", "MANA-USD", "AXS-USD", "COMP-USD", "SNX-USD", "1INCH-USD", "ENA-USD", "POL-USD",
]

STOCK_CANDIDATES = [
    # broad and sector ETFs
    "QQQ", "IWM", "DIA", "XLK", "XLF", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB", "XLC", "SMH", "SOXX", "ARKK",
    "XBI", "IBB", "KRE", "ITB", "JETS", "TAN", "ICLN", "LIT",
    # bonds, commodities, international, metals
    "TLT", "IEF", "HYG", "LQD", "EEM", "EFA", "VNQ", "SLV", "USO", "UNG", "DBC", "DBA",
    # large caps
    "NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "XOM", "LLY", "UNH", "COST", "WMT", "V",
]

# Names whose story leans on AI valuations. The scout flags them; it never hides them.
AI_EXPOSED = [
    "NVDA", "AVGO", "AMD", "SMH", "SOXX", "MSFT", "GOOGL", "META", "AMZN", "ARKK", "QQQ", "XLK",
    "RENDER-USD", "FET-USD", "TAO-USD",
]
