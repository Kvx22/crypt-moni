"""The extractor's label set. Every note carries one catalyst, one direction and any
number of risk flags. Keep this file small and stable: the classifier, the rules,
the outcome model, the dashboard and the stored labels all key off these names."""

# What kind of event the item is about. "other" = nothing specific (market colour, opinion).
CATALYSTS = (
    "listing",          # new exchange listing / trading launch / added to an index
    "delisting",        # removed from an exchange, trading pairs halted for good
    "partnership",      # integration, adoption, customer or distribution deal
    "product",          # product or feature launch, mainnet/testnet, app release
    "upgrade",          # protocol upgrade, hard fork, network change
    "earnings",         # quarterly results, revenue, EPS beat/miss
    "guidance",         # outlook raised/cut, pre-announcement
    "analyst",          # analyst upgrade/downgrade, price-target change, rating
    "insider",          # insider or large-holder buying/selling (Form 4, whale moves)
    "filing",           # other regulatory filings: 8-K, 10-Q, S-1, 13D/G
    "regulatory",       # regulator action or ruling, approval or rejection, licence
    "legal",            # lawsuit, settlement, court ruling, investigation
    "security",         # exploit, hack, theft, bridge drain, critical vulnerability
    "outage",           # network halt, chain outage, exchange downtime
    "unlock",           # token unlock, dilution, share offering, large supply release
    "funding",          # raise, investment, treasury purchase, grant
    "etf",              # ETF approval, filings, flows
    "macro",            # rates, inflation, Fed, jobs, policy that moves the whole market
    "mna",              # merger, acquisition, takeover, spin-off
    "capital_return",   # buyback, dividend, token burn
    "market_structure", # funding rates, open interest, order-book depth, liquidations
    "sentiment",        # crowd mood, social chatter, fear/greed readings
    "other",
)

DIRECTIONS = (-1, 0, 1)   # bearish, neutral/unclear, bullish for the symbol's price

# Risk flags and how hard each one presses on a score (0..1). A flag is a claim that
# something bad is happening; the aggregator turns credible flags into a CAPPED penalty.
RISK_FLAGS = {
    "exploit": 1.0,       # funds stolen / protocol hacked
    "halt": 1.0,          # trading or the network halted
    "bankruptcy": 1.0,    # insolvency, chapter 11, withdrawals frozen
    "delisting": 0.9,     # exchange removing it
    "fraud": 0.9,         # fraud charges, rug pull, misreporting
    "regulatory": 0.6,    # enforcement action, charges, ban
    "lawsuit": 0.4,       # sued, class action
    "dilution": 0.4,      # big unlock / offering
    "outage": 0.3,        # temporary outage
}

# How long a note stays fresh, by evidence kind (seconds). Risk-flagged notes live longer.
FRESH_S = {
    "news": 48 * 3600, "filing": 7 * 86400, "market": 6 * 3600, "sentiment": 12 * 3600,
    "discovery": 72 * 3600, "search": 48 * 3600, "official": 7 * 86400,
}
FLAG_FRESH_S = 7 * 86400

# Outcome horizons. Crypto counts clock hours; stocks count regular-session hourly bars
# (a session has 7 of them: 9:30 .. 15:30 ET), so "1d" = 1 session and "7d" = 5 sessions.
HORIZONS = ("1h", "4h", "8h", "12h", "1d", "7d")
HORIZON_HOURS_CRYPTO = {"1h": 1, "4h": 4, "8h": 8, "12h": 12, "1d": 24, "7d": 168}
HORIZON_BARS_STOCK = {"1h": 1, "4h": 4, "8h": 8, "12h": 12, "1d": 7, "7d": 35}
BENCH = {"crypto": "BTC-USD", "stock": "SPY"}

# The research features the advisors' research desk learns weights for (engine side).
RESEARCH_SIGNALS = ("r_catalyst", "r_sentiment", "r_fundamental", "r_structure", "r_risk", "r_coverage")
RESEARCH_SIGNAL_NAMES = {
    "r_catalyst": "news and catalysts", "r_sentiment": "crowd sentiment", "r_fundamental": "filings and fundamentals",
    "r_structure": "market structure", "r_risk": "risk flags", "r_coverage": "research coverage",
}

# Which catalysts feed which research feature (the rest feed r_catalyst).
FEATURE_OF = {
    "sentiment": "r_sentiment",
    "earnings": "r_fundamental", "guidance": "r_fundamental", "insider": "r_fundamental", "filing": "r_fundamental",
    "capital_return": "r_fundamental", "funding": "r_fundamental",
    "market_structure": "r_structure", "etf": "r_structure", "macro": "r_structure",
}


def feature_of(catalyst):
    return FEATURE_OF.get(catalyst, "r_catalyst")


def severity(flags):
    return max((RISK_FLAGS.get(f, 0.0) for f in flags or ()), default=0.0)
