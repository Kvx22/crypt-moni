"""The label set and the rule labeller.

Every note the research service writes carries three labels:
  catalyst    what kind of event the story is about (taxonomy.CATALYSTS, "other" when none)
  direction   -1 / 0 / +1: is the event bad, unclear or good for the symbol's price
  risk_flags  specific bad things that are happening (taxonomy.RISK_FLAGS)

rule_label() is the first labeller: hand-written keyword rules, title weighted over body.
It is deliberately cautious with risk flags, because a flag turns into a score penalty: a
flag needs a strong, specific phrase ("drained", "$40m stolen", "files for chapter 11"),
not just a scary word, and it is dropped when the phrase is negated ("denies", "no funds
were lost"), hypothetical ("could", "how to protect against") or already resolved
("restored after"). The extractor model (classifier.py) learns from SEED, your corrections
and the confident rule labels, and takes over once it beats what came before.

Direction comes from cue words with negation handling: "approved" is good, "not approved"
or "fails to win approval" is bad; "hack" is bad, "denies hack" cancels out. Price moves on
their own ("jumps 12%") are not cues: the label describes the event, not the tape.
"""
from __future__ import annotations

import math
import re

from .seed_labels import SEED  # noqa: F401  (re-exported: labels is the label-set module)
from .taxonomy import CATALYSTS, DIRECTIONS, RISK_FLAGS  # noqa: F401

TEXT_MAX = 4000     # characters of body the rules read
FLAG_TEXT_MAX = 1200  # flags only from the top of the body: a story's lede, not its background
CONF_MIN, CONF_MAX = 0.3, 0.95
WHY_MAX = 90

LABEL_HELP = {
    "listing": "A new exchange listing, trading launch or index inclusion.",
    "delisting": "Removed from an exchange, trading pairs ended for good, or a delisting notice.",
    "partnership": "An integration, adoption, customer or distribution deal with someone else.",
    "product": "A product or feature launch, app release, mainnet or testnet going live.",
    "upgrade": "A protocol upgrade, hard fork or other change to how the network works.",
    "earnings": "Quarterly or annual results: revenue, profit, earnings per share, beat or miss.",
    "guidance": "The company raises, cuts or confirms its outlook, or pre-announces results.",
    "analyst": "An analyst upgrade or downgrade, a price-target change or new coverage.",
    "insider": "Insiders or large holders buying or selling (Form 4 filings, whale wallets).",
    "filing": "Other regulatory paperwork: 8-K, 10-Q, S-1, 13D/13G and similar filings.",
    "regulatory": "A regulator acts or rules: approval, rejection, licence, ban, charges, fine.",
    "legal": "A lawsuit, settlement, court ruling, criminal case, investigation or bankruptcy court.",
    "security": "An exploit, hack, theft, bridge drain or critical vulnerability.",
    "outage": "A network halt, chain outage, exchange downtime, trading halt or frozen withdrawals.",
    "unlock": "New supply: a token unlock, share offering or other dilution.",
    "funding": "Money coming in: a raise, investment, treasury purchase or grant.",
    "etf": "Exchange-traded fund news: filings, approvals, rejections, launches, inflows and outflows.",
    "macro": "Rates, inflation, jobs, central banks, tariffs: news that moves the whole market.",
    "mna": "A merger, acquisition, takeover bid or spin-off.",
    "capital_return": "Money going back to holders: buybacks, dividends, token burns.",
    "market_structure": "Funding rates, open interest, liquidations, short interest, order-book depth.",
    "sentiment": "Crowd mood: social chatter, fear and greed readings, retail frenzy or gloom.",
    "other": "Nothing specific: price recaps, explainers, opinion and general market colour.",
}

# ---------------------------------------------------------------------------- catalyst rules
# (catalyst, weight, pattern). A title match counts double. Weights: 2 = specific phrase,
# 1 = a usual word for it, 0.5 = a hint that needs company.
_N = r"\$?\d[\d,.]*\s*(?:[kmb]n?|million|billion|thousand)?"   # "$40m", "1.2 billion", "300,000"
_CAT_RULES = [
    ("listing", 2, r"\b(?:lists?|will list|to list|listed)\b(?! (?:as|among|by|of|for|below|above|its)\b)|\blisting (?:on|of)\b"),
    ("listing", 2, r"\b(?:added|joins?|join|to join|inclusion) (?:to |in |into )?(?:the )?(?:\w+ ){0,3}(?:index|indices)\b"),
    ("listing", 1.5, r"\b(?:debut(?:s|ed)?|ipo|goes public|direct listing|begins trading|trading (?:opens|launches|begins)|new trading pairs?|spot trading)\b"),
    ("listing", 1, r"\b(?:listing|launchpool|launchpad)\b"),
    ("delisting", 2, r"\bde-?list(?:s|ed|ing|ment)?\b|\bremov(?:es|ed|ing|al) (?:of )?(?:\w+ ){0,3}(?:trading pairs?|from (?:the )?(?:\w+ ){0,2}(?:index|exchange|platform))"),
    ("delisting", 1.5, r"\b(?:non-?compliance|deficiency) notice\b|\bminimum bid (?:price )?(?:rule|requirement)\b|\bdelisting notice\b|\b(?:ends|ending|to end) (?:spot |margin )?(?:trading|support) for\b"),
    ("partnership", 2, r"\bpartner(?:s|ed|ing)? with\b|\b(?:partnership|alliance|collaboration|teams up|team up|joint venture|integrat(?:es|ed|ion) with)\b"),
    ("partnership", 1.5, r"\b(?:signs?|signed|inks?|inked|wins?|won|secures?|secured|lands?|landed) (?:a |an |its )?(?:\w+ ){0,3}(?:deal|contract|agreement|order)\b"),
    ("partnership", 1, r"\b(?:adopts?|adopted|adoption|selects?|selected|chooses|customer|supplier|distribution deal|integrat(?:es|ed|ion))\b"),
    ("product", 2, r"\b(?:launch(?:es|ed|ing)?|unveil(?:s|ed)?|introduc(?:es|ed)|rolls? out|rolled out|releases?|released|debuts?) (?:a |an |its |the |new )?(?:\w+ ){0,3}(?:app|wallet|platform|product|feature|service|chip|device|model|drug|tool|card|version|lineup|exchange|marketplace)\b"),
    ("product", 2, r"\b(?:mainnet|testnet|devnet)\b(?! upgrade)|\b(?:product launch|goes live|went live|now live|general availability|beta)\b"),
    ("product", 1, r"\b(?:launch(?:es|ed)?|unveil(?:s|ed)?|new feature|recall(?:s|ed)?|fda (?:approval|clears|approves)|clearance)\b"),
    ("upgrade", 2, r"\b(?:hard|soft) fork\b|\b(?:network|protocol|mainnet|chain|consensus|node|client) upgrade\b|\bupgrade (?:goes live|activat\w*|is live|went live|scheduled|delayed|postponed)\b"),
    ("upgrade", 1.5, r"\b(?:improvement proposal|[a-z]{2,4}ip-?\d+|activation|activates|activated|fork|migration to|v\d(?:\.\d)? upgrade)\b"),
    ("upgrade", 0.5, r"\bupgrade\b"),
    ("earnings", 2, r"\b(?:q[1-4]|first|second|third|fourth|quarterly|annual|full-year|fiscal)(?:-quarter)? (?:\w+ ){0,2}(?:results|earnings|profit|loss|revenue|sales)\b|\b(?:beats?|beat|tops?|topped|miss(?:es|ed)?|exceeds?|exceeded|falls short of) (?:\w+ ){0,3}(?:estimates|expectations|forecasts|consensus)\b"),
    ("earnings", 1.5, r"\b(?:eps|earnings per share|net income|operating income|same-store sales|earnings call|reports? (?:a )?(?:quarterly |annual )?(?:profit|loss|revenue))\b"),
    ("earnings", 1, r"\b(?:earnings|revenue|quarter|profit|net loss)\b"),
    ("guidance", 2, r"\b(?:rais(?:es|ed|ing)|lift(?:s|ed)|boost(?:s|ed)|cut(?:s|ting)?|lower(?:s|ed|ing)|slash(?:es|ed)|trim(?:s|med)|reaffirm(?:s|ed)|reiterat(?:es|ed)|withdraw(?:s|n)?|suspends?|pulls?) (?:its |full-year |annual |\d{4} )*(?:guidance|outlook|forecast|targets?)\b"),
    ("guidance", 2, r"\b(?:profit warning|warns (?:on|of) (?:\w+ ){0,2}(?:profit|revenue|sales|earnings)|pre-?announce\w*|preliminary (?:results|revenue))\b"),
    ("guidance", 1, r"\b(?:guidance|outlook|forecast)\b"),
    ("analyst", 2, r"\b(?:upgrad|downgrad)(?:es|ed|e) (?:\w+ ){0,4}(?:to |at )?(?:buy|sell|hold|neutral|outperform|underperform|overweight|underweight|equal[- ]weight|market perform|sector perform|strong buy|peer perform)\b"),
    ("analyst", 2, r"\b(?:price target|target price|initiat(?:es|ed) (?:coverage|at)|resum(?:es|ed) coverage|reiterates? (?:buy|sell|hold|outperform)|analysts? (?:say|see|expect|upgrade|downgrade|raise|cut))\b"),
    ("analyst", 1.5, r"\b(?:upgrades?|upgraded|downgrades?|downgraded)\b(?! (?:goes|activat|to (?:the |its )?(?:network|protocol|mainnet)))"),
    ("analyst", 1, r"\b(?:analyst|analysts|rating|brokerage|research note)\b"),
    ("insider", 2, r"\b(?:insiders?|ceo|cfo|chief executive|founder|director|chairman|co-founder|executive)s? (?:\w+ ){0,2}(?:buys?|bought|sells?|sold|purchas(?:es|ed)|offload(?:s|ed)|dump(?:s|ed)|acquires?|acquired|unload(?:s|ed))\b(?! (?:the |a )?(?:company|firm|business|startup|rival))"),
    ("insider", 2, r"\bform 4\b|\b(?:insider|whale) (?:buying|selling|purchase|sale|sells|buys|wallets?|accumulation|transfers?|moves?)\b|\bwhales?\b"),
    ("insider", 1, r"\b(?:stake sale|sells? (?:shares|stake)|dormant (?:wallet|address)|large holders?|top holders?)\b"),
    ("filing", 2, r"\b(?:8-k|10-q|10-k|s-1|s-3|f-1|13d|13g|13f|form 20-f|proxy statement|def 14a|sc 13[dg])\b"),
    ("filing", 1.5, r"\b(?:files?|filed|filing) (?:a |an |its )?(?:\w+ ){0,2}(?:with (?:the )?(?:sec|regulators)|registration statement|prospectus|annual report|quarterly report|disclosure)\b|\bdiscloses? (?:a )?(?:\w+ ){0,2}stake\b|\bregulatory filing\b"),
    ("filing", 1, r"\b(?:filing|filings|disclos(?:es|ed|ure)|late filing|nt 10-[kq])\b"),
    ("regulatory", 2, r"\b(?:sec|cftc|finra|fca|regulators?|watchdog|supervisor|authority|commission|ministry|agency)\b (?:\w+ ){0,2}(?:approv\w*|reject\w*|den(?:y|ies|ied)|ban\w*|charg\w*|fines?|fined|sues?|sued|su(?:ing)|warns?|orders?|grants?|licen[cs]\w*|clears?|cleared|halts?|suspends?|blocks?|probes?|investigat\w*)\b"),
    ("regulatory", 2, r"\b(?:wells notice|cease[- ]and[- ]desist|enforcement action|consent order|licen[cs]e (?:granted|revoked|approved|application)|regulatory approval|banned|ban on|outlaws?)\b"),
    ("regulatory", 1, r"\b(?:regulator|regulators|regulatory|compliance|licen[cs]e|lawmakers|legislation|bill passes|stablecoin bill|framework)\b"),
    ("legal", 2, r"\b(?:lawsuit|class action|sues|sued|suing|settle(?:s|d|ment)?|court (?:rules|ruling|rejects|approves|dismisses|orders)|judge|jury|verdict|indicted|indictment|pleads? guilty|convicted|sentenced|subpoena|antitrust)\b"),
    ("legal", 2, r"\b(?:chapter (?:11|7|15)|files? for (?:bankruptcy|insolvency|creditor protection)|bankruptcy (?:filing|court|protection)|insolven(?:t|cy)|liquidators?|receivership|administrators? appointed)\b"),
    ("legal", 1, r"\b(?:investigation|probe|charges|charged|allegations|alleged|fraud|court|legal|attorney general|prosecutors?|arrested)\b"),
    ("security", 2, r"\b(?:exploit(?:s|ed)?|hack(?:s|ed|er|ers)?|drain(?:s|ed)?|stolen|theft|heist|attacker|attackers|breach(?:ed)?|compromised|private keys?|flash loan|reentrancy|rug pull|phishing)\b"),
    ("security", 1.5, r"\b(?:vulnerability|vulnerabilities|bug bounty|critical bug|zero-day|white ?hat|security incident|cyberattack|ransomware|malware|audit (?:finds|found|flags))\b"),
    ("security", 0.5, r"\b(?:security|audit|wallet safety)\b"),
    ("outage", 2, r"\b(?:outage|outages|downtime|went down|goes down|offline|stopped producing blocks|block production|halts?|halted|halting|frozen|freezes?|paus(?:es|ed|ing) (?:withdrawals|deposits|trading)|suspends? (?:withdrawals|deposits|trading))\b"),
    ("outage", 1.5, r"\b(?:degraded performance|service disruption|connectivity issues|network congestion|stalled|chain (?:stops|stalls)|restor(?:es|ed)|back online|resum(?:es|ed) (?:trading|withdrawals|block production|service))\b"),
    ("unlock", 2, r"\b(?:token unlocks?|unlock(?:s|ed|ing)?|vesting cliff|cliff unlock|(?:secondary|public|share|stock|follow-on|at-the-market|atm|equity|convertible (?:note|bond)|discounted) offering|dilution|dilutive|new shares|share sale)\b"),
    ("unlock", 1, r"\b(?:supply (?:release|overhang|inflation)|emissions|vesting|lock-?up (?:expiry|expires|ends)|prices? (?:\w+ ){0,2}offering)\b"),
    ("funding", 2, r"\b(?:rais(?:es|ed|ing)|secur(?:es|ed)|closes?|closed|lands?) " + _N + r" (?:\w+ ){0,4}(?:funding|round|seed|series [a-f]|investment|financing|from investors|raise)\b|\bseries [a-f]\b|\bseed round\b|\bfunding round\b"),
    ("funding", 2, r"\b(?:treasury (?:purchase|buy|reserve|strategy|allocation)|adds? (?:\w+ ){0,3}to (?:its )?(?:balance sheet|treasury)|strategic investment|invests? " + _N + r"|grant program|ecosystem fund|backed by)\b"),
    ("funding", 1, r"\b(?:funding|investors|venture|valuation|raise|capital injection|grant)\b"),
    ("etf", 2, r"\b(?:etf|etfs|etp|etps|exchange-traded (?:fund|product)s?|spot (?:\w+ ){0,1}fund|19b-4|n-1a)\b"),
    ("etf", 1, r"\b(?:inflows?|outflows?|net flows|fund flows|aum|assets under management|creations|redemptions)\b"),
    ("macro", 2, r"\b(?:fomc|federal reserve|central bank|rate (?:cut|cuts|hike|hikes|decision)|interest rates?|inflation|cpi|pce|jobs report|nonfarm payrolls?|payrolls|unemployment|jobless claims|gdp|recession|treasury yields?|bond yields?|tariffs?|trade war|stimulus|quantitative (?:easing|tightening)|basis points?)\b"),
    ("macro", 1, r"\b(?:fed|economy|economic data|policymakers|dollar index|risk-off|risk-on|monetary policy|yields)\b"),
    ("mna", 2, r"\b(?:merger|merge with|to merge|acquisition|acquires?|acquired|to acquire|buyout|takeover|tender offer|spin-?off|spins off|agrees to buy|agreed to buy|to be bought|to buy (?:rival|startup|maker))\b"),
    ("mna", 1, r"\b(?:deal valued|all-stock deal|all-cash deal|bid for|bidder|suitor|activist|go-private|take-private)\b"),
    ("capital_return", 2, r"\b(?:buy-?backs?|share repurchases?|repurchase (?:program|plan)|dividends?|special dividend|payout|token burns?|burn(?:s|ed|ing)? " + _N + r"|buy-and-burn|burn mechanism|fee switch|revenue share with (?:holders|stakers))\b"),
    ("capital_return", 1, r"\b(?:returns? (?:cash|capital) to (?:shareholders|holders)|yield to holders|staking rewards? (?:raised|cut))\b"),
    ("market_structure", 2, r"\b(?:funding rates?|open interest|liquidations?|liquidated|short squeeze|short interest|order[- ]book|market depth|basis trade|futures premium|perpetuals?|perps|options expiry|max pain|gamma squeeze|put-call ratio|leverage ratio|days to cover|borrow rate)\b"),
    ("market_structure", 1, r"\b(?:leverage|futures|options|derivatives|liquidity|slippage|spreads? widen|market makers?)\b"),
    ("sentiment", 2, r"\b(?:fear (?:and|&) greed|sentiment|social (?:media )?(?:buzz|volume|mentions|chatter)|trending on|retail (?:frenzy|investors pile|traders pile|interest|mania)|fomo|euphoria|capitulation|crowd mood|meme (?:coin|stock) (?:mania|frenzy)|mentions (?:surge|spike|jump|fall))\b"),
    ("sentiment", 1, r"\b(?:bullish|bearish|hype|mood|chatter|influencers?|reddit|forum|community (?:backlash|outrage|anger)|backlash)\b"),
]
_CAT_RX = [(c, w, re.compile(p, re.I)) for c, w, p in _CAT_RULES]

# ---------------------------------------------------------------------------- direction cues
# (sign, weight, pattern). Positive cues that are negated count as negative; negative cues
# that are negated cancel (a denial is not good news, it is the absence of bad news).
_DIR_RULES = [
    (1, 2, r"\b(?:beats?|beat|tops?|topped|exceeds?|exceeded|surpass(?:es|ed)|crush(?:es|ed)) (?:\w+ ){0,3}(?:estimates|expectations|forecasts|consensus)\b"),
    (1, 2, r"\b(?:rais(?:es|ed|ing)|lift(?:s|ed)|boost(?:s|ed)|hikes?) (?:its |full-year |annual |\d{4} )*(?:guidance|outlook|forecast|price target|target|dividend|payout)\b"),
    (1, 2, r"\bupgrad(?:es|ed|e) (?:\w+ ){0,4}(?:to |at )?(?:buy|outperform|overweight|strong buy|positive|accumulate)\b|\binitiat(?:es|ed) (?:coverage )?(?:\w+ ){0,3}(?:at |with (?:a |an )?)?(?:buy|outperform|overweight)\b"),
    (1, 2, r"\b(?:approv(?:es|ed|al)|green[- ]?lights?|clears?|cleared|grants?|granted|wins? (?:approval|licen[cs]e|lawsuit|case|appeal|ruling)|won (?:approval|the case|its appeal))\b"),
    (1, 2, r"\b(?:lists?|will list|to list|listing on|adds? (?:\w+ ){0,2}trading|added to (?:the )?(?:\w+ ){0,3}index|joins? (?:the )?(?:\w+ ){0,3}index|begins trading)\b(?! (?:as|among|below)\b)"),
    (1, 1.5, r"\b(?:partner(?:s|ed|ship)? with|teams? up with|integrat(?:es|ed|ion) with|signs?|signed|inks?|inked|wins? (?:a |an )?(?:\w+ ){0,3}(?:contract|deal|order)|secur(?:es|ed) (?:a |an )?(?:\w+ ){0,3}(?:contract|deal|order)|adopts?|adopted)\b"),
    (1, 1.5, r"\b(?:record (?:revenue|profit|sales|inflows|high in (?:revenue|users))|surge in (?:revenue|profit|users|demand)|strong demand|returns? to profit|turns? profitable)\b"),
    (1, 1.5, r"\b(?:buy-?backs?|repurchases?|special dividend|raises? (?:its |the )?dividend|initiates? (?:a )?dividend|token burns?|burn(?:s|ed) " + _N + r"|buy-and-burn)\b"),
    (1, 1.5, r"\b(?:rais(?:es|ed)|secur(?:es|ed)|closes?|closed|lands?) " + _N + r"|\b(?:invests? in|strategic investment|treasury purchase|adds? (?:\w+ ){0,3}to (?:its )?(?:balance sheet|treasury))\b"),
    (1, 1.5, r"\b(?:inflows?|net (?:buying|inflows)|accumulat(?:es|ed|ing|ion)|insider buying|insiders? (?:buys?|bought|purchas(?:es|ed))|(?:ceo|cfo|director|founder|chairman)s? (?:\w+ ){0,2}(?:buys?|bought|purchas(?:es|ed)))\b"),
    (1, 1.5, r"\b(?:rate cuts?|cuts? (?:interest )?rates|cooler(?:-than-expected)?|inflation (?:eases|cools|slows)|soft landing|dovish|stimulus)\b"),
    (1, 1.5, r"\b(?:to be (?:acquired|bought)|takeover (?:bid|offer|approach) for|agrees? to be (?:acquired|bought)|buyout offer|premium of \d+%|receives? (?:a )?(?:takeover|buyout|acquisition) (?:bid|offer))\b"),
    (1, 1.5, r"\b(?:short squeeze|shorts? (?:liquidated|squeezed|wiped out)|funding (?:rates? )?(?:turns?|flips?) (?:positive|negative)? ?(?:and )?shorts|extreme fear (?:often|has historically) marked)\b"),
    (1, 1, r"\b(?:upgrade (?:goes|went) live|activat(?:es|ed) (?:successfully|smoothly|on schedule)|successfully|launch(?:es|ed)|goes live|went live|unveil(?:s|ed)|rolls? out|restor(?:es|ed)|back online|resum(?:es|ed)|recover(?:s|ed)|dismiss(?:es|ed)|acquitted|settles? (?:\w+ ){0,3}(?:for|with)|bullish|optimism|euphoria|greed|soars? (?:on|after) (?:approval|deal|news)|expands? (?:to|into))\b"),
    (-1, 2, r"\b(?:miss(?:es|ed)?|falls? short of|fell short of|lags?|trails?|below) (?:\w+ ){0,3}(?:estimates|expectations|forecasts|consensus)\b"),
    (-1, 2, r"\b(?:cut(?:s|ting)?|lower(?:s|ed|ing)|slash(?:es|ed)|trim(?:s|med)|reduc(?:es|ed)|withdraw(?:s|n)?|suspends?|pulls?) (?:its |full-year |annual |\d{4} )*(?:guidance|outlook|forecast|price target|target|dividend|payout)\b|\bprofit warning\b|\bwarns (?:on|of) (?:\w+ ){0,2}(?:profit|revenue|sales|earnings|demand)\b"),
    (-1, 2, r"\bdowngrad(?:es|ed|e)\b|\b(?:initiat(?:es|ed) (?:coverage )?(?:\w+ ){0,3}(?:at |with (?:a |an )?)?(?:sell|underperform|underweight))\b"),
    (-1, 2, r"\bde-?list(?:s|ed|ing|ment)?\b|\b(?:non-?compliance|deficiency) notice\b|\b(?:ends|ending|to end) (?:spot |margin )?(?:trading|support) for\b"),
    (-1, 2, r"\b(?:exploit(?:ed|s)?|hack(?:ed|s)?|drain(?:ed|s)?|stolen|theft|heist|breach(?:ed)?|compromised|rug pull|attackers? (?:stole|drained|made off))\b"),
    (-1, 2, r"\b(?:outage|downtime|halt(?:s|ed)?|frozen|freezes?|stopped producing blocks|offline|paus(?:es|ed) (?:withdrawals|deposits)|suspends? (?:withdrawals|deposits|trading))\b"),
    (-1, 2, r"\b(?:sues?|sued|suing|lawsuit|class action|charg(?:es|ed)|indict(?:ed|ment)|fraud|ponzi|subpoena|wells notice|cease[- ]and[- ]desist|enforcement action|fine[sd]?|penalt(?:y|ies)|bans?|banned|probe|investigation|arrested|convicted|guilty|sanction(?:s|ed))\b"),
    (-1, 2, r"\b(?:rejects?|rejected|rejection of|den(?:ies|ied|y)|blocks?|blocked|vetoe?s?|vetoed|delays?|delayed|postpones?|postponed|revok(?:es|ed)) (?:\w+ ){0,3}(?:application|applications|bid|proposal|request|filing|etf|listing|approval|merger|deal|offer|launch|upgrade|licen[cs]e|decision|plan)s?\b"),
    (-1, 2, r"\b(?:chapter (?:11|7)|bankrupt(?:cy)?|insolven(?:t|cy)|liquidators?|receivership|creditor protection|winds? down|wind-down|shuts? down|shutting down|collapses?|collapsed)\b"),
    (-1, 1.5, r"\b(?:unlock(?:s|ed)?|dilution|dilutive|(?:secondary|public|share|stock|follow-on|at-the-market|discounted|equity) offering|share sale|new shares|lock-?up (?:expiry|expires|ends))\b"),
    (-1, 1.5, r"\b(?:outflows?|insider selling|insiders? (?:sells?|sold|dump(?:s|ed)|unload(?:s|ed))|(?:ceo|cfo|director|founder|chairman)s? (?:\w+ ){0,2}(?:sells?|sold|dump(?:s|ed)|offload(?:s|ed))|whales? (?:dump(?:s|ed)?|sells?|sold|moves? (?:\w+ ){0,4}to (?:exchanges?|an exchange)))\b"),
    (-1, 1.5, r"\b(?:rate hikes?|hikes? (?:interest )?rates|hotter(?:-than-expected)?|inflation (?:accelerates|jumps|surges|heats up|rises)|hawkish|recession|tariffs? (?:on|hike|increase)|jobless claims (?:jump|rise|surge)|stagflation)\b"),
    (-1, 1.5, r"\b(?:longs? (?:liquidated|wiped out|flushed)|long liquidations|liquidation cascade|deleverag\w*|short interest (?:climbs|rises|jumps|hits)|record short interest)\b"),
    (-1, 1, r"\b(?:delay(?:s|ed)?|postpone[sd]?|recall(?:s|ed)?|layoffs?|lays? off|cuts? (?:\d+ )?jobs|net loss widens|loss widens|plunges? (?:on|after)|warns?|weak demand|slump in (?:sales|demand|revenue)|loses? (?:a |its )?(?:\w+ ){0,2}(?:contract|customer|licen[cs]e)|terminat(?:es|ed) (?:\w+ ){0,3}(?:partnership|agreement|deal|contract)|ends? (?:its )?partnership|bearish|panic|extreme fear|capitulation|backlash|vulnerability|critical bug|resigns?|steps down)\b"),
]
_DIR_RX = [(s, w, re.compile(p, re.I)) for s, w, p in _DIR_RULES]

# ---------------------------------------------------------------------------- risk flags
# Each flag needs a strong, specific phrase. "about hacks in general" is not enough: an
# exploit flag wants a victim ("Zephyr exploited", "drained", "$40m stolen").
_FLAG_RULES = {
    "exploit": [
        r"\b(?:was|were|been|gets?|got) (?:hacked|exploited|drained|breached|compromised)\b",
        r"\b[A-Z][\w.-]*(?: [A-Z][\w.-]*){0,3} (?:hacked|exploited|drained)\b",
        r"\b(?:exploited|drained) (?:for|of) " + _N,
        r"\b(?:hack|exploit|attack|breach|heist|theft)\w* (?:\w+ ){0,3}(?:drains?|drained|nets?|netted|costs?|steals?|stole|stolen|loses?|lost) " + _N,
        r"\b(?:attackers?|hackers?|thieves|exploiters?) (?:\w+ ){0,2}(?:drain(?:ed|s)?|st(?:ole|eal|eals)|made off with|siphon(?:ed|s)?|emptied|took)\b",
        _N + r" (?:\w+ ){0,3}(?:stolen|drained|siphoned|lost) (?:in|from|after|during) (?:a |an |the )?(?:\w+ ){0,3}(?:hack|exploit|attack|breach|heist)",
        _N + r" (?:\w+ ){0,2}(?:stolen|drained|siphoned)\b",
        r"\b(?:suffers?|suffered|hit by|victim of|falls? victim to|confirms?|confirmed) (?:a |an )?(?:\w+ ){0,2}(?:exploit|hack|breach|attack|heist)\b",
        r"\b(?:bridge|protocol|exchange|wallet|vault|pool) (?:\w+ ){0,2}(?:drained|exploited|hacked)\b",
        r"\b(?:private keys? (?:were )?(?:compromised|leaked|stolen)|rug[- ]pull(?:ed)?)\b",
    ],
    "halt": [
        r"\b(?:trading|network|chain|blockchain|block production|mainnet|withdrawals and trading) (?:\w+ ){0,2}(?:halted|halts|is halted|was halted|suspended|stopped|stops)\b",
        r"\b(?:halts?|halted|suspends?|suspended|stops?|stopped) (?:all |spot |the )?(?:trading|block production|the (?:network|chain))\b",
        r"\b(?:stopped|stops|ceased) producing blocks\b|\btrading halt\b|\bnetwork halt\b|\bchain halt\b|\bhalted (?:for|after|pending)\b",
    ],
    "bankruptcy": [
        r"\b(?:files?|filed|filing|seeks?|sought) (?:\w+ ){0,2}(?:for )?(?:chapter (?:11|7|15)|bankruptcy|insolvency|creditor protection)\b",
        r"\bchapter (?:11|7|15) (?:bankruptcy|protection|filing|proceedings)\b|\b(?:enters?|entered|placed (?:in|into)) (?:\w+ ){0,1}(?:administration|receivership|liquidation|bankruptcy)\b",
        r"\b(?:is|was|declared|declares?|now) insolvent\b|\binsolvency (?:filing|proceedings)\b|\bliquidators? (?:appointed|named)\b",
        r"\b(?:freezes?|froze|frozen|halts?|halted|suspends?|suspended|pauses?|paused) (?:all |customer |user )?withdrawals\b|\bwithdrawals (?:\w+ ){0,1}(?:frozen|halted|suspended|paused)\b",
    ],
    "delisting": [
        r"\b(?:to |will |plans to |set to )?de-?list(?:s|ed|ing)?\b(?! (?:risk|fears?|concerns?)\b)",
        r"\b(?:delisting|non-?compliance|deficiency) notice\b|\b(?:ends|ending|to end) (?:spot |margin )?trading for\b",
        r"\b(?:removed|removes|removing|dropped|drops) (?:\w+ ){0,3}from (?:the )?(?:\w+ ){0,2}(?:exchange|platform)\b",
    ],
    "fraud": [
        r"\b(?:charged with|accused of|indicted (?:for|on)|convicted of|guilty (?:of|to)|charges? of|alleging|alleges) (?:\w+ ){0,3}(?:fraud|wire fraud|securities fraud|market manipulation|misappropriat\w*|embezzl\w*)\b",
        r"\b(?:fraud|manipulation) (?:charges?|case|indictment|conviction|complaint)\b|\bponzi\b|\brug[- ]pull(?:ed)?\b|\bexit scam\b",
        r"\b(?:accounting (?:irregularities|fraud|scandal)|cooked the books|inflated (?:revenue|user numbers|reserves)|misstat\w+ (?:revenue|earnings|results)|restates? (?:\w+ ){0,2}(?:results|earnings|financials)|fake (?:reserves|volume)|wash trading)\b",
    ],
    "regulatory": [
        r"\b(?:sec|cftc|finra|doj|fca|regulators?|watchdog|authorities|government|ministry|prosecutors?|attorney general|commission)\b (?:\w+ ){0,2}(?:sues?|sued|suing|charg(?:es|ed)|fin(?:es|ed)|bans?|banned|orders? (?:\w+ ){0,2}(?:to (?:stop|halt|cease)|shut)|revok(?:es|ed)|suspends?|sanction(?:s|ed))\b",
        r"\b(?:wells notice|cease[- ]and[- ]desist|enforcement action|consent order)\b",
        r"\b(?:licen[cs]e (?:revoked|suspended|pulled|withdrawn)|banned (?:in|from|by)|bans? (?:on )?(?:trading|sales|the token|the coin|the app))\b",
    ],
    "lawsuit": [
        r"\b(?:sues?|sued|suing|files? (?:a )?suit|filed (?:a )?suit)\b",
        r"\b(?:class[- ]action|lawsuit|complaint) (?:\w+ ){0,2}(?:filed|against|accus\w*|alleg\w*|seeks?|claims?)\b|\b(?:files?|filed|faces?|facing|hit with) (?:a )?(?:\w+ ){0,2}(?:lawsuit|class[- ]action|suit)\b",
    ],
    "dilution": [
        r"\b(?:prices?|priced|announces?|launch(?:es|ed)|files? for|plans?|proposed) (?:a |an |its )?(?:\$?[\d.,]+\s*(?:[mb]n?|million|billion) )?(?:secondary|public|share|stock|follow-on|at-the-market|atm|equity|discounted|registered direct) offering\b",
        r"\b(?:secondary|follow-on|at-the-market|registered direct|discounted) (?:share |stock |equity )?offering\b",
        r"\b(?:unlocks?|unlocking|release[sd]?) (?:of )?(?:about |roughly |nearly |over |some )?(?:\d[\d.,]*%|" + _N + r") (?:\w+ ){0,4}(?:of (?:\w+ ){0,2}(?:supply|tokens|circulating)|tokens|worth|in tokens)",
        r"\b(?:large|massive|major|huge|biggest|record|cliff) (?:token )?unlock\b|\bunlock (?:of|worth|equal to) " + _N,
        r"\b(?:issues?|issuing|sells?|selling) " + _N + r" (?:new |additional )?shares\b|\bdilut(?:es|ion|ive) (?:\w+ ){0,2}(?:shareholders|holders)\b",
    ],
    "outage": [
        r"\b(?:outage|outages|downtime|service disruption|degraded performance)\b",
        r"\b(?:goes|went|is|was) (?:down|offline)\b|\bstopped (?:processing|confirming) (?:transactions|blocks)\b",
        r"\b(?:users|customers|traders) (?:\w+ ){0,2}(?:unable to|cannot|can't|could not) (?:\w+ ){0,1}(?:log in|trade|withdraw|access)\b",
    ],
}
_FLAG_RX = {f: [re.compile(p, re.I if not p.startswith(r"\b[A-Z]") else 0) for p in ps] for f, ps in _FLAG_RULES.items()}

# A cue is negated when one of these sits in the few words before it (same sentence).
_NEGATORS = re.compile(r"\b(?:not|no|never|neither|nor|without|fails? to|failed to|failing to|denies|denied|deny|"
                       r"rejects|rejected|refutes?|dismisses|dismissed|rules? out|ruled out|unlikely to|won't|isn't|aren't|wasn't|"
                       r"weren't|doesn't|didn't|don't|hasn't|haven't|cannot|can't|couldn't|false|fake|unfounded|debunk\w*|"
                       r"avoids?|avoided|averts?|averted|no longer)\b", re.I)
# ... or when a denial follows right after it ("hack rumours denied").
_POST_NEG = re.compile(r"^\W*(?:\w+\W+){0,3}?(?:denied|false|unfounded|debunked|dismissed|ruled out|unconfirmed|not true)\b", re.I)
# Flags are claims that something bad IS happening; a maybe or a how-to is not that claim.
_HYPOTHETICAL = re.compile(r"\b(?:could|may|might|would|if|whether|risks?|fears?|feared|potential|possible|possibly|"
                           r"how|avoid|prevent|protect|lessons?|guide|tips|threat of|rumou?rs?|unconfirmed|speculation|"
                           r"what happens|scenario|history of|biggest|worst|years? ago|last year|since the|anniversary)\b", re.I)
_GENERIC_TITLE = re.compile(r"^\s*(?:how to|how do|what is|what are|what's|why (?:do|does|is|are)|explainer|explained|"
                            r"guide|opinion|op-ed|analysis:|commentary|column|q&a|faq|the (?:\d+ )?(?:biggest|worst|best)|"
                            r"\d+ (?:ways|things|tips|lessons|reasons)|a (?:brief )?history|lessons from|ask )\b", re.I)
_RESOLVED = re.compile(r"\b(?:restor(?:es|ed|ing)|back online|resum(?:es|ed|ing)|resolved|recover(?:s|ed)|reopen(?:s|ed)|"
                       r"lifts? (?:the )?halt|after (?:a |an |the )?(?:brief |short |\d+-hour )?(?:outage|halt|downtime))\b", re.I)
_SENT_BREAK = re.compile(r"[.;!?|\n–—]|\s-\s|:\s")
_WORD = re.compile(r"[\w'$%-]+")


def _s(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    return x if isinstance(x, str) else ""


def _before(s, start, words=4):
    """The few words before position `start`, stopping at the start of the sentence."""
    seg = s[max(0, start - 60):start]
    cut = None
    for m in _SENT_BREAK.finditer(seg):
        cut = m.end()
    if cut is not None:
        seg = seg[cut:]
    return " ".join(_WORD.findall(seg)[-words:])


def _negated(s, m):
    return bool(_NEGATORS.search(_before(s, m.start()))) or bool(_POST_NEG.match(s[m.end():m.end() + 40]))


def _hypothetical(s, m):
    return bool(_HYPOTHETICAL.search(_before(s, m.start(), words=6)))


def _sentence(s, m):
    """The sentence (or headline clause) around a match."""
    a = 0
    for b in _SENT_BREAK.finditer(s, 0, m.start()):
        a = b.end()
    e = _SENT_BREAK.search(s, m.end())
    return s[a:e.start() if e else len(s)]


def _short(t):
    t = " ".join(_s(t).split())
    return t if len(t) <= 40 else t[:39] + "…"


def _flags(title, body):
    """Risk flags named by strong phrases, skipping negated, hypothetical and resolved ones."""
    found, why = [], {}
    generic = bool(_GENERIC_TITLE.search(title))
    for part in (title, body):
        if not part or generic:
            continue
        for flag, rxs in _FLAG_RX.items():
            if flag in why:
                continue
            for rx in rxs:
                for m in rx.finditer(part):
                    if _negated(part, m) or _hypothetical(part, m):
                        continue
                    if flag in ("outage", "halt") and _RESOLVED.search(_sentence(part, m)):
                        continue
                    why[flag] = m.group(0)
                    break
                if flag in why:
                    break
    found = [f for f in RISK_FLAGS if f in why]
    return found, why


def rule_label(title, text="") -> dict:
    """Label one story with keyword rules.

    Returns dict(catalyst, direction, risk_flags, confidence, why): catalyst from
    taxonomy.CATALYSTS ("other" with confidence 0.3 when nothing matches), direction -1/0/+1,
    risk_flags in taxonomy order, confidence in [0.3, 0.95], and `why` = the matched cues as a
    short string ("" when nothing matched). Never raises on odd input.
    """
    title = " ".join(_s(title).split())[:600]
    body = " ".join(_s(text).split())[:TEXT_MAX]

    # catalyst: best total score; title matches count double, each pattern counts once per part
    score, cue = {}, {}
    for cat, w, rx in _CAT_RX:
        for part, mult in ((title, 2.0), (body, 1.0)):
            m = rx.search(part) if part else None
            if m:
                score[cat] = score.get(cat, 0.0) + w * mult
                if cat not in cue or mult == 2.0 and cue[cat][1] < 2.0:
                    cue[cat] = (m.group(0), mult)

    flags, flag_why = _flags(title, body[:FLAG_TEXT_MAX])
    # a flag is strong evidence for its catalyst even when the catalyst words were thin
    for f in flags:
        cat = _FLAG_CATALYST.get(f)
        if cat:
            score[cat] = score.get(cat, 0.0) + 1.5
            cue.setdefault(cat, (flag_why[f], 1.0))

    # direction: signed cue weights, negation-aware
    dscore, dcue = 0.0, ""
    for sign, w, rx in _DIR_RX:
        for part, mult in ((title, 2.0), (body, 1.0)):
            if not part:
                continue
            for m in rx.finditer(part):
                neg = _negated(part, m)
                s = -sign if (neg and sign > 0) else (0 if neg else sign)
                if s:
                    dscore += s * w * mult
                    if not dcue or mult == 2.0 and abs(w) >= 2:
                        dcue = ("not " if neg else "") + m.group(0)
                break   # one count per cue per part: repetition is not extra evidence
    direction = (dscore > 0.75) - (dscore < -0.75)
    if flags:   # a credible flag is bad news whatever else the story says
        direction = -1

    if not score:
        return dict(catalyst="other", direction=direction if flags else 0, risk_flags=flags, confidence=CONF_MIN,
                    why=_short(dcue) if flags else "")
    ranked = sorted(score.items(), key=lambda kv: (-kv[1], CATALYSTS.index(kv[0])))
    best_cat, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    raw = CONF_MIN + 0.65 * (1.0 - math.exp(-best / 2.5))
    margin = (best - second) / best if best > 0 else 0.0
    conf = CONF_MIN + (raw - CONF_MIN) * (0.55 + 0.45 * margin)
    if flags:
        conf += 0.05
    conf = round(min(CONF_MAX, max(CONF_MIN, conf)), 3)

    why = f"{best_cat}: “{_short(cue[best_cat][0])}”"
    if dcue and direction:
        why += f"; {'+' if direction > 0 else '-'}: “{_short(dcue)}”"
    if flags:
        why += "; flags: " + ", ".join(flags)
    return dict(catalyst=best_cat, direction=int(direction), risk_flags=flags, confidence=conf, why=why[:WHY_MAX])


_FLAG_CATALYST = {"exploit": "security", "halt": "outage", "outage": "outage", "delisting": "delisting",
                  "bankruptcy": "legal", "lawsuit": "legal", "fraud": "legal", "regulatory": "regulatory",
                  "dilution": "unlock"}
