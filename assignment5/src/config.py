"""Sample definition, ticker/topic maps and paths. Every other module reads from here."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# Sample window: the 2026 midterm campaign up to the build date. Election day is 3 Nov 2026.
START = os.environ.get("A5_START", "2026-03-02")
END = os.environ.get("A5_END", "2026-10-09")
ELECTION_DAY = "2026-11-03"
TZ = "America/New_York"
MARKET_CLOSE_HOUR = 16  # text stamped at/after 16:00 ET belongs to the next trading day

# ---------------------------------------------------------------------------
# Voter priority topics: seed vocabulary (used to label unsupervised topics)
# and the sector ETFs each topic is hypothesised to move.
# ---------------------------------------------------------------------------
TOPICS: dict[str, list[str]] = {
    "Economy/Inflation": [
        "inflation", "prices", "groceries", "wages", "economy", "jobs", "tariffs",
        "recession", "gas", "fed", "unemployment", "layoffs", "paycheck",
    ],
    "Housing Affordability": [
        "housing", "mortgage", "mortgages", "homebuyers", "homebuyer", "rent", "rents",
        "homeownership", "downpayment", "starter", "homes", "affordability", "zoning",
    ],
    "Healthcare Affordability": [
        "premiums", "deductible", "deductibles", "copay", "subsidies", "aca", "obamacare",
        "drug", "drugs", "prescription", "insulin", "medical", "medications", "seniors",
        "uninsured", "copays",
    ],
    "Healthcare": [
        "healthcare", "care", "medicare", "medicaid", "hospital", "hospitals", "coverage",
        "insurance", "doctors", "pharma", "abortion", "vaccines", "rural",
    ],
    "Energy/Climate": [
        "energy", "climate", "oil", "drilling", "solar", "wind", "renewable",
        "emissions", "pipeline", "electricity", "ev", "fracking", "clean",
    ],
    "Immigration": [
        "immigration", "border", "migrants", "asylum", "deportation", "ice",
        "visa", "citizenship", "illegal", "wall", "daca",
    ],
    "Foreign Policy/Defense": [
        "defense", "military", "ukraine", "china", "taiwan", "nato", "israel",
        "iran", "war", "troops", "pentagon", "missiles", "sanctions",
    ],
    "Technology/AI": [
        "ai", "artificial", "tech", "algorithm", "chips", "data", "privacy",
        "regulation", "automation", "deepfakes", "antitrust", "semiconductor",
    ],
}

# Topic -> ETFs, with the sign of the hypothesised exposure to a rise in the
# probability of Democratic control (+1 benefits, -1 hurts, 0 ambiguous).
SECTOR_MAP: dict[str, dict[str, int]] = {
    "Foreign Policy/Defense": {"XAR": -1, "ITA": -1},
    "Energy/Climate": {"ICLN": +1, "XLE": -1},
    "Healthcare": {"XLV": -1},
    "Healthcare Affordability": {"XLV": -1},
    "Housing Affordability": {"ITB": 0, "XHB": 0},
    "Economy/Inflation": {"XLF": -1},
    "Technology/AI": {"XLK": 0},
}
SECTOR_ETFS = ["XLE", "ICLN", "XLV", "XLF", "XAR", "ITA", "XLK", "ITB", "XHB"]
# Single-name midterm basket, by policy exposure
BASKET_STOCKS = {
    "Clean energy": ["ENPH"], "Traditional energy": ["XOM"], "Managed care": ["UNH"],
    "Banks": ["JPM"], "Defense": ["LMT"],
    "Big Tech / AI": ["NVDA", "MSFT", "GOOGL", "META", "AAPL"],
}
STOCKS = [t for v in BASKET_STOCKS.values() for t in v]
ASSETS = SECTOR_ETFS + STOCKS
# Same-exposure funds from other issuers, used only to audit each feed day by day
SIBLINGS = {"SPY": ["IVV", "VOO"], "XLE": ["VDE"], "XLV": ["VHT"], "XLF": ["VFH"],
            "XLK": ["VGT"], "XAR": ["ITA"], "ITB": ["XHB"]}
BENCHMARK = "SPY"
YAHOO_MACRO = {"VIX": "^VIX"}
FRED_SERIES = {"DGS2": "DGS2", "DGS3MO": "DGS3MO"}

# ---------------------------------------------------------------------------
# Prediction-market contracts. Identifiers change between cycles and must be
# checked against each venue before a live run; override with env vars.
# ---------------------------------------------------------------------------
CHAMBERS = ("House", "Senate")
# Contracts tracked: chamber control plus the Democratic sweep (D House and D Senate)
CONTRACTS = ("House", "Senate", "Sweep")
# Polymarket markets are located by question text via Gamma /public-search, so no slug is pinned.
KALSHI_SERIES = {
    "House": os.environ.get("KALSHI_HOUSE_SERIES", "CONTROLH"),
    "Senate": os.environ.get("KALSHI_SENATE_SERIES", "CONTROLS"),
    "Sweep": os.environ.get("KALSHI_SWEEP_SERIES", "KXBALANCEPOWERCOMBO"),
}
KALSHI_EVENTS = {  # chamber markets are <event>-D/-R; the combo is <event>-DD/-DR/-RD/-RR
    "House": os.environ.get("KALSHI_HOUSE_EVENT", "CONTROLH-2026"),
    "Senate": os.environ.get("KALSHI_SENATE_EVENT", "CONTROLS-2026"),
    "Sweep": os.environ.get("KALSHI_SWEEP_EVENT", "KXBALANCEPOWERCOMBO-27FEB"),
}
KALSHI_OUTCOMES = {"House": ("D", "R"), "Senate": ("D", "R"),
                   "Sweep": ("DD", "DR", "RD", "RR")}  # House first, then Senate
POLYMARKET_SWEEP_SEARCH = "Balance of Power: 2026 Midterms"
PREDICTIT_MARKET_IDS = {  # "Which party will win the House in the 2026 election?" etc.
    "House": int(os.environ.get("PREDICTIT_HOUSE_ID", 8157)),
    "Senate": int(os.environ.get("PREDICTIT_SENATE_ID", 8155)),
}
# Venue weights in the consensus probability, renormalised over the venues quoting
# that day. PredictIt only publishes daily bars that close at midnight ET, which
# cannot be aligned to the 16:00 equity close, so it is kept as a cross-check
# series with zero weight.
VENUE_WEIGHTS = {"polymarket": 0.5, "kalshi": 0.5, "predictit": 0.0}

# GDELT DOC 2.0 queries: midterm coverage that also mentions each issue. GDELT
# ignores words under three letters, so short seeds (ai, ev, ice) are spelled out.
GDELT_MIDTERMS = '(midterms OR "midterm election" OR "midterm elections")'
GDELT_TOPIC_TERMS = {
    "Economy/Inflation": '(inflation OR "grocery prices" OR economy OR tariffs OR jobs)',
    "Housing Affordability": ('(mortgage OR "mortgage rates" OR "home prices" OR "housing costs" '
                              'OR homebuyers OR "housing affordability" OR rent)'),
    "Healthcare Affordability": ('("health insurance premiums" OR "drug prices" OR "ACA subsidies" '
                                 'OR "health care costs" OR "out-of-pocket" '
                                 'OR "prescription drug")'),
    "Healthcare": '(healthcare OR medicare OR medicaid OR hospitals OR "health coverage")',
    "Energy/Climate": '(energy OR climate OR oil OR gasoline OR renewable OR electricity)',
    "Immigration": '(immigration OR border OR migrants OR deportation OR asylum)',
    "Foreign Policy/Defense": '(defense OR military OR ukraine OR china OR israel OR iran)',
    "Technology/AI": '("artificial intelligence" OR "big tech" OR semiconductor OR "data privacy")',
}
GDELT_FILTERS = "sourcecountry:US sourcelang:english"

# ---------------------------------------------------------------------------
# Social media quality filters (bot and astroturf screening).
# ---------------------------------------------------------------------------
TWITTERAPI_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
TWITTERAPI_INFO_URL = "https://api.twitterapi.io/oapi/my/info"
# Credit budget: 15 credits per returned tweet (minimum 15 per call). The ledger in
# data/raw/tweets/ledger.csv is the authority, since the account balance updates late.
TWITTER_BUDGET_CREDITS = int(os.environ.get("TWITTER_BUDGET_CREDITS", 100_000))
TWITTER_BALANCE_FLOOR = int(os.environ.get("TWITTER_BALANCE_FLOOR", 15_000))
CREDITS_PER_TWEET = 15
# Issue terms OR political context, English originals with some engagement but not viral
# (a band that favours ordinary voters over national accounts), sorted by relevance.
# (the API rejects queries over 512 characters, dates and filters included)
TWITTER_ISSUES = ('(inflation OR "grocery prices" OR tariffs OR mortgage OR "home prices" OR '
                  'rent OR premiums OR "drug prices" OR obamacare OR medicaid OR medicare OR '
                  'energy OR climate OR border OR immigration OR deportation OR ukraine OR '
                  'china OR military OR "artificial intelligence")')
TWITTER_CONTEXT = '(midterms OR midterm OR congress OR democrats OR republicans OR GOP OR ballot)'
# Affordability stratum: oversamples the pocketbook issues (housing and healthcare
# costs). Used for topic tone only; salience comes from the general stratum.
TWITTER_AFFORDABILITY = ('(mortgage OR "mortgage rates" OR "home prices" OR rent OR '
                         '"housing costs" OR homebuyers OR "first-time buyers" OR premiums OR '
                         'deductibles OR "drug prices" OR "health insurance" OR obamacare OR '
                         '"ACA subsidies" OR "medical bills")')
TWITTER_FILTERS = "lang:en -is:retweet -is:reply min_faves:5 -min_faves:150"
MIN_FOLLOWERS = 10
MIN_VIEWS = 100
ASTROTURF_MIN_ACCOUNTS = 5  # identical normalised text from >= this many accounts is dropped

# LSA: k chosen by interpretability and stability (see BRAINSTORMING.md section 1.5)
LSA_K = 30

# Econometrics defaults
MAX_LAGS = 5
HIGH_NEWS_QUANTILE = 0.80  # days above this volume quantile form the Rigobon-Sack H regime
SEED = 7871

for _d in (RAW_DIR, INTERIM_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
