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
        "inflation", "prices", "groceries", "rent", "wages", "economy", "jobs",
        "tariffs", "cost", "mortgage", "recession", "gas", "affordability", "fed",
    ],
    "Healthcare": [
        "healthcare", "medicare", "medicaid", "insurance", "premiums", "aca",
        "obamacare", "drug", "prescription", "hospital", "coverage", "pharma",
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
    "Economy/Inflation": {"XLF": -1},
    "Technology/AI": {"XLK": 0},
}
SECTOR_ETFS = ["XLE", "ICLN", "XLV", "XLF", "XAR", "ITA", "XLK"]
BENCHMARK = "SPY"
YAHOO_MACRO = {"VIX": "^VIX"}
FRED_SERIES = {"DGS2": "DGS2", "DGS3MO": "DGS3MO"}

# ---------------------------------------------------------------------------
# Prediction-market contracts. Identifiers change between cycles and must be
# checked against each venue before a live run; override with env vars.
# ---------------------------------------------------------------------------
CHAMBERS = ("House", "Senate")
# Polymarket markets are located by question text via Gamma /public-search, so no slug is pinned.
KALSHI_SERIES = {
    "House": os.environ.get("KALSHI_HOUSE_SERIES", "CONTROLH"),
    "Senate": os.environ.get("KALSHI_SENATE_SERIES", "CONTROLS"),
}
KALSHI_EVENTS = {  # markets are <event>-D and <event>-R
    "House": os.environ.get("KALSHI_HOUSE_EVENT", "CONTROLH-2026"),
    "Senate": os.environ.get("KALSHI_SENATE_EVENT", "CONTROLS-2026"),
}
# Venue weights in the consensus probability (liquidity-motivated; renormalised
# over whichever venues report on a given day).
VENUE_WEIGHTS = {"polymarket": 0.45, "kalshi": 0.45, "predictit": 0.10}

# ---------------------------------------------------------------------------
# Social media quality filters (bot and astroturf screening).
# ---------------------------------------------------------------------------
TWITTERAPI_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
MIN_FOLLOWERS = 10
MIN_VIEWS = 100
ASTROTURF_MIN_ACCOUNTS = 5  # identical normalised text from >= this many accounts is dropped

# Econometrics defaults
MAX_LAGS = 5
HIGH_NEWS_QUANTILE = 0.80  # days above this volume quantile form the Rigobon-Sack H regime
SEED = 7871

for _d in (RAW_DIR, INTERIM_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
