"""Sample construction: the Table 1 waterfall, end to end.

The waterfall starts at the ARK holdings snapshot and ends at the estimation
sample, so the universe-level counts produced by scripts 00-01 and the
filing-level filters specified in the brief appear in one running table with a
`remaining` column at every step.

Every filter that touches the data is recorded. Two of the recorded steps are
not in the brief's numbered list and are flagged as such in the `note` column:

*   **Amendments.** ``EdgarClient.list_filings`` is called with
    ``include_amendments=False`` by script 02, so 10-K/A and 10-Q/A filings were
    excluded before anything was downloaded. They are counted here by re-listing
    with amendments included, so the row is reported rather than hidden.
*   **Duplicate accessions.** Script 02 iterates over tickers, and Alphabet is
    held by ARK under two share classes (GOOG and GOOGL) that share one CIK.
    Its filings were therefore downloaded twice under two labels. These are the
    same document, not two filings.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    PricePanel,
    Waterfall,
    assign_day0,
    one_per_firm_quarter,
    trading_calendar,
)
from .config import DATA, INTERIM_DIR, PRICE_DIR, ROOT, UNIVERSE_DIR

# Thresholds fixed by the brief so Table 1 is comparable across the class.
MIN_WORDS = {"10-K": 2000, "10-Q": 1000}
MIN_PRICE = 3.00
MIN_TRADING_DAYS = 60

# Event windows, also fixed by the brief.
EVENT_WINDOW = (0, 3)            # filing-period excess return vs SPY
PRE_WINDOW = (-60, -6)           # pre-filing control window
POST_VOL_WINDOW = (4, 63)        # post-filing realised volatility

# Alphabet is one company. Keep the Class A line; GOOG (Class C) is the
# duplicate. The choice is stated rather than left to row order.
PREFERRED_TICKER_FOR_CIK = {"0001652044": "GOOGL"}


def load_inputs() -> dict:
    """Read everything the pipeline produced."""
    meta = pd.read_csv(
        INTERIM_DIR / "filings_meta.csv",
        dtype={"cik": str},
        parse_dates=["filing_date", "report_date"],
    )
    meta["acceptance_datetime"] = pd.to_datetime(
        meta["acceptance_datetime"], utc=True, errors="coerce", format="mixed"
    )
    prices = pd.read_csv(PRICE_DIR / "prices.csv", index_col=0, parse_dates=True)
    volume = pd.read_csv(PRICE_DIR / "volume.csv", index_col=0, parse_dates=True)
    shares = pd.read_csv(PRICE_DIR / "shares.csv", dtype={"cik": str})
    universe = pd.read_csv(UNIVERSE_DIR / "universe.csv", dtype={"cik": str})
    ark = pd.read_csv(UNIVERSE_DIR / "ark_holdings_raw.csv")
    amended = None
    p = INTERIM_DIR / "filings_with_amendments.csv"
    if p.exists():
        amended = pd.read_csv(p, dtype={"cik": str})
    return {
        "meta": meta, "prices": prices, "volume": volume, "shares": shares,
        "universe": universe, "ark": ark, "amended": amended,
    }


def read_counts(text_path: str) -> tuple[dict, int, int]:
    """Token counts for one filing, from the gzipped extracted text."""
    from .parse import tokenize

    path = Path(text_path)
    if not path.is_absolute():
        path = ROOT / path
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        tokens = tokenize(fh.read())
    from collections import Counter

    return Counter(tokens), len(tokens), len(set(tokens))


def build_sample(inputs: dict | None = None) -> dict:
    """Apply the waterfall and return the estimation sample plus Table 1."""
    d = inputs or load_inputs()
    meta, prices, volume = d["meta"], d["prices"], d["volume"]
    universe, ark, amended = d["universe"], d["ark"], d["amended"]

    calendar = trading_calendar(prices, "SPY")
    panel = PricePanel(prices, calendar, volume)

    # ---------------- universe level ----------------
    n_positions = len(ark)
    # universe.csv holds only the tickers that matched a CIK, so the seven
    # unmatched ones have to be added back to recover the pre-match count.
    n_no_cik = 7   # reported by script 01: ADYEN, ARKY, BYDDY, HO, KMTUY, PRNT, SE
    n_matched = int(universe["ticker"].nunique())
    n_tickers = n_matched + n_no_cik
    n_filers = int((universe["status"] == "domestic_filer").sum())
    n_no_10x = n_matched - n_filers
    n_ciks = int(universe.loc[universe["status"] == "domestic_filer", "cik"].nunique())

    n_amendments = 0
    if amended is not None:
        n_amendments = int(amended["form"].str.contains("/A", na=False).sum())
    n_candidates = len(meta) + n_amendments

    wf = Waterfall("ARK holdings positions (6 funds, frozen snapshot)", n_positions,
                   "union of the six funds' published holdings")
    wf.record("Unique US-listed tickers after cleaning",
              n_positions - n_tickers, n_tickers,
              "drops cross-fund duplicates, crypto and non-US listings")
    wf.record("Drop: no CIK on file (foreign or private filer)",
              n_no_cik, n_matched, "ADYEN, ARKY, BYDDY, HO, KMTUY, PRNT, SE")
    wf.record("Drop: CIK found, no 10-K or 10-Q in 2021-2025",
              n_no_10x, n_filers,
              f"foreign private issuers filing 20-F/40-F, and recent IPOs. "
              f"{n_filers} tickers = {n_ciks} distinct CIKs (GOOG and GOOGL share one)")
    wf.record("Candidate 10-K / 10-Q filings, 2021-2025", np.nan, n_candidates,
              "includes amendments, before any filing-level filter")

    # ---------------- filing level ----------------
    df = meta.copy()

    # Step 1: amendments and parse failures.
    n_parse_fail = int((df["n_words"] == 0).sum())
    wf.record("1. Drop amendments (10-K/A, 10-Q/A) and parse failures",
              n_amendments + n_parse_fail, len(df) - n_parse_fail,
              f"{n_amendments} amendments excluded at the EDGAR query stage; "
              f"{n_parse_fail} filings failed to parse")
    df = df[df["n_words"] > 0]

    # Step 1b: duplicate accessions (not in the brief's list; reported anyway).
    def _keep_dupe(cik: str, sub: pd.DataFrame) -> pd.Series:
        pref = PREFERRED_TICKER_FOR_CIK.get(cik)
        if pref is not None and (sub["ticker"] == pref).any():
            return sub["ticker"] == pref
        return sub["ticker"] == sorted(sub["ticker"].unique())[0]

    dup_mask = pd.Series(True, index=df.index)
    for cik, sub in df.groupby("cik"):
        if sub["accession"].nunique() < len(sub):
            dup_mask.loc[sub.index] = _keep_dupe(cik, sub)
    df = wf.apply(df, dup_mask,
                  "1b. Drop duplicate accessions (same CIK, two share classes)",
                  "not in the brief's list. Alphabet is held as GOOG and GOOGL, "
                  "one CIK; script 02 iterates tickers so its filings arrived twice. "
                  "Class A (GOOGL) kept")

    # Step 2: minimum word count, by form.
    floor = df["form"].map(MIN_WORDS)
    df = wf.apply(df, df["n_words"] >= floor,
                  "2. Drop filings under the word-count floor",
                  "2,000 words for a 10-K, 1,000 for a 10-Q")

    # Step 3: one filing per company per calendar quarter, earliest kept.
    df["cal_quarter"] = df["filing_date"].dt.to_period("Q")
    df = wf.apply(df, pd.Series(one_per_firm_quarter(df), index=df.index),
                  "3. Keep one filing per company per calendar quarter",
                  "earliest filing date kept; ties broken by accession. "
                  "Quarter is the quarter of the filing date")

    # Day 0, before the filters that depend on it.
    df = assign_day0(df, calendar)

    # Step 4: usable day 0 and a day -1 price of at least $3.
    df["price_m1"] = [
        panel.price_at(t, d0, -1) if pd.notna(d0) else np.nan
        for t, d0 in zip(df["ticker"], df["day0"])
    ]
    usable = df["day0"].notna() & df["price_m1"].notna() & (df["price_m1"] >= MIN_PRICE)
    n_no_day0 = int((df["day0"].isna() | df["price_m1"].isna()).sum())
    df = wf.apply(df, usable,
                  "4. Require a usable day 0 and a day -1 price of at least $3",
                  f"{n_no_day0} lacked a tradable day 0 or a day -1 close; "
                  f"the rest were sub-$3. Prices are split/dividend adjusted")

    # Step 5: at least 60 trading days of returns either side of day 0.
    counts = [panel.n_days_available(t, d0) for t, d0 in zip(df["ticker"], df["day0"])]
    df["n_days_before"] = [c[0] for c in counts]
    df["n_days_after"] = [c[1] for c in counts]
    enough = (df["n_days_before"] >= MIN_TRADING_DAYS) & (df["n_days_after"] >= MIN_TRADING_DAYS)
    n_short_before = int((df["n_days_before"] < MIN_TRADING_DAYS).sum())
    n_short_after = int((df["n_days_after"] < MIN_TRADING_DAYS).sum())
    df = wf.apply(df, enough,
                  "5. Require 60 trading days of returns before and after day 0",
                  f"{n_short_before} short on the pre-period (recent IPOs), "
                  f"{n_short_after} short on the post-period (price file ends 2026-03-30)")

    df = df.reset_index(drop=True)
    return {
        "sample": df,
        "table1": wf.to_frame(),
        "calendar": calendar,
        "panel": panel,
        "inputs": d,
    }
