"""Download (or reuse) every data source, run the quality checks, write DATA_AUDIT.md.

    python audit_data.py              # refresh downloads first
    python audit_data.py --no-refresh # audit the cache in data/raw/
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src import config
from src import data_quality as dq
from src.data_loader import _read_cached_ohlc, download_stooq, load_market_panel
from src.prediction_markets import load_prediction_markets
from src.table_generator import to_markdown

OUT = Path(__file__).resolve().parent / "DATA_AUDIT.md"


def section(title: str, df: pd.DataFrame, note: str = "", digits: int = 4) -> str:
    body = to_markdown(df, digits) if len(df) else "_None._"
    return f"### {title}\n\n{note + chr(10) + chr(10) if note else ''}{body}\n\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    refresh = not args.no_refresh

    mkt = load_market_panel(refresh=refresh)
    pm = load_prediction_markets(refresh=refresh)
    ohlc = _read_cached_ohlc(config.START, config.END)
    fred = pd.read_csv(config.RAW_DIR / "fred.csv", index_col=0, parse_dates=True)
    try:
        stooq = download_stooq(config.SECTOR_ETFS + [config.BENCHMARK], config.START, config.END)
        xsrc = dq.cross_source_returns(ohlc, stooq)
    except Exception as exc:
        xsrc = pd.DataFrame({"error": [repr(exc)]})

    agree, gaps = dq.venue_agreement(pm.by_venue)
    pi_agree, _ = dq.venue_agreement(pm.by_venue, "polymarket", "predictit")
    md = [f"# Data audit\n\nSample {config.START} to {config.END}; "
          f"{len(mkt.returns)} NYSE trading days. Market data source: **{mkt.source}**.\n\n",
          "## 1. Equity, VIX and Treasury data\n\n",
          section("Alignment to the NYSE calendar", mkt.alignment.fillna(0).astype(int)),
          section("OHLC consistency violations", dq.ohlc_violations(ohlc)),
          section("Return outliers (> 5 robust SD)",
                  dq.return_outliers(pd.concat([mkt.returns, mkt.excess.add_suffix("_ex")],
                                               axis=1)),
                  "Each is checked against the second vendor below before being trusted."),
          section("Consecutive unchanged closes", dq.stale_prices(mkt.close)),
          section("Yahoo (adjusted) vs Stooq (unadjusted) daily returns", xsrc,
                  "Differences on ex-dividend days are expected; anything else is a feed error."),
          section("FRED yields", dq.fred_checks(fred.loc[config.START:config.END])),
          "## 2. Prediction markets\n\n",
          section("Coverage (trading days with a quote at or before 16:00 ET)", pm.coverage),
          section("Quote staleness at the 16:00 ET close", dq.quote_staleness(pm.by_venue)),
          section("Polymarket vs Kalshi agreement", agree),
          section("Days the two venues differ by more than 5pp", gaps),
          section("Polymarket vs PredictIt agreement (PredictIt is daily, cross-check only)",
                  pi_agree),
          section("Consensus moves larger than 4pp, by venue", dq.large_moves(
              pm.by_venue[pm.by_venue["venue"] != "predictit"], pm.consensus)),
          section("Consensus P(Dem) summary", pm.consensus.describe().T),
          section("Errors on this run", pd.DataFrame({"error": pm.errors}) if pm.errors
                  else pd.DataFrame())]
    OUT.write_text("".join(md), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
