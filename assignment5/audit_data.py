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
from src.data_loader import (_read_cached_ohlc, download_fred, download_yahoo,
                             load_market_panel)
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
    sib_tickers = sorted({s for v in config.SIBLINGS.values() for s in v} - set(ohlc))
    try:
        sibs = download_yahoo(sib_tickers, config.START, config.END)
        sib_summary, sib_flags = dq.sibling_divergence(ohlc, sibs, config.SIBLINGS)
    except Exception as exc:
        sib_summary, sib_flags = pd.DataFrame({"error": [repr(exc)]}), pd.DataFrame()
    try:
        sp = download_fred(["SP500"], config.START, config.END)
        idx_check, idx_flags = dq.sibling_divergence(
            {"SPY": ohlc["SPY"]}, {"SP500": sp.rename(columns={"SP500": "Close"})},
            {"SPY": ["SP500"]})
    except Exception as exc:
        idx_check, idx_flags = pd.DataFrame({"error": [repr(exc)]}), pd.DataFrame()

    from src import nlp_topics as nt
    from src.data_loader import trading_calendar

    cal = trading_calendar(config.START, config.END)
    tweets = nt.load_cached_tweets()
    tweet_cov = dq.tweet_coverage(tweets, cal) if len(tweets) else pd.DataFrame()
    tw_filter = nt.quality_filter(tweets)[1].as_frame() if len(tweets) else pd.DataFrame()
    led_path = config.RAW_DIR / "tweets" / "ledger.csv"
    led = pd.read_csv(led_path) if led_path.exists() else pd.DataFrame({"credits": []})
    ledger_calls, ledger_credits = len(led), int(led["credits"].sum())
    gd_summary, gd_flags = dq.gdelt_coverage(
        nt.collect_gdelt_timelines(config.START, config.END, fetch=False), config.START, config.END)

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
          section("Same-exposure funds from other issuers (second-source check)", sib_summary,
                  "Stooq blocked repeated downloads, so each fund is triangulated against funds "
                  "with the same exposure from other issuers instead."),
          section("Days a fund diverges from its sibling by > 6 robust SD", sib_flags),
          section("SPY vs the S&P 500 index (FRED SP500)", idx_check,
                  "25 Jun - 1 Jul 2026: the index (FRED and Yahoo ^GSPC, one upstream source) "
                  "and the three S&P 500 trackers SPY, IVV, VOO disagree by 15-30bp a day, "
                  "while agreeing on the week's total (+155 to +170bp). Daily closes that week "
                  "are mistimed in at least one feed. Sensitivity: replacing SPY with the "
                  "median of SPY/IVV/VOO (mean change 0.9bp) or dropping 24 Jun - 2 Jul leaves "
                  "every Stage 2 result's sign and significance unchanged (see the report)."),
          section("SPY vs index divergence days", idx_flags),
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
          "## 3. Text\n\n",
          section("Tweets by stratum (twitterapi.io)", tweet_cov,
                  f"Paid calls in the ledger: {ledger_calls} for {ledger_credits:,} credits."),
          section("Tweet quality-filter waterfall", tw_filter),
          section("GDELT DOC timelines", gd_summary),
          section("GDELT outage days (masked, never read as quiet news days)", gd_flags),
          section("Errors on this run", pd.DataFrame({"error": pm.errors}) if pm.errors
                  else pd.DataFrame())]
    OUT.write_text("".join(md), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
