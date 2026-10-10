"""Paid tweet collection from twitterapi.io, within a hard credit budget.

    TWITTERAPI_IO_KEY=... python collect_tweets.py --page 1
    TWITTERAPI_IO_KEY=... python collect_tweets.py --page 2

Every call is written to data/raw/tweets/ledger.csv and every response cached,
so re-running never pays twice. Stops at TWITTER_BUDGET_CREDITS spent or when
the account balance would fall under TWITTER_BALANCE_FLOOR.
"""
import argparse
import logging

from src import config, nlp_topics as nt
from src.data_loader import trading_calendar

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first N sessions")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cal = trading_calendar(config.START, config.END)
    if a.limit:
        cal = cal[:a.limit]
    ledger = nt.CreditLedger()
    before = ledger.spent()
    df = nt.collect_tweets(cal, page=a.page, ledger=ledger)
    print(f"spent this run: {ledger.spent() - before:,} credits; total {ledger.spent():,} of "
          f"{config.TWITTER_BUDGET_CREDITS:,}; cached tweets {len(df):,}; "
          f"account balance {nt.account_balance():,}")
