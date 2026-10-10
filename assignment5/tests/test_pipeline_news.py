"""News-salience check on a toy panel (fixture data only)."""
import numpy as np
import pandas as pd

from src import nlp_svd_topics as lsa
from src.data_loader import trading_calendar
from src.pipeline import news_salience_check


def test_news_check_handles_partial_gdelt_and_validates_against_x():
    rng = np.random.default_rng(0)
    cal = trading_calendar("2026-03-02", "2026-08-31")
    n = len(cal)
    agenda = pd.Series(np.convolve(rng.normal(size=n + 9), np.ones(10) / 10, "valid"), cal)
    news = pd.DataFrame({"count_All": 500.0, "sal_Healthcare": 0.2 + 0.05 * agenda,
                         "count_Healthcare": 100.0, "tone_Healthcare": rng.normal(size=n),
                         "sal_Immigration": 0.1 + 0.02 * rng.normal(size=n)}, index=cal)
    news.iloc[40:45] = np.nan                                   # an outage week
    sal = pd.DataFrame({"Healthcare": 0.3 + 0.05 * agenda + 0.01 * rng.normal(size=n),
                        "Immigration": 0.2 + 0.02 * rng.normal(size=n)}, index=cal)
    daily = lsa.DailyLSA(None, sal, None, None, None, None, None, {}, None, None)
    dprob = pd.DataFrame({c: rng.normal(0, 0.01, n) for c in ("House", "Senate", "Sweep")},
                         index=cal)
    ctrl = pd.DataFrame({"SPY_ret": rng.normal(0, 0.01, n)}, index=cal)
    tests, val, status = news_salience_check(news, daily, dprob, ctrl, 1, "HC1")
    assert status.loc["Healthcare"].all() and not status.loc["Economy/Inflation"].any()
    assert set(tests["cause"]) == {"news_sal_Healthcare", "news_sal_Immigration",
                                   "news_tone_Healthcare"}
    assert len(tests) == 9 and tests["q_value"].notna().all()
    v = val.set_index("issue")
    assert v.loc["Healthcare", "corr_levels"] > 0.8          # X tracks the shared agenda
    assert abs(v.loc["Immigration", "corr_levels"]) < 0.4    # independent noise does not
