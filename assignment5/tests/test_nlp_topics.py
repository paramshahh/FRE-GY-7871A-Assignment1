import numpy as np
import pandas as pd
import pytest

from src import nlp_topics as nt
from src.data_loader import trading_calendar


def _tw(i, text, followers=500, views=1000, rt=False, author=None, ts="2026-09-10 14:00+00:00"):
    return {"id": str(i), "text": text, "followers": followers, "views": views,
            "is_retweet": rt, "author": author or f"u{i}", "lang": "en", "source": "twitter",
            "created_at": pd.Timestamp(ts)}


def test_build_query_has_filters_and_window():
    q = nt.build_query("(midterms)", pd.Timestamp("2026-09-10", tz="UTC"),
                       pd.Timestamp("2026-09-10 06:00", tz="UTC"))
    assert "-is:retweet" in q and "lang:en" in q
    assert "since:2026-09-10_00:00:00_UTC" in q and "until:2026-09-10_06:00:00_UTC" in q


def test_parse_tweets_schema():
    raw = [{"id": "1", "text": "RT @x: the border", "createdAt": "Thu Sep 10 14:00:00 +0000 2026",
            "viewCount": 250, "likeCount": 3, "isReply": False, "lang": "en",
            "author": {"userName": "a", "followers": 42}}]
    df = nt.parse_tweets(raw)
    assert df.loc[0, "is_retweet"] and df.loc[0, "followers"] == 42
    assert df.loc[0, "created_at"] == pd.Timestamp("2026-09-10 14:00", tz="UTC")


def test_quality_filter_removes_bots_retweets_and_astroturf():
    rows = [_tw(1, "grocery prices keep rising every single week here"),
            _tw(2, "rt copy of grocery prices keep rising", rt=True),
            _tw(3, "medicaid cuts will close rural hospitals soon", followers=3),
            _tw(4, "the border policy debate tonight was intense", views=20)]
    rows += [_tw(10 + i, "share this now huge news about the border #vote2026") for i in range(6)]
    rows.append({"id": "n1", "text": "Midterm polls tighten as voters weigh inflation",
                 "source": "gdelt", "lang": "en",
                 "created_at": pd.Timestamp("2026-09-10", tz="UTC")})
    out, rep = nt.quality_filter(pd.DataFrame(rows))
    assert set(out["id"]) == {"1", "n1"}
    tab = rep.as_frame()
    assert tab["removed"].sum() == len(rows) - 2


def test_tone_formula_bounds_and_values():
    cal = trading_calendar("2026-09-08", "2026-09-11")
    d = pd.DataFrame({"created_at": pd.to_datetime(["2026-09-08 14:00+00:00"] * 4),
                      "topic": ["Healthcare"] * 3 + ["Other"],
                      "polarity": [1, -1, -1, 1], "anxiety": 0, "anger": 0, "optimism": 0})
    daily = nt.daily_aggregate(d, cal)
    assert daily.tone.loc["2026-09-08", "Healthcare"] == pytest.approx(-1 / 3)
    assert daily.salience.loc["2026-09-08", "Healthcare"] == pytest.approx(0.75)
    assert np.isnan(daily.tone.loc["2026-09-09", "Healthcare"])
    assert daily.tone.abs().max().max() <= 1


def test_topic_conditioning_flips_generic_reading():
    sc = nt.ToneScorer()
    text = "grocery prices are rising and rent is higher again"
    assert sc.compound(text, "Economy/Inflation") < sc.compound(text) - 0.2
    assert sc.polarity(sc.compound(text, "Economy/Inflation")) == -1


def test_nmf_recovers_seeded_topics():
    rng = np.random.default_rng(0)
    bank = {"Economy/Inflation": "inflation prices groceries rent wages economy",
            "Immigration": "border immigration migrants asylum deportation visa",
            "Healthcare": "medicare medicaid insurance premiums hospital coverage"}
    texts, truth = [], []
    for k, words in bank.items():
        w = words.split()
        for _ in range(80):
            texts.append(" ".join(rng.choice(w, 6)) + " today voters")
            truth.append(k)
    tm = nt.fit_topics(pd.Series(texts), method="nmf", n_components=4)
    acc = (tm.labels.values == np.array(truth)).mean()
    assert acc > 0.9
    assert 0 < nt.topic_diversity(tm.top_terms) <= 1


def test_label_components_threshold():
    labs = nt.label_components([["border", "migrants", "asylum"], ["rally", "tonight", "turnout"]])
    assert labs == ["Immigration", "Other"]


def test_gdelt_parse():
    df = nt.parse_gdelt_artlist({"articles": [{"url": "u", "title": "Midterms heat up",
                                               "seendate": "20260910T123000Z", "domain": "x.com"}]})
    assert df.loc[0, "created_at"] == pd.Timestamp("2026-09-10 12:30", tz="UTC")


def test_gdelt_utc_days_roll_to_next_session_and_drop_outages():
    cal = trading_calendar("2026-09-08", "2026-09-18")
    idx = pd.date_range("2026-09-07", "2026-09-17", freq="D")
    norm = pd.Series(100000.0, index=idx)
    norm.loc["2026-09-15"] = 500.0                      # feed outage
    tl = pd.DataFrame({"count_All": 100.0, "count_Healthcare": 10.0, "tone_All": -1.0,
                       "tone_Healthcare": -2.0, "norm": norm}, index=idx)
    out = nt.gdelt_to_trading_days(tl, cal)
    # Tue 8 Sep (UTC) ends after the close -> Wed 9 Sep; Fri+Sat+Sun -> Mon 14 Sep
    assert out.loc["2026-09-09", "count_All"] == 100
    assert out.loc["2026-09-14", "count_All"] == 300
    assert out.loc["2026-09-14", "sal_Healthcare"] == pytest.approx(0.1)
    assert np.isnan(out.loc["2026-09-16", "count_All"])   # outage day (15th) -> NaN
    assert nt.gdelt_outages(norm).sum() == 1


def test_session_windows_close_to_close_and_weekend_rolls_to_monday():
    cal = trading_calendar("2026-09-10", "2026-09-15")
    w = nt.session_windows(cal)
    mon = [x for x in w if x[0] == pd.Timestamp("2026-09-14")][0]
    assert mon[1] == pd.Timestamp("2026-09-11 20:00", tz="UTC")   # Fri 16:00 EDT
    assert mon[2] == pd.Timestamp("2026-09-14 20:00", tz="UTC")
    assert all(a[2] == b[1] for a, b in zip(w, w[1:]))


def test_ledger_enforces_budget(tmp_path):
    led = nt.CreditLedger(tmp_path / "ledger.csv", budget=600)
    assert led.can_spend(300)
    led.record("d1", 20)
    assert led.spent() == 300 and led.can_spend(300) and not led.can_spend(301)
    led.record("d2", 0)          # an empty page still costs the 15-credit minimum
    assert led.spent() == 315


def test_query_length_guard():
    lo, hi = pd.Timestamp("2026-09-10", tz="UTC"), pd.Timestamp("2026-09-11", tz="UTC")
    with pytest.raises(ValueError):
        nt.build_query("(" + " OR ".join(["inflation"] * 60) + ")", lo, hi)


def test_ledger_counts_attempts(tmp_path):
    led = nt.CreditLedger(tmp_path / "ledger.csv")
    led.record("2026-08-14_p1", 0)
    led.record("2026-08-14_p1", 0)
    assert led.attempts("2026-08-14_p1") == 2 and led.attempts("other") == 0


def test_gdelt_missing_days_are_outages():
    cal = trading_calendar("2026-09-08", "2026-09-18")
    idx = pd.date_range("2026-09-07", "2026-09-17", freq="D").drop(pd.Timestamp("2026-09-15"))
    tl = pd.DataFrame({"count_All": 100.0, "tone_All": -1.0, "norm": 1e5}, index=idx)
    out = nt.gdelt_to_trading_days(tl, cal)
    assert np.isnan(out.loc["2026-09-16", "count_All"])   # the 15th is absent -> masked
    assert out.loc["2026-09-16", "outage_days"] == 1
    assert out.loc["2026-09-17", "count_All"] == 100


def test_gdelt_outage_at_sample_end_is_caught():
    norm = pd.Series([1e5] * 30 + [5e4, 4000, 678], index=pd.date_range("2026-09-01", periods=33))
    out = nt.gdelt_outages(norm)
    assert out.iloc[-1] and out.iloc[-2] and not out.iloc[-3] and out.iloc[:30].sum() == 0


def test_gdelt_partial_cache_count_without_tone():
    cal = trading_calendar("2026-09-08", "2026-09-18")
    idx = pd.date_range("2026-09-07", "2026-09-17", freq="D")
    tl = pd.DataFrame({"count_All": 100.0, "tone_All": -1.0, "count_Healthcare": 9.0,
                       "norm": 1e5}, index=idx)
    out = nt.gdelt_to_trading_days(tl, cal)
    assert "tone_Healthcare" not in out and out["sal_Healthcare"].dropna().eq(0.09).all()
