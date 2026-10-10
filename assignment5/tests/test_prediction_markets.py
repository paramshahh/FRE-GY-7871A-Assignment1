import json

import numpy as np
import pandas as pd
import pytest

from src import prediction_markets as pm
from src.data_loader import trading_calendar


def test_party_and_chamber_parsing():
    assert pm.party_of("Democratic Party") == "D"
    assert pm.party_of("Will the Republican Party control the House?") == "R"
    assert pm.party_of("GOP") == "R"
    assert pm.party_of("Independent") is None
    assert pm.chamber_of("control the Senate after") == "Senate"


def test_remove_overround():
    p = pm.remove_overround({"D": 0.66, "R": 0.40})
    assert sum(p.values()) == pytest.approx(1.0) and p["D"] == pytest.approx(0.66 / 1.06)


def test_polymarket_token_lookup_from_search_payload():
    q = "Will the {} Party control the House after the 2026 Midterm elections?"
    payload = {"events": [{"markets": [
        {"question": q.format("Democratic"), "outcomes": json.dumps(["Yes", "No"]),
         "clobTokenIds": json.dumps(["111", "112"])},
        {"question": q.format("Republican"), "outcomes": '["Yes", "No"]',
         "clobTokenIds": '["221", "222"]'},
        {"question": "Will the Democratic Party control the Senate after the 2026 Midterm "
                     "elections?", "outcomes": '["Yes","No"]', "clobTokenIds": '["331","332"]'},
    ]}]}
    assert pm.find_polymarket_tokens(payload, "House") == {"D": "111", "R": "221"}
    assert pm.find_polymarket_tokens(payload, "Senate") == {"D": "331"}


def test_polymarket_history_and_two_party():
    d = pm.parse_polymarket_history({"history": [{"t": 1759700000, "p": 0.90},
                                                 {"t": 1759786400, "p": 0.92}]})
    r = pm.parse_polymarket_history({"history": [{"t": 1759700000, "p": 0.12},
                                                 {"t": 1759786400, "p": 0.10}]})
    p = pm._two_party_prob({"D": d, "R": r})
    assert p.iloc[0] == pytest.approx(0.90 / 1.02)


def test_kalshi_candles_cents_dollars_and_midpoint():
    payload = {"candlesticks": [
        {"end_period_ts": 1759700000, "price": {"close": 91}},
        {"end_period_ts": 1759786400, "price": {"close_dollars": "0.9200"}},
        {"end_period_ts": 1759872800, "price": {}, "yes_bid": {"close": 90},
         "yes_ask": {"close": 94}},
    ]}
    s = pm.parse_kalshi_candles(payload)
    assert s.tolist() == pytest.approx([0.91, 0.92, 0.92])
    assert s.index[0] == pd.Timestamp(1759700000 - 1, unit="s", tz="UTC")
    wide = pm.parse_kalshi_candles({"candlesticks": [
        {"end_period_ts": 1759700000, "price": {"close_dollars": "0.50"},
         "yes_bid": {"close_dollars": "0.60"}, "yes_ask": {"close_dollars": "0.62"}},
        {"end_period_ts": 1759703600, "price": {"close_dollars": "0.50"},
         "yes_bid": {"close_dollars": "0.40"}, "yes_ask": {"close_dollars": "0.70"}}]})
    assert wide.tolist() == pytest.approx([0.61, 0.50])   # midpoint unless spread too wide


def test_predictit_snapshot_and_csv(tmp_path):
    snap = {"id": 8157, "contracts": [{"name": "Democratic", "lastTradePrice": 0.84},
                                      {"name": "Republican", "lastTradePrice": 0.19}]}
    assert pm.parse_predictit_snapshot(snap) == {"D": 0.84, "R": 0.19}
    f = tmp_path / "house_export.csv"
    f.write_text("ContractName,Date,OpenSharePrice,HighSharePrice,LowSharePrice,"
                 "CloseSharePrice,TradeVolume\n"
                 "Democratic,10/1/2026 12:00:00 AM,$0.83,$0.85,$0.82,$0.84,10\n"
                 "Republican,10/1/2026 12:00:00 AM,$0.18,$0.19,$0.16,$0.17,3\n")
    df = pm.parse_predictit_csv(f)
    assert set(df["party"]) == {"D", "R"} and df["price"].max() == pytest.approx(0.84)


def test_to_daily_uses_last_quote_before_close():
    cal = trading_calendar("2026-10-05", "2026-10-09")
    s = pd.Series([0.50, 0.55, 0.60], index=pd.to_datetime(
        ["2026-10-06 14:00+00:00", "2026-10-06 19:00+00:00", "2026-10-06 21:00+00:00"]))
    d = pm.to_daily(s, cal)
    assert d.loc["2026-10-06", "prob_dem"] == 0.55 and d.loc["2026-10-07", "prob_dem"] == 0.60
    assert d.loc["2026-10-06", "quote_ts"] == pd.Timestamp("2026-10-06 19:00", tz="UTC")
    assert np.isnan(d.loc["2026-10-05", "prob_dem"])


def test_windows_cover_range_without_gaps():
    w = pm._windows("2026-03-02", "2026-03-31", 14)
    assert all(a[1] == b[0] for a, b in zip(w, w[1:]))
    assert w[-1][1] - w[0][0] == 30 * 86400 - 3600   # ET midnights; DST starts 8 Mar
    assert max(b - a for a, b in w) <= 14 * 86400


def test_consensus_renormalises_over_reporting_venues():
    long = pd.DataFrame({"date": pd.to_datetime(["2026-10-06"] * 2 + ["2026-10-07"]),
                         "venue": ["polymarket", "kalshi", "polymarket"],
                         "chamber": "House", "prob_dem": [0.90, 0.80, 0.70]})
    c = pm.consensus(long, {"polymarket": 0.5, "kalshi": 0.5})
    assert c.loc["2026-10-06", "House"] == pytest.approx(0.85)
    assert c.loc["2026-10-07", "House"] == pytest.approx(0.70)


def test_sweep_keys_and_normalisation():
    assert pm.sweep_key("2026 Balance of Power: D Senate, D House") == "DD"
    assert pm.sweep_key("2026 Balance of Power: R Senate, D House") == "DR"
    assert pm.sweep_key("D-House, R-Senate") == "DR"
    assert pm.sweep_key("R-House, D-Senate") == "RD"
    assert pm.sweep_key("Democrats sweep") == "DD" and pm.sweep_key("Other") is None
    idx = pd.to_datetime(["2026-10-01 15:00+00:00"])
    p = pm._target_prob({k: pd.Series([v], idx) for k, v in
                         {"DD": 0.61, "DR": 0.29, "RD": 0.01, "RR": 0.10}.items()}, "DD")
    assert p.iloc[0] == pytest.approx(0.61 / 1.01)
