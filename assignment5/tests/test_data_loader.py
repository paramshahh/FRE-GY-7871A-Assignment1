import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl


def test_calendar_excludes_nyse_holidays_and_weekends():
    cal = dl.trading_calendar("2026-06-15", "2026-07-10")
    assert pd.Timestamp("2026-06-19") not in cal   # Juneteenth (Friday)
    assert pd.Timestamp("2026-07-03") not in cal   # Independence Day observed (4 Jul is Sat)
    assert pd.Timestamp("2026-06-20") not in cal   # Saturday
    assert pd.Timestamp("2026-06-18") in cal
    assert pd.Timestamp("2026-04-03") not in dl.trading_calendar("2026-04-01", "2026-04-07")


def test_after_close_and_weekend_roll_forward():
    cal = dl.trading_calendar("2026-09-01", "2026-09-30")
    ts = pd.Series(pd.to_datetime([
        "2026-09-10 19:59:00+00:00",   # 15:59 ET Thursday -> Thursday
        "2026-09-10 20:00:00+00:00",   # 16:00 ET Thursday -> Friday
        "2026-09-12 15:00:00+00:00",   # Saturday -> Monday 14th
        "2026-09-07 15:00:00+00:00",   # Labor Day -> Tuesday 8th
    ]))
    out = dl.to_trading_day(ts, cal)
    assert list(out.dt.strftime("%Y-%m-%d")) == ["2026-09-10", "2026-09-11", "2026-09-14",
                                                 "2026-09-08"]


def test_after_end_of_calendar_is_nat():
    cal = dl.trading_calendar("2026-09-01", "2026-09-04")
    out = dl.to_trading_day(pd.Series(pd.to_datetime(["2026-09-04 21:00:00+00:00"])), cal)
    assert out.isna().all()


def test_returns_do_not_span_missing_days():
    cal = dl.trading_calendar("2026-09-01", "2026-09-10")
    close = pd.DataFrame({"A": [100, 101, 102, 103, 104, 105, 106]}, index=cal[:7])
    close = close.drop(index=cal[3])
    al = dl.align_to_calendar(close, cal[:7])
    r = dl.log_returns(al)
    assert np.isnan(r["A"].iloc[3]) and np.isnan(r["A"].iloc[4])
    assert r["A"].iloc[5] == pytest.approx(np.log(105 / 104))


def test_ffill_is_capped_and_reported():
    cal = dl.trading_calendar("2026-09-01", "2026-09-15")
    s = pd.DataFrame({"DGS2": [3.5] + [np.nan] * 5 + [3.6] * (len(cal) - 6)}, index=cal)
    rep = dl.AlignmentReport()
    out = dl.align_to_calendar(s, cal, max_ffill=3, report=rep)
    assert rep.missing["DGS2"] == 5 and rep.filled["DGS2"] == 3
    assert out["DGS2"].isna().sum() == 2


def test_excess_returns_and_range_vol():
    r = pd.DataFrame({"SPY": [0.01, -0.02], "XLE": [0.03, -0.01]})
    ex = dl.excess_returns(r)
    assert list(ex.columns) == ["XLE"]
    assert ex["XLE"].tolist() == pytest.approx([0.02, 0.01])
    v = dl.range_volatility(pd.DataFrame({"X": [11.0]}), pd.DataFrame({"X": [9.0]}))
    assert v["X"].iloc[0] == pytest.approx(0.2)


def test_parse_fred_csv_handles_dots():
    txt = "observation_date,DGS2\n2026-10-08,3.51\n2026-10-09,.\n"
    df = dl.parse_fred_csv(txt, "DGS2")
    assert df["DGS2"].iloc[0] == 3.51 and np.isnan(df["DGS2"].iloc[1])


def test_clean_index_converts_tz_and_dedupes():
    idx = pd.to_datetime(["2026-09-01 13:30", "2026-09-01 20:00", "2026-09-02 13:30"]).tz_localize(
        "UTC")
    df = dl.clean_index(pd.DataFrame({"x": [1, 2, 3]}, index=idx))
    assert df.index.tz is None and len(df) == 2 and df["x"].iloc[0] == 2
