"""Estimator checks on simulated processes with known parameters.

These simulations exist only to verify that the tests have size and power;
nothing here feeds the analysis.
"""
import numpy as np
import pandas as pd
import pytest

from src import granger_econometrics as ge
from src import table_generator as tg


def _pair(n=400, b=0.4, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.2 * y[t - 1] + b * x[t - 1] + rng.normal()
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.Series(x, idx, name="x"), pd.Series(y, idx, name="y")


def test_granger_detects_planted_effect_and_not_reverse():
    x, y = _pair()
    fwd = ge.granger_f(y, x, lags=2)
    rev = ge.granger_f(x, y, lags=2)
    assert fwd.p_value < 1e-6 and fwd.sum_gamma > 0.25
    assert rev.p_value > 0.01


def test_granger_size_under_null():
    rejections = 0
    for s in range(200):
        x, y = _pair(n=150, b=0.0, seed=s)
        rejections += ge.granger_f(y, x, lags=2, cov="HC1").p_value < 0.05
    assert rejections / 200 < 0.10


def test_granger_matches_statsmodels_var():
    x, y = _pair()
    df = pd.concat([y, x], axis=1)
    ours = ge.granger_f(y, x, lags=2)
    var = ge.var_causality(df, "y", "x", 2)
    assert ours.F == pytest.approx(var["F"], rel=0.02)


def test_benjamini_hochberg():
    q = ge.benjamini_hochberg(pd.Series([0.01, 0.04, 0.03, 0.5, np.nan]))
    assert q.iloc[0] == pytest.approx(0.04)
    assert q.iloc[1] == pytest.approx(0.16 / 3) and q.iloc[2] == pytest.approx(0.16 / 3)
    assert np.isnan(q.iloc[4])


def test_rigobon_sack_recovers_beta_where_ols_is_biased():
    rng = np.random.default_rng(3)
    n, beta, alpha = 4000, 0.5, 0.6
    h = rng.random(n) < 0.3
    eps = rng.normal(0, np.where(h, 3.0, 1.0))      # dProb structural shock, heteroskedastic
    eta = rng.normal(0, 1.0, n)                      # return shock, homoskedastic
    x = (eps + alpha * eta) / (1 - alpha * beta)     # dp = a*r + eps ; r = b*dp + eta
    y = beta * x + eta
    idx = pd.RangeIndex(n)
    r = ge.rigobon_sack(pd.Series(x, idx), pd.Series(y, idx), pd.Series(h.astype(int), idx),
                        n_boot=200)
    assert abs(r.beta_ols - beta) > 0.1
    assert r.beta_rs == pytest.approx(beta, abs=0.08)
    assert r.ci_low < beta < r.ci_high and r.var_ratio_x > 3 and r.var_ratio_p < 0.01
    iv = ge.rigobon_sack_iv(pd.Series(x, idx), pd.Series(y, idx), pd.Series(h.astype(int), idx))
    assert iv == pytest.approx(r.beta_rs, rel=1e-6)


def test_lag_selection_and_stationarity():
    x, y = _pair()
    tab, chosen = ge.select_lag(pd.concat([y, x], axis=1), maxlags=4)
    assert {"aic", "bic"} <= set(chosen) and chosen["bic"] >= 1 and len(tab) == 5
    rw = pd.Series(np.cumsum(np.random.default_rng(0).normal(size=300)), name="rw")
    out, diffed = ge.make_stationary(pd.concat([rw, x.reset_index(drop=True)], axis=1))
    assert diffed["rw"] and not diffed["x"]


def test_granger_matrix_has_q_values_and_tables_render():
    x, y = _pair()
    m = ge.granger_matrix(pd.DataFrame({"tone_Healthcare": x}), pd.DataFrame({"dProb_House": y}),
                          lags=2)
    assert {"q_value", "F", "p_value"} <= set(m.columns)
    t2 = tg.table2_stage1(m)
    md = tg.to_markdown(t2)
    tex = tg.to_latex(t2, "Table 2", "tab:2", notes="x_y & z")
    assert "Healthcare" in md and r"\toprule" in tex and r"x\_y \& z" in tex


def test_high_news_days_and_event_study():
    idx = pd.bdate_range("2026-01-01", periods=60)
    vol = pd.Series(np.arange(60), idx)
    h = ge.high_news_days(vol, 0.8, event_days=["2026-01-02"])
    assert h.sum() == 12 + 1
    ex = pd.DataFrame({"XLE": np.r_[np.zeros(30), 0.01, np.zeros(29)]}, idx)
    dp = pd.Series(np.r_[np.zeros(30), 0.05, np.zeros(29)], idx)
    es = ge.event_study(ex, dp, n_events=1, window=3)
    assert es.loc[0, "XLE"] == pytest.approx(0.01) and es.loc[-1, "XLE"] == pytest.approx(0)


def test_factor_oos_uses_only_post_training_dates():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2025-01-01", periods=300)
    f = pd.DataFrame({"F1": rng.normal(size=300)}, idx)
    y = pd.DataFrame({"a": 0.5 * f["F1"].shift(1) + rng.normal(size=300),
                      "b": -0.5 * f["F1"].shift(1) + rng.normal(size=300)}, idx)
    V = pd.DataFrame({"F1": [0.7071, -0.7071]}, index=["a", "b"])
    out = ge.factor_oos(f, V, y, idx[179])
    assert out.loc[0, "oos_days"] == 120 and out.loc[0, "oos_corr"] > 0.3
    assert out.loc[0, "targets_right_sign"] == 1.0
    # destroy the post-training relation: the OOS score must collapse
    y.loc[idx[180]:] = rng.normal(size=(120, 2))
    assert abs(ge.factor_oos(f, V, y, idx[179]).loc[0, "oos_corr"]) < 0.2


def test_robustness_battery_separates_real_from_outlier_driven():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2026-01-01", periods=300)
    x = pd.Series(rng.normal(size=300), idx, name="dProb")
    y = pd.Series(0.35 * x.shift(1).fillna(0) + rng.normal(size=300), idx, name="R")
    assert ge.granger_robustness(y, x)["robust"]
    # a relation carried entirely by two jump days must fail the battery
    x2 = pd.Series(rng.normal(size=300), idx, name="dProb")
    y2 = pd.Series(rng.normal(size=300), idx, name="R")
    for d in (50, 200):
        x2.iloc[d] = 12.0
        y2.iloc[d + 1] = 15.0
    r = ge.granger_robustness(y2, x2)
    assert r["p"] < 0.05 and not r["robust"]
