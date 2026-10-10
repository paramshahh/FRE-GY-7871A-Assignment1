"""Stationarity, lag selection, Granger F-tests, VAR causality and Rigobon-Sack.

Stage 1:  dProb_t = a + sum_j b_j dProb_{t-j} + sum_j g_j Tone_{k,t-j} + controls + e_t
Stage 2:  R^ex_{s,t} (or Vol_{s,t}) on its own lags and lags of dProb_t

Every test is a single-equation OLS F-test of H0: all g_j = 0 (the Granger
restriction), optionally with heteroskedasticity-robust (HC1) or HAC Wald
statistics, plus the joint VAR test from statsmodels for comparison. A table
of tests is corrected for multiple comparisons with Benjamini-Hochberg.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

from . import config


# ---------------------------------------------------------------------------
# Stationarity and lag order
# ---------------------------------------------------------------------------
def adf_table(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        x = df[c].dropna()
        if len(x) < 20 or x.std() == 0:
            rows.append({"series": c, "adf_stat": np.nan, "p_value": np.nan, "stationary": False})
            continue
        stat, p, *_ = adfuller(x, autolag="AIC", result_object=False)
        rows.append({"series": c, "adf_stat": stat, "p_value": p, "stationary": p < alpha})
    return pd.DataFrame(rows).set_index("series")


def make_stationary(df: pd.DataFrame, alpha: float = 0.05) -> tuple[pd.DataFrame, pd.Series]:
    """First-difference any column whose ADF test does not reject a unit root."""
    tab = adf_table(df, alpha)
    out = df.copy()
    diffed = ~tab["stationary"]
    for c in tab.index[diffed]:
        out[c] = df[c].diff()
    return out, diffed.rename("differenced")


def select_lag(df: pd.DataFrame, maxlags: int = config.MAX_LAGS) -> tuple[pd.DataFrame, dict]:
    """Information criteria for VAR(1..maxlags); returns the table and the chosen orders."""
    data = df.dropna()
    maxlags = int(min(maxlags, max(1, (len(data) - 10) // (data.shape[1] + 1) - 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = VAR(data).select_order(maxlags)
    tab = pd.DataFrame({ic: [sel.ics[ic][p] for p in range(maxlags + 1)]
                        for ic in ("aic", "bic", "hqic", "fpe")})
    tab.index.name = "lags"
    chosen = {k: max(1, int(v)) for k, v in sel.selected_orders.items()}
    return tab, chosen


# ---------------------------------------------------------------------------
# Granger F-test
# ---------------------------------------------------------------------------
def lag_matrix(s: pd.Series, lags: int, name: str) -> pd.DataFrame:
    return pd.concat({f"{name}_L{j}": s.shift(j) for j in range(1, lags + 1)}, axis=1)


@dataclass
class GrangerResult:
    cause: str
    effect: str
    lags: int
    F: float
    p_value: float
    df_num: int
    df_den: int
    nobs: int
    sum_gamma: float
    r2_gain: float
    cov: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def granger_f(y: pd.Series, x: pd.Series, lags: int, controls: pd.DataFrame | None = None,
              cov: str = "nonrobust", control_lags: bool = True) -> GrangerResult:
    """F-test that lags 1..p of ``x`` add nothing to an AR(p) for ``y`` (plus controls).

    ``cov`` = 'nonrobust' gives the classical F; 'HC1' or 'HAC' give the Wald
    F with robust covariance. Controls enter lagged (predictive design) unless
    ``control_lags`` is False, in which case they enter contemporaneously.
    """
    yn, xn = y.name or "y", x.name or "x"
    parts = [y.rename("y"), lag_matrix(y, lags, "y"), lag_matrix(x, lags, "x")]
    if controls is not None and controls.shape[1]:
        c = controls.shift(1) if control_lags else controls
        parts.append(c.add_prefix("c_"))
    data = pd.concat(parts, axis=1).dropna()
    xcols = [f"x_L{j}" for j in range(1, lags + 1)]
    X = sm.add_constant(data.drop(columns="y"), has_constant="add")
    fit_kw = {}
    if cov == "HAC":
        fit_kw = {"cov_type": "HAC", "cov_kwds": {"maxlags": lags}}
    elif cov != "nonrobust":
        fit_kw = {"cov_type": cov}
    unres = sm.OLS(data["y"], X).fit(**fit_kw)
    restr = sm.OLS(data["y"], X.drop(columns=xcols)).fit()
    R = np.zeros((lags, X.shape[1]))
    for i, c in enumerate(xcols):
        R[i, list(X.columns).index(c)] = 1.0
    ft = unres.f_test(R)
    return GrangerResult(
        cause=str(xn), effect=str(yn), lags=lags, F=float(np.squeeze(ft.fvalue)),
        p_value=float(np.squeeze(ft.pvalue)), df_num=int(ft.df_num), df_den=int(ft.df_denom),
        nobs=int(unres.nobs), sum_gamma=float(unres.params[xcols].sum()),
        r2_gain=float(unres.rsquared - restr.rsquared), cov=cov)


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    """BH false-discovery-rate adjusted p-values (q-values)."""
    p = p.astype(float)
    ok = p.notna()
    v = p[ok].values
    n = len(v)
    if n == 0:
        return p
    order = np.argsort(v)
    ranked = v[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = pd.Series(np.nan, index=p.index)
    out.loc[ok[ok].index[order]] = q
    return out


def choose_lags(y: pd.Series, x: pd.Series, rule: str | int, maxlags: int) -> int:
    if isinstance(rule, (int, np.integer)):
        return int(rule)
    _, chosen = select_lag(pd.concat([y, x], axis=1), maxlags)
    return chosen.get(rule, 1)


def granger_matrix(causes: pd.DataFrame, effects: pd.DataFrame, lags: str | int = "bic",
                   controls: pd.DataFrame | None = None, cov: str = "HC1",
                   maxlags: int = config.MAX_LAGS) -> pd.DataFrame:
    """Granger test for every (cause, effect) pair, with BH q-values over the whole table."""
    rows = []
    for e in effects.columns:
        for c in causes.columns:
            y, x = effects[e].rename(e), causes[c].rename(c)
            if y.dropna().shape[0] < 30 or x.dropna().std() == 0:
                continue
            p = choose_lags(y, x, lags, maxlags)
            ctrl = controls.drop(columns=[k for k in (e, c) if k in controls], errors="ignore") \
                if controls is not None else None
            rows.append(granger_f(y, x, p, ctrl, cov=cov).as_dict())
    df = pd.DataFrame(rows)
    if len(df):
        df["q_value"] = benjamini_hochberg(df["p_value"])
    return df


def var_causality(df: pd.DataFrame, caused: str, causing: str, lags: int) -> dict:
    """statsmodels joint VAR Granger test (F form) for a two-variable system."""
    data = df[[caused, causing]].dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VAR(data).fit(lags)
        t = res.test_causality(caused, [causing], kind="f")
    return {"F": float(t.test_statistic), "p_value": float(t.pvalue), "lags": lags,
            "nobs": int(res.nobs)}


def var_irf(df: pd.DataFrame, lags: int, horizon: int = 10) -> pd.DataFrame:
    """Orthogonalised impulse responses: response of col 1 to a shock in col 0."""
    data = df.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VAR(data).fit(lags)
        irf = res.irf(horizon)
    resp = irf.orth_irfs[:, 1, 0]
    se = irf.stderr(orth=True)[:, 1, 0]
    return pd.DataFrame({"h": range(horizon + 1), "irf": resp, "lo": resp - 1.96 * se,
                         "hi": resp + 1.96 * se})


# ---------------------------------------------------------------------------
# Rigobon-Sack identification through heteroskedasticity
# ---------------------------------------------------------------------------
@dataclass
class RigobonSackResult:
    beta_ols: float
    beta_rs: float
    ci_low: float
    ci_high: float
    var_ratio_x: float      # Var_H(dProb) / Var_L(dProb): instrument relevance
    var_ratio_p: float      # one-sided F-test p-value of that ratio
    var_ratio_y: float
    n_high: int
    n_low: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _rs_beta(x: np.ndarray, y: np.ndarray, h: np.ndarray) -> float:
    xh, yh, xl, yl = x[h], y[h], x[~h], y[~h]
    dcov = np.cov(xh, yh)[0, 1] - np.cov(xl, yl)[0, 1]
    dvar = np.var(xh, ddof=1) - np.var(xl, ddof=1)
    return dcov / dvar if dvar != 0 else np.nan


def rigobon_sack_iv(x: pd.Series, y: pd.Series, high: pd.Series) -> float:
    """IV form of the estimator: instrument w_t = x_t/T_H on H days and -x_t/T_L on L days.

    Numerically equal to dCov/dVar when both series are demeaned within regime.
    """
    d = pd.concat([x.rename("x"), y.rename("y"), high.rename("h")], axis=1).dropna()
    h = d["h"].astype(bool)
    xd = d["x"] - d.groupby(h)["x"].transform("mean")
    yd = d["y"] - d.groupby(h)["y"].transform("mean")
    nh, nl = h.sum() - 1, (~h).sum() - 1
    w = np.where(h, xd / nh, -xd / nl)
    return float((w * yd).sum() / (w * xd).sum())


def rigobon_sack(x: pd.Series, y: pd.Series, high: pd.Series, n_boot: int = 999,
                 seed: int = config.SEED) -> RigobonSackResult:
    """Effect of ``x`` (dProb) on ``y`` (excess return) identified by a variance shift in x."""
    from scipy import stats

    d = pd.concat([x.rename("x"), y.rename("y"), high.rename("h")], axis=1).dropna()
    xv, yv, h = d["x"].values, d["y"].values, d["h"].astype(bool).values
    beta_ols = float(np.cov(xv, yv)[0, 1] / np.var(xv, ddof=1))
    beta = _rs_beta(xv, yv, h)
    rng = np.random.default_rng(seed)
    ih, il = np.where(h)[0], np.where(~h)[0]
    boots = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(ih, len(ih)), rng.choice(il, len(il))]
        boots.append(_rs_beta(xv[idx], yv[idx], h[idx]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    vh, vl = np.var(xv[h], ddof=1), np.var(xv[~h], ddof=1)
    ratio = vh / vl
    p = 1 - stats.f.cdf(ratio, len(ih) - 1, len(il) - 1)
    return RigobonSackResult(beta_ols, float(beta), float(lo), float(hi), float(ratio),
                             float(p), float(np.var(yv[h], ddof=1) / np.var(yv[~h], ddof=1)),
                             int(h.sum()), int((~h).sum()))


def high_news_days(volume: pd.Series, quantile: float = config.HIGH_NEWS_QUANTILE,
                   event_days: list[str] | None = None) -> pd.Series:
    """H regime: discourse volume above its ``quantile``, plus any scheduled event days."""
    h = (volume > volume.quantile(quantile)).astype(int)
    if event_days:
        h.loc[h.index.isin(pd.to_datetime(event_days))] = 1
    return h.rename("high_news")


# ---------------------------------------------------------------------------
# Event study for the dashboard
# ---------------------------------------------------------------------------
def event_study(excess: pd.DataFrame, dprob: pd.Series, n_events: int = 10,
                window: int = 5) -> pd.DataFrame:
    """Mean cumulative excess return around the largest |dProb| days, signed by the move."""
    dp = dprob.dropna()
    events = dp.abs().nlargest(n_events).index
    rows = []
    idx = excess.index
    for d in events:
        i = idx.get_loc(d)
        sign = np.sign(dp.loc[d])
        lo, hi = max(0, i - window), min(len(idx), i + window + 1)
        seg = excess.iloc[lo:hi].fillna(0) * sign
        cum = seg.cumsum() - seg.cumsum().iloc[i - lo - 1 if i - lo >= 1 else 0]
        cum.index = range(lo - i, hi - i)
        rows.append(cum)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows).groupby(level=0).mean()
    out.index.name = "day"
    return out
