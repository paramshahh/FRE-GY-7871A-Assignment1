"""End-to-end pipeline shared by the dashboard, the report and the notebook.

data -> LSA topics -> daily topic sentiment / salience -> GDELT news regime ->
covariance SVD -> Stage 1 and Stage 2 Granger tests -> Rigobon-Sack -> tables.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from . import granger_econometrics as ge
from . import nlp_svd_topics as lsa
from . import nlp_topics as nt
from . import table_generator as tg
from .data_loader import MarketPanel, load_market_panel
from .prediction_markets import MarketProbabilities, load_prediction_markets

log = logging.getLogger(__name__)
RESULTS_PATH = config.OUTPUT_DIR / "results.pkl"
CONTROLS = ("SPY_ret", "dVIX", "dDGS2")


@dataclass
class Results:
    market: MarketPanel
    pm: MarketProbabilities
    filter_report: pd.DataFrame
    lsa_model: lsa.LSAModel
    stability: pd.DataFrame
    spectrum: np.ndarray
    model_comparison: pd.DataFrame
    daily: lsa.DailyLSA
    signals: pd.DataFrame          # S_k,t used in the tests (5-session pooled)
    signals_robust: pd.DataFrame   # 3-session pooled robustness version
    signal_coverage: pd.DataFrame
    news: pd.DataFrame             # GDELT on trading days: counts, tone, salience, outages
    high_news: pd.Series           # 1 = H regime, 0 = L, NaN = GDELT outage
    adf: pd.DataFrame
    lag_table: pd.DataFrame
    cov_svd: lsa.CovarianceSVD
    factor_tests: pd.DataFrame
    factor_oos: pd.DataFrame
    stage1: pd.DataFrame
    stage1_robust: pd.DataFrame
    stage1_reverse: pd.DataFrame
    stage1_salience: pd.DataFrame
    stage2_ret: pd.DataFrame
    stage2_vol: pd.DataFrame
    robustness: pd.DataFrame       # battery for every test with q < 0.10
    sentiment_to_asset: pd.DataFrame
    rigobon_sack: pd.DataFrame
    tables: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)


def load_tweets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached tweets (collection is paid and done separately by collect_tweets.py), filtered."""
    raw = nt.load_cached_tweets()
    if raw.empty:
        raise ConnectionError("No cached tweets: run collect_tweets.py first")
    tweets, rep = nt.quality_filter(raw)
    return tweets.reset_index(drop=True), rep.as_frame()


def load_news(calendar: pd.DatetimeIndex, refresh: bool,
              quantile: float) -> tuple[pd.DataFrame, pd.Series]:
    """GDELT timelines on trading days and the high-news regime (NaN on outage sessions)."""
    tl = nt.collect_gdelt_timelines(config.START, config.END, fetch=refresh)
    if tl.empty or "count_All" not in tl:
        raise ConnectionError("No GDELT timelines cached: run the collector first")
    news = nt.gdelt_to_trading_days(tl, calendar)
    vol = news["count_All"].where(news["outage_days"] == 0)
    high = (vol > vol.quantile(quantile)).astype(float).where(vol.notna()).rename("high_news")
    return news, high


def _tests(causes, effects, controls, lags, cov, prefix_c="", prefix_e=""):
    return ge.granger_matrix(causes.add_prefix(prefix_c), effects.add_prefix(prefix_e),
                             lags=lags, controls=controls, cov=cov)


def run_pipeline(refresh: bool = False, k: int = config.LSA_K, lags: str | int = "bic",
                 cov: str = "HC1", high_quantile: float = config.HIGH_NEWS_QUANTILE,
                 n_boot_rs: int = 999, save: bool = True) -> Results:
    """Run everything. ``refresh`` re-downloads the free sources (prices, FRED, prediction
    markets, GDELT); tweets are always read from the cache."""
    market = load_market_panel(refresh=refresh)
    pm = load_prediction_markets(refresh=refresh)
    cal = market.returns.index
    tweets, frep = load_tweets()

    # --- topics and sentiment ------------------------------------------------
    model = lsa.fit_lsa(tweets["text"], k=k)
    stability = lsa.label_stability(tweets["text"], model, n_boot=5)
    X, _ = lsa.build_tfidf(tweets["text"])
    spectrum = lsa.svd_check(X, 60)
    comparison = nt.compare_topic_models(tweets["clean"], n_components=12)
    daily = lsa.daily_sentiment(model, tweets, cal,
                                salience_mask=tweets["stratum"].eq("general").values)
    S, coverage = lsa.sentiment_signals(daily, window=5, min_eff_docs=5)
    S_rob, _ = lsa.sentiment_signals(daily, window=3, min_eff_docs=3, min_coverage=0.7)
    news, high = load_news(cal, refresh, high_quantile)

    # --- series for the tests -------------------------------------------------
    dprob = pm.dprob[list(config.CONTRACTS)]
    controls = market.macro[[c for c in CONTROLS if c in market.macro]]
    vol = market.vol
    vol_st, _ = ge.make_stationary(vol)
    adf = ge.adf_table(pd.concat([S.add_prefix("S_"), dprob.add_prefix("dProb_"),
                                  market.excess, vol.add_prefix("vol_")], axis=1))
    lag_table, _ = ge.select_lag(pd.concat([dprob, S], axis=1).dropna())

    # --- covariance SVD: s_{t-1} vs (dProb_t, excess_t), training window only --------
    targets = pd.concat([dprob.add_prefix("dProb_"), market.excess], axis=1)
    cs = lsa.covariance_svd(S, targets, k=3, lag=1, train_frac=0.6, n_perm=999)
    f_tests = _tests(cs.factors, targets, controls, lags, cov)
    f_oos = ge.factor_oos(cs.factors, cs.V, targets, cs.train_end, lag=1)

    # --- Stage 1 ----------------------------------------------------------------------
    stage1 = _tests(S, dprob, controls, lags, cov, "S_", "dProb_")
    stage1_rob = _tests(S_rob, dprob, controls, lags, cov, "S_", "dProb_")
    reverse = _tests(dprob, S, controls, lags, cov, "dProb_", "S_")
    sal = daily.salience[S.columns].rolling(5, min_periods=1).mean()
    stage1_sal = _tests(sal.fillna(sal.expanding().mean().shift(1)).fillna(0), dprob,
                        controls, lags, cov, "sal_", "dProb_")

    # --- Stage 2 ----------------------------------------------------------------------
    stage2_ret = _tests(dprob, market.excess, controls, lags, cov, "dProb_")
    stage2_vol = _tests(dprob, vol_st, controls, lags, cov, "dProb_", "vol_")
    s2a = []
    for issue, assets in config.ISSUE_ASSETS.items():
        have = [a for a in assets if a in market.excess]
        if issue in S and have:
            s2a.append(_tests(S[[issue]], market.excess[have], controls, lags, cov, "S_"))
    s2a = pd.concat(s2a, ignore_index=True) if s2a else pd.DataFrame()
    if len(s2a):
        s2a["q_value"] = ge.benjamini_hochberg(s2a["p_value"])

    # --- robustness battery for every FDR survivor (Stage 1 and Stage 2) --------------
    rob = []
    surv = pd.concat([stage1.assign(stage="1"), stage2_ret.assign(stage="2 ret"),
                      stage2_vol.assign(stage="2 vol")], ignore_index=True)
    surv = surv[surv["q_value"] < 0.10]
    panel = pd.concat([S.add_prefix("S_"), dprob.add_prefix("dProb_"), market.excess,
                       vol_st.add_prefix("vol_")], axis=1)
    for _, r in surv.iterrows():
        rob.append({"stage": r["stage"], **ge.granger_robustness(
            panel[r["effect"]], panel[r["cause"]], controls, int(r["lags"]),
            exclude=["2026-06-05"])})
    robustness = pd.DataFrame(rob)

    # --- Rigobon-Sack: H = top GDELT-volume sessions; outage sessions excluded ---------
    rs_rows = []
    for ch in config.CONTRACTS:
        for asset in market.excess.columns:
            try:
                r = ge.rigobon_sack(pm.dprob[ch], market.excess[asset], high, n_boot=n_boot_rs)
                rs_rows.append({"asset": asset, "contract": ch, **r.as_dict()})
            except Exception as exc:  # degenerate regime split
                log.warning("RS %s/%s failed: %s", asset, ch, exc)
    rs = pd.DataFrame(rs_rows)

    # --- tables -----------------------------------------------------------------------
    t1a = tg.table1_loadings(model, stability)
    t1b = tg.table1_issues(daily, S, coverage)
    t2 = tg.table2_stage1(stage1, reverse, stage1_rob)
    t3 = tg.table3_basket(stage2_ret, stage2_vol, rs)
    tables = {
        "table1a_svd_topics": (t1a, "Table 1A: SVD latent topics and term loadings (U_k)",
                               "Varimax-rotated LSA, k = 30, uncentred TF-IDF. Each row is one "
                               "labelled pole; loadings are entries of U_k. Stability: share of "
                               "5 refits on 80% subsamples in which the pole reappears "
                               "(same label, loading cosine >= 0.6)."),
        "table1b_issue_sentiment": (t1b, "Table 1B: Topic salience and sentiment by issue",
                                    "Salience from the general stratum only; S_k,t pooled over "
                                    "5 sessions with >= 5 effective documents. % re-signed: "
                                    "issue-conditioned polarity differs from generic VADER."),
        "table2_granger_stage1": (t2, "Table 2: Granger causality, topic sentiment to "
                                  "prediction-market probabilities",
                                  f"HC1 F-tests, lags by {str(lags).upper()}, lagged SPY, VIX "
                                  "and 2Y controls. q: Benjamini-Hochberg FDR across the table. "
                                  "p (3d): 3-session pooled sentiment. p (rev): dProb -> S."),
        "table3_basket_reactivity": (t3, "Table 3: Midterm basket reactivity to prediction-"
                                     "market shifts",
                                     "Ret/Vol: Granger F-tests of dProb lags on excess returns "
                                     "(vs SPY) and range volatility. Beta RS: Rigobon-Sack, H = "
                                     "top 20% GDELT midterm-news sessions, regime bootstrap CI."),
    }
    tg.write_tables(tables)
    res = Results(market, pm, frep, model, stability, spectrum, comparison, daily, S, S_rob,
                  coverage, news, high, adf, lag_table, cs, f_tests, f_oos, stage1, stage1_rob,
                  reverse, stage1_sal, stage2_ret, stage2_vol, robustness, s2a, rs,
                  {k_: v[0] for k_, v in tables.items()},
                  {"start": config.START, "end": config.END, "k": k, "lags": lags, "cov": cov,
                   "high_quantile": high_quantile, "n_tweets": len(tweets),
                   "built": pd.Timestamp.now(tz=config.TZ).isoformat()})
    if save:
        with open(RESULTS_PATH, "wb") as fh:
            pickle.dump(res, fh)
    return res


def load_results() -> Results:
    with open(RESULTS_PATH, "rb") as fh:
        return pickle.load(fh)
