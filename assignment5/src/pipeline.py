"""End-to-end pipeline shared by the dashboard, the report and the notebook."""
from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field

import pandas as pd

from . import config
from . import granger_econometrics as ge
from . import nlp_topics as nt
from . import table_generator as tg
from .data_loader import MarketPanel, load_market_panel
from .prediction_markets import MarketProbabilities, load_prediction_markets

log = logging.getLogger(__name__)
RESULTS_PATH = config.OUTPUT_DIR / "results.pkl"


@dataclass
class Results:
    market: MarketPanel
    pm: MarketProbabilities
    filter_report: pd.DataFrame
    text_sources: dict
    topic_model: nt.TopicModel
    model_comparison: pd.DataFrame
    scored: pd.DataFrame
    daily: nt.DailyText
    tone_filled: pd.DataFrame
    high_news: pd.Series
    adf: pd.DataFrame
    lag_table: pd.DataFrame
    stage1: pd.DataFrame
    stage1_reverse: pd.DataFrame
    stage1_salience: pd.DataFrame
    stage2_ret: pd.DataFrame
    stage2_vol: pd.DataFrame
    tone_to_sector: pd.DataFrame
    rigobon_sack: pd.DataFrame
    tables: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)


def load_text(start: str, end: str, refresh: bool = True) -> tuple[pd.DataFrame, dict]:
    """Tweets (if a twitterapi.io key is set) and GDELT headlines. Raises if neither works."""
    frames, status = [], {}
    if os.environ.get("TWITTERAPI_IO_KEY") or (config.RAW_DIR / "tweets").exists():
        try:
            client = nt.TwitterAPIClient() if refresh and os.environ.get("TWITTERAPI_IO_KEY") \
                else None
            if client is None:
                files = sorted((config.RAW_DIR / "tweets").glob("*.csv"))
                df = pd.concat([pd.read_csv(f, dtype={"id": str}) for f in files])
                df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
            else:
                df = nt.collect_tweets(start, end, client)
            frames.append(df)
            status["twitter"] = f"{len(df):,} tweets"
        except Exception as exc:
            status["twitter"] = f"failed: {exc!r}"
    else:
        status["twitter"] = "skipped: TWITTERAPI_IO_KEY not set"
    try:
        if refresh:
            df = nt.collect_gdelt(start, end)
        else:
            files = sorted((config.RAW_DIR / "gdelt").glob("*.csv"))
            df = pd.concat([pd.read_csv(f) for f in files])
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        frames.append(df)
        status["gdelt"] = f"{len(df):,} headlines"
    except Exception as exc:
        status["gdelt"] = f"failed: {exc!r}"
    if not frames:
        raise ConnectionError(f"No text data could be loaded: {status}")
    return pd.concat(frames, ignore_index=True), status


def fill_tone(tone: pd.DataFrame) -> pd.DataFrame:
    """Days with no documents on a topic carry no information: use the expanding mean to date."""
    prior = tone.expanding().mean().shift(1)
    return tone.fillna(prior).fillna(tone.mean())


def run_pipeline(start: str = config.START, end: str = config.END, refresh: bool = True,
                 topic_method: str = "nmf", n_components: int = 12, lags: str | int = "bic",
                 cov: str = "HC1", high_quantile: float = config.HIGH_NEWS_QUANTILE,
                 save: bool = True) -> Results:
    market = load_market_panel(start, end, refresh=refresh)
    pm = load_prediction_markets(start, end, refresh=refresh)
    raw_text, text_status = load_text(start, end, refresh=refresh)
    cal = market.returns.index

    filtered, frep = nt.quality_filter(raw_text)
    tm = nt.fit_topics(filtered["clean"], method=topic_method, n_components=n_components)
    comparison = nt.compare_topic_models(filtered["clean"], n_components=n_components)
    scored = nt.score_documents(filtered, tm.labels)
    daily = nt.daily_aggregate(scored, cal)
    tone = fill_tone(daily.tone)

    dprob = pm.dprob.add_prefix("dProb_")
    controls = market.macro[[c for c in ("SPY_ret", "dVIX", "dDGS2") if c in market.macro]]
    tone_c = tone.add_prefix("tone_")
    sal_c = daily.salience.fillna(0).add_prefix("sal_")
    vol = market.vol.add_prefix("vol_")
    vol_st, _ = ge.make_stationary(vol)

    adf = ge.adf_table(pd.concat([dprob, tone_c, market.excess, vol], axis=1))
    lag_tab, _ = ge.select_lag(pd.concat([dprob, tone_c], axis=1).dropna())
    kw = dict(lags=lags, controls=controls, cov=cov)
    stage1 = ge.granger_matrix(tone_c, dprob, **kw)
    reverse = ge.granger_matrix(dprob, tone_c, **kw)
    stage1_sal = ge.granger_matrix(sal_c, dprob, **kw)
    stage2_ret = ge.granger_matrix(dprob, market.excess, **kw)
    stage2_vol = ge.granger_matrix(dprob, vol_st, **kw)
    t2s = []
    for topic, etfs in config.SECTOR_MAP.items():
        for etf in etfs:
            if etf in market.excess:
                t2s.append(ge.granger_matrix(tone_c[[f"tone_{topic}"]],
                                             market.excess[[etf]], **kw))
    tone_to_sector = pd.concat(t2s, ignore_index=True) if t2s else pd.DataFrame()
    if len(tone_to_sector):
        tone_to_sector["q_value"] = ge.benjamini_hochberg(tone_to_sector["p_value"])

    high = ge.high_news_days(daily.volume, high_quantile)
    rs_rows = []
    for ch in config.CHAMBERS:
        for etf in market.excess.columns:
            try:
                r = ge.rigobon_sack(pm.dprob[ch], market.excess[etf], high)
                rs_rows.append({"etf": etf, "chamber": ch, **r.as_dict()})
            except Exception as exc:
                log.warning("RS %s/%s failed: %s", etf, ch, exc)
    rs = pd.DataFrame(rs_rows)

    t1 = tg.table1_topics(scored, daily.salience, daily.tone)
    t2 = tg.table2_stage1(stage1, reverse)
    t3 = tg.table3_sectors(stage2_ret, stage2_vol, rs)
    tables = {
        "table1_topic_salience": (t1, "Table 1: Topic salience and sentiment distribution "
                                  "across midterm priority issues",
                                  "Salience is the daily share of all filtered documents; "
                                  "Tone = (Pos - Neg)/Total per topic-day. % Re-signed is the "
                                  "share of documents whose topic-conditioned polarity differs "
                                  "from generic VADER."),
        "table2_granger_stage1": (t2, "Table 2: Granger causality, public tone to prediction-"
                                  "market probabilities",
                                  f"OLS F-tests ({cov} covariance) of lags of Tone_k in an "
                                  "AR model of dProb with lagged SPY, VIX and 2Y controls; lags "
                                  "by BIC. q is the Benjamini-Hochberg FDR-adjusted p-value. "
                                  "*** p<0.01, ** p<0.05, * p<0.10."),
        "table3_sector_reactivity": (t3, "Table 3: Sector ETF reactivity to prediction-market "
                                     "shifts", "Ret/Vol: Granger F-tests of dProb lags on excess "
                                     "returns (vs SPY) and range volatility. β RS: Rigobon-Sack "
                                     "estimate identified by the variance shift of dProb on "
                                     f"high-news days (top {int(100 * (1 - high_quantile))}% of "
                                     "discourse volume), bootstrap 95% CI."),
    }
    tg.write_tables(tables)
    res = Results(market, pm, frep.as_frame(), text_status, tm, comparison, scored, daily,
                  tone, high, adf, lag_tab, stage1, reverse, stage1_sal, stage2_ret, stage2_vol,
                  tone_to_sector, rs, {k: v[0] for k, v in tables.items()},
                  {"start": start, "end": end, "topic_method": topic_method, "lags": lags,
                   "cov": cov, "high_quantile": high_quantile,
                   "built": pd.Timestamp.now(tz=config.TZ).isoformat()})
    if save:
        with open(RESULTS_PATH, "wb") as fh:
            pickle.dump(res, fh)
    return res


def load_results() -> Results:
    with open(RESULTS_PATH, "rb") as fh:
        return pickle.load(fh)
