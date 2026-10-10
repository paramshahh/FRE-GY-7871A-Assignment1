"""Text ingestion, topic extraction and topic-conditioned emotional tone.

Sources
-------
* Tweets: twitterapi.io ``GET /twitter/tweet/advanced_search`` (header
  ``X-API-Key``, env ``TWITTERAPI_IO_KEY``). Each calendar day is split into
  four 6-hour windows queried separately, which keeps ``queryType=Latest``
  from sampling only the last hour of a day and keeps pagination shallow.
* News: GDELT DOC 2.0 ``artlist`` (free, no key; one request per ~5 s; it only
  searches roughly the last three months of coverage).

Pipeline
--------
raw text -> quality filter -> clean -> topic (NMF/LDA/BERTopic, labelled with
seed vocabulary) -> topic-conditioned polarity and emotions -> daily salience
and Tone_{k,t} = (Pos - Neg) / Total on the trading-day calendar.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

from . import config
from .data_loader import to_trading_day

log = logging.getLogger(__name__)

TOPIC_NAMES = list(config.TOPICS)
OTHER = "Other"

ELECTION_QUERY = ('(midterms OR midterm OR "2026 election" OR congress OR senate OR '
                  '"house race" OR ballot OR candidate OR #midterms2026 OR #vote2026)')
GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY = ('(midterms OR "midterm elections" OR "midterm election") '
               'sourcecountry:US sourcelang:english')


# ---------------------------------------------------------------------------
# twitterapi.io
# ---------------------------------------------------------------------------
class TwitterAPIClient:
    """Thin client for twitterapi.io advanced search."""

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None,
                 pause: float = 0.25):
        self.api_key = api_key or os.environ.get("TWITTERAPI_IO_KEY", "")
        if not self.api_key:
            raise RuntimeError("Set TWITTERAPI_IO_KEY to use twitterapi.io")
        self.s = session or requests.Session()
        self.s.headers["X-API-Key"] = self.api_key
        self.pause = pause

    def search(self, query: str, max_pages: int = 5, query_type: str = "Latest") -> list[dict]:
        tweets, cursor, seen = [], "", set()
        for _ in range(max_pages):
            r = self.s.get(config.TWITTERAPI_URL, timeout=30,
                           params={"query": query, "queryType": query_type, "cursor": cursor})
            if r.status_code == 429:
                time.sleep(5)
                continue
            r.raise_for_status()
            js = r.json()
            page = [t for t in js.get("tweets", []) if t.get("id") not in seen]
            if not page:  # guards against has_next_page=true with repeated/empty pages
                break
            seen.update(t.get("id") for t in page)
            tweets.extend(page)
            cursor = js.get("next_cursor") or ""
            if not js.get("has_next_page") or not cursor:
                break
            time.sleep(self.pause)
        return tweets


def build_query(base: str, since: pd.Timestamp, until: pd.Timestamp) -> str:
    """Advanced-search query for [since, until) in UTC, English, no retweets."""
    fmt = "%Y-%m-%d_%H:%M:%S_UTC"
    return (f"{base} lang:en -is:retweet "
            f"since:{since.strftime(fmt)} until:{until.strftime(fmt)}")


def parse_tweets(raw: list[dict]) -> pd.DataFrame:
    """twitterapi.io tweet objects -> flat frame."""
    rows = []
    for t in raw:
        a = t.get("author") or {}
        text = t.get("text") or ""
        rows.append({
            "id": str(t.get("id")),
            "created_at": pd.to_datetime(t.get("createdAt"), format="%a %b %d %H:%M:%S %z %Y",
                                         errors="coerce", utc=True),
            "text": text,
            "is_retweet": bool(t.get("retweeted_tweet")) or text.startswith("RT @"),
            "is_reply": bool(t.get("isReply")),
            "followers": a.get("followers"),
            "views": t.get("viewCount"),
            "likes": t.get("likeCount"),
            "author": a.get("userName") or a.get("id"),
            "lang": t.get("lang"),
            "source": "twitter",
        })
    return pd.DataFrame(rows)


def collect_tweets(start: str, end: str, client: TwitterAPIClient | None = None,
                   pages_per_window: int = 3, base: str = ELECTION_QUERY) -> pd.DataFrame:
    """Backfill [start, end] in 6-hour windows, caching each day to data/raw/tweets/."""
    client = client or TwitterAPIClient()
    out_dir = config.RAW_DIR / "tweets"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for day in pd.date_range(start, end, freq="D", tz="UTC"):
        path = out_dir / f"{day:%Y-%m-%d}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, dtype={"id": str}))
            continue
        raw = []
        for h in range(0, 24, 6):
            lo = day + pd.Timedelta(hours=h)
            raw += client.search(build_query(base, lo, lo + pd.Timedelta(hours=6)),
                                 max_pages=pages_per_window)
        df = parse_tweets(raw)
        df.to_csv(path, index=False)
        frames.append(df)
        log.info("%s: %d tweets", day.date(), len(df))
    df = pd.concat(frames, ignore_index=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# GDELT news headlines
# ---------------------------------------------------------------------------
def parse_gdelt_artlist(payload: dict) -> pd.DataFrame:
    arts = payload.get("articles", []) or []
    df = pd.DataFrame([{
        "id": a.get("url"),
        "created_at": pd.to_datetime(a.get("seendate"), format="%Y%m%dT%H%M%SZ",
                                     errors="coerce", utc=True),
        "text": a.get("title", ""),
        "domain": a.get("domain"),
        "lang": "en",
        "source": "gdelt",
    } for a in arts])
    return df


def collect_gdelt(start: str, end: str, session: requests.Session | None = None,
                  query: str = GDELT_QUERY, pause: float = 12.0) -> pd.DataFrame:
    """Up to 250 headlines per calendar day, cached to data/raw/gdelt/."""
    s = session or requests.Session()
    out_dir = config.RAW_DIR / "gdelt"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for day in pd.date_range(start, end, freq="D"):
        path = out_dir / f"{day:%Y-%m-%d}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
            continue
        df = parse_gdelt_artlist(gdelt_request({
            "query": query, "mode": "artlist", "maxrecords": 250, "sort": "hybridrel",
            "startdatetime": f"{day:%Y%m%d}000000",
            "enddatetime": f"{day:%Y%m%d}235959"}, s, pause=pause))
        df.to_csv(path, index=False)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    return df.drop_duplicates(subset=["text"])


def gdelt_request(params: dict, session: requests.Session | None = None,
                  pause: float = 12.0, retries: int = 6) -> dict:
    """GDELT DOC call with patient back-off; GDELT answers 429 in plain text when throttled."""
    s = session or requests.Session()
    for i in range(retries):
        time.sleep(pause if i == 0 else 30 * i)
        r = s.get(GDELT_DOC, params={**params, "format": "json"}, timeout=90)
        if r.status_code == 429 or r.text.startswith("Please limit"):
            continue
        r.raise_for_status()
        return r.json() if r.text.strip() else {}
    raise ConnectionError("GDELT kept rate-limiting the request")


def parse_gdelt_timeline(payload: dict) -> pd.DataFrame:
    """timelinevolraw/timelinetone payload -> date-indexed frame (value, plus norm if present)."""
    tl = (payload.get("timeline") or [{}])[0].get("data", [])
    df = pd.DataFrame(tl)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%SZ")
    return df.set_index("date")


def collect_gdelt_timelines(start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """Daily article counts, total monitored volume and GDELT tone, overall and per topic.

    Columns: ``count_<topic>``, ``tone_<topic>`` (and ``count_All``/``tone_All``
    for all midterm coverage) plus ``norm`` = all US English articles GDELT
    monitored that day, the denominator for salience.
    """
    path = config.RAW_DIR / "gdelt_timelines.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    s = requests.Session()
    window = {"startdatetime": f"{pd.Timestamp(start):%Y%m%d}000000",
              "enddatetime": f"{pd.Timestamp(end):%Y%m%d}235959"}
    queries = {"All": f"{config.GDELT_MIDTERMS} {config.GDELT_FILTERS}"}
    queries.update({k: f"{config.GDELT_MIDTERMS} {v} {config.GDELT_FILTERS}"
                    for k, v in config.GDELT_TOPIC_TERMS.items()})
    cols = {}
    for name, q in queries.items():
        vol = parse_gdelt_timeline(gdelt_request({"query": q, "mode": "timelinevolraw",
                                                  "timelinesmooth": 0, **window}, s))
        tone = parse_gdelt_timeline(gdelt_request({"query": q, "mode": "timelinetone",
                                                   "timelinesmooth": 0, **window}, s))
        cols[f"count_{name}"] = vol["value"]
        if name == "All":
            cols["norm"] = vol["norm"]
        cols[f"tone_{name}"] = tone["value"]
        log.info("GDELT %s: %d days", name, len(vol))
    df = pd.DataFrame(cols)
    df.to_csv(path)
    return df


def gdelt_outages(norm: pd.Series, window: int = 15, floor: float = 0.3) -> pd.Series:
    """Days where GDELT's total monitored volume is below ``floor`` x its rolling median."""
    med = norm.rolling(window, center=True, min_periods=5).median()
    return (norm < floor * med).rename("outage")


def gdelt_to_trading_days(tl: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily GDELT timelines (UTC days) -> trading days, with outage days set to NaN.

    A UTC day ends at 20:00 ET (19:00 in winter), after the equity close, so its
    counts are information for the *next* session. Counts are summed over the
    UTC days that map to a session (weekends roll into Monday) and tones are
    article-weighted means. Salience is each topic's share of midterm coverage.
    """
    tl = tl.copy()
    bad = gdelt_outages(tl["norm"])
    tl.loc[bad] = np.nan
    ends = pd.Series(tl.index + pd.Timedelta(days=1), index=tl.index).dt.tz_localize("UTC")
    tl["session"] = to_trading_day(ends, calendar).values
    tl = tl.dropna(subset=["session"])
    topics = [c[len("count_"):] for c in tl.columns if c.startswith("count_")]
    out = pd.DataFrame(index=calendar)
    g = tl.groupby("session")
    for k in topics:
        n = g[f"count_{k}"].sum(min_count=1)
        w = (tl[f"tone_{k}"] * tl[f"count_{k}"]).groupby(tl["session"]).sum(min_count=1)
        out[f"count_{k}"] = n
        out[f"tone_{k}"] = w / n.replace(0, np.nan)
    out["norm"] = g["norm"].sum(min_count=1)
    out["outage_days"] = bad.groupby(tl["session"]).sum() if len(tl) else 0
    for k in topics:
        if k != "All":
            out[f"sal_{k}"] = out[f"count_{k}"] / out["count_All"].replace(0, np.nan)
    out.index.name = "date"
    return out


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------
_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"@\w+")
_NONWORD = re.compile(r"[^a-z0-9#\s']")


def normalise(text: str) -> str:
    t = _URL.sub(" ", str(text).lower())
    t = _MENTION.sub(" ", t)
    t = re.sub(r"^rt\s+:?", " ", t)
    t = _NONWORD.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class FilterReport:
    steps: list[tuple[str, int]] = field(default_factory=list)

    def add(self, name: str, n: int) -> None:
        self.steps.append((name, n))

    def as_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.steps, columns=["step", "remaining"])
        df["removed"] = (-df["remaining"].diff()).fillna(0).astype(int)
        return df


def quality_filter(df: pd.DataFrame, min_followers: int = config.MIN_FOLLOWERS,
                   min_views: int = config.MIN_VIEWS,
                   astroturf_accounts: int = config.ASTROTURF_MIN_ACCOUNTS,
                   ) -> tuple[pd.DataFrame, FilterReport]:
    """Bot and astroturf screen. Rules that need tweet metadata skip news rows."""
    rep = FilterReport()
    out = df.copy()
    rep.add("raw", len(out))
    out = out.drop_duplicates(subset=["id"])
    rep.add("duplicate ids", len(out))
    if "lang" in out:
        out = out[out["lang"].fillna("en").eq("en")]
    rep.add("non-English", len(out))
    tw = out["source"].eq("twitter") if "source" in out else pd.Series(True, index=out.index)
    if "is_retweet" in out:
        out = out[~(tw & out["is_retweet"].fillna(False).astype(bool))]
        tw = tw.loc[out.index]
    rep.add("retweets", len(out))
    if "followers" in out:
        f = pd.to_numeric(out["followers"], errors="coerce").fillna(0)
        out = out[~tw | (f >= min_followers)]
        tw = tw.loc[out.index]
    rep.add(f"followers < {min_followers}", len(out))
    if "views" in out:
        v = pd.to_numeric(out["views"], errors="coerce").fillna(0)
        out = out[~tw | (v >= min_views)]
        tw = tw.loc[out.index]
    rep.add(f"views < {min_views}", len(out))
    out = out.assign(clean=out["text"].map(normalise))
    if "author" in out:
        n_auth = out.groupby("clean")["author"].transform("nunique")
        out = out[~(tw & (n_auth >= astroturf_accounts))]
    rep.add(f"astroturf (same text, >= {astroturf_accounts} accounts)", len(out))
    out = out[out["clean"].str.split().str.len() >= 4]
    rep.add("fewer than 4 tokens", len(out))
    return out, rep


# ---------------------------------------------------------------------------
# Topic models
# ---------------------------------------------------------------------------
def label_components(top_terms: list[list[str]], seeds: dict[str, list[str]] = config.TOPICS,
                     min_hits: int = 2) -> list[str]:
    """Name each unsupervised component by its overlap with the seed vocabularies."""
    labels = []
    seed_sets = {k: set(v) for k, v in seeds.items()}
    for terms in top_terms:
        toks = set(" ".join(terms).split())
        hits = {k: len(toks & s) for k, s in seed_sets.items()}
        best = max(hits, key=hits.get)
        labels.append(best if hits[best] >= min_hits else OTHER)
    return labels


def seed_topic(clean: str, seeds: dict[str, list[str]] = config.TOPICS) -> str:
    toks = set(clean.split())
    hits = {k: len(toks & set(v)) for k, v in seeds.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] > 0 else OTHER


@dataclass
class TopicModel:
    method: str
    n_components: int
    top_terms: list[list[str]]
    component_labels: list[str]
    doc_topic: np.ndarray  # documents x components
    labels: pd.Series      # final priority-topic label per document


def fit_topics(texts: pd.Series, method: str = "nmf", n_components: int = 12,
               n_top: int = 15, min_weight: float = 0.02, seed: int = config.SEED) -> TopicModel:
    """Fit NMF (default), LDA or BERTopic and map components to priority topics.

    Over-clustering (``n_components`` > number of priority topics) and then
    labelling lets several components roll up to one issue and lets campaign
    chatter fall into ``Other`` instead of being forced into an issue.
    Documents with negligible component weight fall back to seed matching.
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    texts = texts.fillna("").astype(str)
    min_df = 3 if len(texts) > 200 else 1
    if method == "bertopic":
        return _fit_bertopic(texts, n_components, n_top)
    if method == "lda":
        from sklearn.decomposition import LatentDirichletAllocation

        vec = CountVectorizer(stop_words="english", min_df=min_df, max_df=0.5, ngram_range=(1, 2))
        X = vec.fit_transform(texts)
        model = LatentDirichletAllocation(n_components=n_components, random_state=seed,
                                          learning_method="batch", max_iter=30)
        W = model.fit_transform(X)
    else:
        from sklearn.decomposition import NMF

        vec = TfidfVectorizer(stop_words="english", min_df=min_df, max_df=0.5,
                              ngram_range=(1, 2), sublinear_tf=True)
        X = vec.fit_transform(texts)
        model = NMF(n_components=n_components, random_state=seed, init="nndsvda", max_iter=400)
        W = model.fit_transform(X)
    vocab = np.array(vec.get_feature_names_out())
    top = [list(vocab[np.argsort(row)[::-1][:n_top]]) for row in model.components_]
    comp_labels = label_components(top)
    best = W.argmax(axis=1)
    lab = np.array(comp_labels, dtype=object)[best]
    weak = W.max(axis=1) < (min_weight if method == "nmf" else 1.0 / n_components + 1e-9)
    lab = np.where(weak, [seed_topic(t) for t in texts], lab)
    return TopicModel(method, n_components, top, comp_labels, W,
                      pd.Series(lab, index=texts.index, name="topic"))


def _fit_bertopic(texts: pd.Series, n_components: int, n_top: int) -> TopicModel:
    from bertopic import BERTopic  # optional dependency

    model = BERTopic(nr_topics=n_components, min_topic_size=20, calculate_probabilities=False)
    topics, _ = model.fit_transform(texts.tolist())
    info = model.get_topics()
    ids = sorted(k for k in info if k != -1)
    top = [[w for w, _ in info[k][:n_top]] for k in ids]
    comp_labels = label_components(top)
    m = dict(zip(ids, comp_labels))
    lab = [m.get(t, OTHER) if t != -1 else seed_topic(x) for t, x in zip(topics, texts)]
    W = np.zeros((len(texts), len(ids)))
    for i, t in enumerate(topics):
        if t in ids:
            W[i, ids.index(t)] = 1
    return TopicModel("bertopic", len(ids), top, comp_labels, W,
                      pd.Series(lab, index=texts.index, name="topic"))


def umass_coherence(texts: pd.Series, top_terms: list[list[str]], n: int = 10) -> float:
    """Mean UMass coherence of the top-n unigrams of each component (higher is better)."""
    docs = [set(t.split()) for t in texts]
    df1: dict[str, int] = {}
    for d in docs:
        for w in d:
            df1[w] = df1.get(w, 0) + 1
    scores = []
    for terms in top_terms:
        words = [w for w in terms if " " not in w][:n]
        s, c = 0.0, 0
        for i in range(1, len(words)):
            for j in range(i):
                co = sum(1 for d in docs if words[i] in d and words[j] in d)
                if df1.get(words[j], 0):
                    s += np.log((co + 1) / df1[words[j]])
                    c += 1
        if c:
            scores.append(s / c)
    return float(np.mean(scores)) if scores else np.nan


def topic_diversity(top_terms: list[list[str]], n: int = 10) -> float:
    """Share of unique words among all components' top-n words."""
    words = [w for t in top_terms for w in t[:n]]
    return len(set(words)) / max(len(words), 1)


def compare_topic_models(texts: pd.Series, n_components: int = 12,
                         methods: tuple[str, ...] = ("nmf", "lda")) -> pd.DataFrame:
    """Coherence, diversity and labelled-topic coverage for each method."""
    rows = []
    for m in methods:
        try:
            tm = fit_topics(texts, method=m, n_components=n_components)
        except ImportError:
            continue
        rows.append({"method": m.upper(), "components": tm.n_components,
                     "umass_coherence": umass_coherence(texts, tm.top_terms),
                     "topic_diversity": topic_diversity(tm.top_terms),
                     "share_labelled": float((tm.labels != OTHER).mean()),
                     "issues_found": len(set(tm.component_labels) - {OTHER})})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Topic-conditioned tone and emotions
# ---------------------------------------------------------------------------
# Words whose valence depends on the issue. Generic sentiment models (VADER,
# off-the-shelf FinBERT) score "rising", "growth" or "cuts" the same way in
# every context; "prices rising" and "coverage rising" are opposites.
TOPIC_LEXICON: dict[str, dict[str, float]] = {
    "Economy/Inflation": {"rising": -2.0, "higher": -1.5, "soaring": -2.5, "surge": -2.0,
                          "surging": -2.0, "spike": -2.0, "hike": -1.8, "expensive": -2.0,
                          "unaffordable": -2.5, "tariffs": -1.0, "recession": -2.5,
                          "layoffs": -2.5, "cooling": 1.5, "falling": 1.0, "cheaper": 2.0,
                          "affordable": 1.8, "growth": 1.5, "hiring": 1.8},
    "Housing Affordability": {"rising": -2.0, "higher": -1.5, "soaring": -2.5, "surge": -2.0,
                              "unaffordable": -2.5, "expensive": -2.0, "priced": -1.5,
                              "outbid": -1.8, "eviction": -2.5, "evictions": -2.5,
                              "falling": 1.2, "lower": 1.5, "cooling": 1.2, "cheaper": 2.0,
                              "affordable": 1.8, "approved": 1.2, "closing": 0.5},
    "Healthcare Affordability": {"rising": -2.0, "higher": -1.5, "spike": -2.0, "hike": -2.0,
                                 "expire": -1.8, "expired": -1.8, "expiring": -1.5,
                                 "cliff": -1.8, "doubled": -2.0, "unaffordable": -2.5,
                                 "capped": 1.8, "cap": 1.2, "lower": 1.5, "cheaper": 2.0,
                                 "extended": 1.5, "saving": 1.8, "negotiated": 1.2},
    "Healthcare": {"cuts": -2.5, "cut": -2.0, "rising": -2.0, "premiums": -0.5,
                   "uninsured": -2.0, "expire": -1.5, "expired": -1.5, "lost": -2.0,
                   "expand": 1.8, "expanded": 1.8, "covered": 1.5, "capped": 1.5,
                   "lower": 1.5, "saving": 1.8},
    "Energy/Climate": {"rising": -1.8, "blackouts": -2.5, "blackout": -2.5, "doubled": -1.8,
                       "repealed": -1.5, "cheaper": 2.0, "booming": 2.0, "affordable": 1.8,
                       "lower": 1.2, "spill": -2.0, "wildfire": -2.0},
    "Immigration": {"surge": -2.0, "crisis": -2.5, "chaos": -2.5, "overwhelm": -2.0,
                    "raids": -2.0, "secure": 1.5, "down": 0.8, "shrinking": 1.2, "deal": 1.2},
    "Foreign Policy/Defense": {"escalation": -2.5, "invasion": -2.5, "strikes": -1.5,
                               "missiles": -1.5, "ceasefire": 2.0, "peace": 2.0,
                               "deterrence": 1.0, "allies": 0.8},
    "Technology/AI": {"deepfakes": -2.5, "deepfake": -2.5, "replace": -1.5, "surveillance": -2.0,
                      "breach": -2.5, "innovation": 1.8, "hiring": 1.8, "chips": 0.5},
}

EMOTIONS: dict[str, set[str]] = {
    "anxiety": {"worried", "worry", "afraid", "scared", "scary", "fear", "fears", "anxious",
                "nervous", "terrifying", "uncertain", "uncertainty", "panic", "risk", "threat",
                "struggling", "can't", "cannot", "crushing", "dread"},
    "anger": {"furious", "angry", "outrage", "outrageous", "disgrace", "disgusting", "corrupt",
              "betrayal", "betrayed", "sick", "rigged", "lies", "liars", "ridiculous",
              "shameful", "insane", "chaos"},
    "optimism": {"hope", "hopeful", "optimistic", "proud", "confident", "relief", "better",
                 "improving", "progress", "excited", "finally", "winning", "strong", "good",
                 "great", "booming"},
}


class ToneScorer:
    """VADER compound with an issue-specific lexicon overlay, per topic."""

    def __init__(self, threshold: float = 0.05):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self.threshold = threshold
        self.generic = SentimentIntensityAnalyzer()
        self.by_topic = {}
        for k, extra in TOPIC_LEXICON.items():
            a = SentimentIntensityAnalyzer()
            a.lexicon = {**a.lexicon, **extra}
            self.by_topic[k] = a

    def compound(self, text: str, topic: str | None = None) -> float:
        a = self.by_topic.get(topic, self.generic)
        return a.polarity_scores(text)["compound"]

    def polarity(self, compound: float) -> int:
        return 1 if compound >= self.threshold else (-1 if compound <= -self.threshold else 0)


def emotion_flags(clean: str) -> dict[str, int]:
    toks = set(clean.split())
    return {e: int(bool(toks & words)) for e, words in EMOTIONS.items()}


def score_documents(df: pd.DataFrame, topics: pd.Series,
                    scorer: ToneScorer | None = None) -> pd.DataFrame:
    """Per-document topic, conditioned and generic polarity, and emotion flags."""
    scorer = scorer or ToneScorer()
    out = df.copy()
    out["topic"] = topics.reindex(out.index)
    out["compound"] = [scorer.compound(t, k) for t, k in zip(out["text"], out["topic"])]
    out["compound_generic"] = [scorer.compound(t) for t in out["text"]]
    out["polarity"] = out["compound"].map(scorer.polarity)
    out["polarity_generic"] = out["compound_generic"].map(scorer.polarity)
    emo = pd.DataFrame([emotion_flags(c) for c in out["clean"]], index=out.index)
    return pd.concat([out, emo], axis=1)


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------
@dataclass
class DailyText:
    salience: pd.DataFrame   # date x topic, share of all documents that day
    tone: pd.DataFrame       # date x topic, (Pos - Neg) / Total, NaN if no documents
    counts: pd.DataFrame     # date x topic, document counts
    emotions: dict[str, pd.DataFrame]  # emotion -> date x topic share of documents
    volume: pd.Series        # documents per trading day


def daily_aggregate(scored: pd.DataFrame, calendar: pd.DatetimeIndex,
                    topics: list[str] = TOPIC_NAMES) -> DailyText:
    """Aggregate to trading days; post-close and weekend text rolls to the next session."""
    d = scored.copy()
    d["date"] = to_trading_day(d["created_at"], calendar)
    d = d.dropna(subset=["date"])
    g = d.groupby(["date", "topic"])
    counts = g.size().unstack(fill_value=0).reindex(index=calendar, fill_value=0)
    for k in topics + [OTHER]:
        if k not in counts:
            counts[k] = 0
    volume = counts.sum(axis=1)
    pos = d[d["polarity"] > 0].groupby(["date", "topic"]).size().unstack(fill_value=0)
    neg = d[d["polarity"] < 0].groupby(["date", "topic"]).size().unstack(fill_value=0)
    pos = pos.reindex(index=calendar, columns=topics, fill_value=0)
    neg = neg.reindex(index=calendar, columns=topics, fill_value=0)
    tot = counts[topics].replace(0, np.nan)
    tone = (pos - neg) / tot
    salience = counts[topics].div(volume.replace(0, np.nan), axis=0)
    emotions = {e: d.groupby(["date", "topic"])[e].mean().unstack()
                .reindex(index=calendar, columns=topics) for e in EMOTIONS}
    for f in (tone, salience, counts):
        f.index.name = "date"
    return DailyText(salience, tone, counts[topics + [OTHER]], emotions, volume)
