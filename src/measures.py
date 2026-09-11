"""Scoring the corpus and assembling the analysis panel.

Both tone measures, both lexicons, plus the event-window outcomes and the
controls the regressions need. The expensive part -- re-tokenising 1,542
filings -- is cached to ``data/interim/analysis_panel.csv`` so the notebook
runs quickly on a second pass.

The corpus used for ``df`` and ``N`` is the *final estimation sample*, the same
set of documents the regressions run on. Scoring one corpus and regressing
another is explicitly penalised by the brief, so the two cannot drift apart
here: ``build_panel`` takes the filtered sample and derives everything from it.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import doc_frequency, proportional_score, tfidf_score
from .config import INTERIM_DIR
from .lexicons import load_all
from .sample import EVENT_WINDOW, POST_VOL_WINDOW, PRE_WINDOW, read_counts

LEXICONS = ("Negative", "Uncertainty")
PANEL_PATH = INTERIM_DIR / "analysis_panel.csv"
WORDFREQ_PATH = INTERIM_DIR / "corpus_word_freq.csv"


def load_corpus_counts(sample: pd.DataFrame) -> tuple[list[Counter], pd.DataFrame]:
    """Token counts for every filing in the sample, in sample order.

    ``n_words`` and ``n_distinct`` are recomputed here rather than taken from
    ``filings_meta.csv`` so that the numbers scored are demonstrably the ones
    produced by the same tokeniser call.
    """
    counts, n_words, n_distinct = [], [], []
    for i, path in enumerate(sample["text_path"], 1):
        c, nw, nd = read_counts(path)
        counts.append(c)
        n_words.append(nw)
        n_distinct.append(nd)
        if i % 250 == 0:
            print(f"  tokenised {i}/{len(sample)}", flush=True)
    recomputed = pd.DataFrame(
        {"n_words_scored": n_words, "n_distinct_scored": n_distinct},
        index=sample.index,
    )
    return counts, recomputed


def score_corpus(counts: list[Counter], n_words: list[int], n_distinct: list[int],
                 word_lists: dict[str, set[str]]) -> tuple[pd.DataFrame, dict]:
    """Proportional and tf.idf scores for each lexicon, over one corpus."""
    out, diagnostics = {}, {}
    for name in LEXICONS:
        wl = word_lists[name]
        df_counts, n_docs = doc_frequency(counts, wl)
        out[f"prop_{name.lower()}"] = [
            proportional_score(c, nw, wl) for c, nw in zip(counts, n_words)
        ]
        out[f"tfidf_{name.lower()}"] = [
            tfidf_score(c, nw, nd, wl, df_counts, n_docs)
            for c, nw, nd in zip(counts, n_words, n_distinct)
        ]
        diagnostics[name] = {
            "n_docs": n_docs,
            "list_size": len(wl),
            "words_present": int(sum(1 for w in wl if df_counts.get(w, 0) > 0)),
            "doc_freq": df_counts,
        }
    return pd.DataFrame(out), diagnostics


def corpus_word_frequency(counts: list[Counter],
                          word_lists: dict[str, set[str]]) -> pd.DataFrame:
    """Total occurrences of every lexicon word across the corpus, for Table 3."""
    rows = []
    for name in LEXICONS:
        wl = word_lists[name]
        total: Counter = Counter()
        for c in counts:
            for w in wl:
                v = c.get(w, 0)
                if v:
                    total[w] += v
        grand = sum(total.values())
        for word, n in total.items():
            rows.append({"lexicon": name, "word": word, "count": n,
                         "share_of_lexicon_count": n / grand if grand else np.nan})
    out = pd.DataFrame(rows)
    return out.sort_values(["lexicon", "count"], ascending=[True, False]).reset_index(drop=True)


def add_market_variables(sample: pd.DataFrame, panel, shares: pd.DataFrame) -> pd.DataFrame:
    """Event-window outcomes and the regression controls.

    log size uses the share count printed on *that* filing, from shares.csv,
    against the day -1 close. Using a current share count with a historical
    price is the look-ahead bug the brief calls out; joining on accession is
    what prevents it.
    """
    df = sample.copy()
    tick, d0 = df["ticker"].to_numpy(), df["day0"].to_numpy()

    df["exret_0_3"] = [panel.excess_return(t, d, *EVENT_WINDOW) for t, d in zip(tick, d0)]
    df["pre_exret"] = [panel.excess_return(t, d, *PRE_WINDOW) for t, d in zip(tick, d0)]
    df["pre_vol"] = [panel.realised_vol(t, d, *PRE_WINDOW) for t, d in zip(tick, d0)]
    df["post_vol"] = [panel.realised_vol(t, d, *POST_VOL_WINDOW) for t, d in zip(tick, d0)]
    df["dollar_vol"] = [panel.mean_dollar_volume(t, d, *PRE_WINDOW) for t, d in zip(tick, d0)]

    sh = shares[["accession", "shares_outstanding", "shares_tag"]].drop_duplicates("accession")
    df = df.merge(sh, on="accession", how="left")

    df["mktcap"] = df["price_m1"] * df["shares_outstanding"]
    df["log_size"] = np.log(df["mktcap"].where(df["mktcap"] > 0))
    df["log_dollar_vol"] = np.log(df["dollar_vol"].where(df["dollar_vol"] > 0))
    df["is_10k"] = (df["form"] == "10-K").astype(int)
    df["quarter"] = df["cal_quarter"].astype(str)
    df["firm"] = df["cik"].astype(str)
    return df


def build_panel(sample: pd.DataFrame, panel, shares: pd.DataFrame,
                use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The full analysis panel: scores + outcomes + controls, plus Table 3 input."""
    if use_cache and PANEL_PATH.exists() and WORDFREQ_PATH.exists():
        out = pd.read_csv(PANEL_PATH, dtype={"cik": str, "firm": str},
                          parse_dates=["filing_date", "day0", "day0_naive"])
        wf = pd.read_csv(WORDFREQ_PATH)
        return out, wf, {}

    word_lists = load_all()
    counts, recomputed = load_corpus_counts(sample)
    nw = recomputed["n_words_scored"].tolist()
    nd = recomputed["n_distinct_scored"].tolist()

    scores, diagnostics = score_corpus(counts, nw, nd, word_lists)
    wordfreq = corpus_word_frequency(counts, word_lists)

    out = pd.concat([sample.reset_index(drop=True),
                     recomputed.reset_index(drop=True),
                     scores.reset_index(drop=True)], axis=1)
    out = add_market_variables(out, panel, shares)

    out.to_csv(PANEL_PATH, index=False)
    wordfreq.to_csv(WORDFREQ_PATH, index=False)
    return out, wordfreq, diagnostics
