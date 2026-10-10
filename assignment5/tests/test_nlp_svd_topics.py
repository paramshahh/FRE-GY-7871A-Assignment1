"""LSA engine checks. Toy corpora and simulated series are test fixtures only."""
import numpy as np
import pandas as pd
import pytest

from src import nlp_svd_topics as lsa
from src.data_loader import trading_calendar

BANK = {
    "Housing Affordability": "mortgage rent homebuyers housing homes zoning starter mortgages",
    "Immigration": "border immigration migrants asylum deportation visa citizenship daca",
    "Healthcare Affordability": "premiums deductibles subsidies obamacare insulin prescription",
}


def _corpus(n=60, seed=0):
    rng = np.random.default_rng(seed)
    texts, truth = [], []
    for k, words in BANK.items():
        w = words.split()
        for _ in range(n):
            texts.append(" ".join(rng.choice(w, 6)) + " congress voters #midterms")
            truth.append(k)
    return pd.Series(texts), np.array(truth)


def test_tokenize_keeps_hashtags_and_cashtags():
    toks = lsa.tokenize("Mortgage rates up again @user https://x.co #Midterms2026 $NVDA the")
    assert "#midterms2026" in toks and "$nvda" in toks and "mortgage" in toks
    assert "the" not in toks and not any(t.startswith("@") or "http" in t for t in toks)


def test_tfidf_is_not_centred_and_stays_sparse():
    texts, _ = _corpus()
    X, _ = lsa.build_tfidf(texts)
    assert X.nnz < 0.5 * X.shape[0] * X.shape[1]
    assert X.min() >= 0


def test_lsa_matches_arpack():
    texts, _ = _corpus()
    m = lsa.fit_lsa(texts, k=5)
    X, _ = lsa.build_tfidf(texts)
    assert m.singular_values[:4] == pytest.approx(lsa.svd_check(X, 4), rel=1e-3)


def test_first_component_is_mean_direction_when_vocabulary_is_shared():
    # keep the shared campaign words (max_df=1.0): the uncentred first axis is one-signed
    texts, _ = _corpus()
    X, _ = lsa.build_tfidf(texts, max_df=1.0)
    from sklearn.decomposition import TruncatedSVD
    u0 = TruncatedSVD(3, random_state=0).fit(X).components_[0]
    u0 = u0 * np.sign(u0[np.abs(u0).argmax()])
    assert (u0 >= -1e-9).all()
    terms = np.array([f"t{i}" for i in range(len(u0))])
    poles = lsa.label_poles(np.c_[u0, np.r_[u0[:-1], -1.0]], terms)
    assert poles.loc[0, "label"] == "General"


def test_poles_and_weights_recover_topics():
    texts, truth = _corpus()
    m = lsa.fit_lsa(texts, k=5)
    W = lsa.topic_weights(m)
    dom = lsa.dominant_topic(W)
    assert (dom.values == truth).mean() > 0.9
    assert set(BANK) <= set(m.poles["label"])


def test_daily_sentiment_formula_and_projection():
    texts, truth = _corpus(n=30)
    cal = trading_calendar("2026-09-08", "2026-09-30")
    ts = pd.Series(pd.to_datetime("2026-09-08 14:00", utc=True)
                   + pd.to_timedelta(np.arange(len(texts)) % 10, unit="D"))
    mood = np.where(truth == "Housing Affordability", " rent soaring unaffordable disaster",
                    " great relief wonderful")
    docs = pd.DataFrame({"text": texts + mood, "created_at": ts})
    m = lsa.fit_lsa(docs["text"], k=5)
    d = lsa.daily_sentiment(m, docs, cal)
    s = d.sentiment.dropna(how="all")
    assert s.abs().max().max() <= 1 + 1e-9
    assert s["Housing Affordability"].mean() < -0.5 and s["Immigration"].mean() > 0.5
    assert d.beta.shape[1] == 5 and d.beta.dropna().shape[0] == d.n_docs.gt(0).sum()


def test_covariance_svd_finds_planted_factor_without_lookahead():
    rng = np.random.default_rng(1)
    T, K, N = 600, 6, 5
    idx = pd.bdate_range("2024-01-01", periods=T)
    s = pd.DataFrame(rng.normal(size=(T, K)), idx, [f"s{i}" for i in range(K)])
    f = (s["s0"] + s["s1"]) / np.sqrt(2)
    load = np.array([1.0, 0.8, 0.0, 0.0, -0.5])
    y = pd.DataFrame(0.25 * np.outer(f.shift(1).fillna(0), load) + rng.normal(size=(T, N)),
                     idx, [f"y{i}" for i in range(N)])
    cs = lsa.covariance_svd(s, y, k=2, lag=1, n_perm=200)
    u = cs.U["F1"].abs()
    assert set(u.nlargest(2).index) == {"s0", "s1"}
    assert cs.p_values[0] < 0.01 and cs.p_values[1] > 0.05
    assert cs.train_end < idx[int(0.61 * T)]
    assert cs.factors.shape == (T, 2)
