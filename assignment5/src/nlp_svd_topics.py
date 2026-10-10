"""LSA topic engine: TF-IDF, truncated SVD, daily projection, topic-conditioned
sentiment, and SVD of the sentiment-return cross-covariance.

Notation follows the lecture (04b-SVD-Appl). With the term-document matrix
X (m terms x n documents), LSA keeps the k largest singular triplets

    X ~= U_k S_k V_k^T,   U_k: term loadings (m x k),  V_k: document loadings (n x k).

scikit-learn stores documents in rows, i.e. it factors X^T = V_k S_k U_k^T, so
``TruncatedSVD.components_`` is U_k^T and ``transform`` returns V_k S_k.

The matrix is *not* demeaned. TF-IDF is sparse (well over 99% zeros); centring
would make it dense, and it would also say that a term a document never uses
carries negative information, i.e. replace the natural prior that unobserved
terms have zero weight with "below-average weight". The price is that the first
singular vector is the average-document direction (all loadings one sign); it
measures overall discourse intensity, not an issue, and is reported separately.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from . import config
from .data_loader import to_trading_day
from .nlp_topics import OTHER, ToneScorer

TOKEN = re.compile(r"(?u)[#$]?[a-z][a-z0-9_'-]+")
# scikit-learn's English list plus Twitter/web noise; issue words are never stopped
STOP = set(ENGLISH_STOP_WORDS) | {
    "amp", "via", "it's", "don't", "we're", "i'm", "he's", "that's", "they're", "you're",
    "can't", "won't", "didn't", "doesn't", "isn't", "let's", "just", "like", "get", "got",
    "going", "want", "need", "said", "says", "say", "today", "time", "year", "years", "new",
    "make", "know", "think", "right", "way", "really", "rt", "us", "lot", "day", "week"}
_URL = re.compile(r"https?://\S+")
_MENTION = re.compile(r"@\w+")


def tokenize(text: str) -> list[str]:
    """Lower-case words, keeping #hashtags and $CASHTAGS; URLs and @handles removed."""
    t = _MENTION.sub(" ", _URL.sub(" ", str(text).lower()))
    return [w.strip("'-") for w in TOKEN.findall(t) if w not in STOP and len(w.strip("'-")) > 1]


def build_tfidf(texts: pd.Series, min_df: int = 3, max_df: float = 0.4,
                max_features: int = 20000) -> tuple[sparse.csr_matrix, TfidfVectorizer]:
    """Documents x terms TF-IDF (unigrams + bigrams, sublinear tf, L2 rows), never centred."""
    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                          ngram_range=(1, 2), min_df=min_df if len(texts) > 50 else 1,
                          max_df=max_df, max_features=max_features, sublinear_tf=True)
    X = vec.fit_transform(texts.fillna("").astype(str))
    return X.tocsr(), vec


# ---------------------------------------------------------------------------
# LSA
# ---------------------------------------------------------------------------
def varimax(L: np.ndarray, gamma: float = 1.0, max_iter: int = 200,
            tol: float = 1e-8) -> np.ndarray:
    """Orthogonal rotation matrix R maximising the variance of squared loadings of L R.

    Rotating the retained term loadings inside the same k-dimensional subspace
    leaves the rank-k approximation, its fitted values and its explained
    variance unchanged; it only concentrates each axis on fewer terms, which is
    what makes LSA axes readable as topics.
    """
    p, k = L.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        Lr = L @ R
        u, s, vt = np.linalg.svd(L.T @ (Lr ** 3 - (gamma / p) * Lr @ np.diag((Lr ** 2).sum(0))))
        R = u @ vt
        d_old, d = d, s.sum()
        if d_old and d / d_old < 1 + tol:
            break
    return R


@dataclass
class LSAModel:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    terms: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    U: np.ndarray                 # terms x k term loadings, (rotated,) sign-oriented
    rotation: np.ndarray          # (k-1) x (k-1) varimax rotation of components 1..k-1
    poles: pd.DataFrame           # one row per (component, sign): label, seed hits, top terms
    doc_scores: np.ndarray        # documents x k, cosine-normalised V_k S_k


def fit_lsa(texts: pd.Series, k: int = 20, seed: int = config.SEED,
            min_df: int = 3, rotate: bool = True) -> LSAModel:
    """Truncated SVD of the (uncentred) TF-IDF matrix and labelling of each component pole.

    With ``rotate``, components 1..k-1 are varimax-rotated (component 0, the
    mean-document axis, is left alone). Document scores are rotated with them,
    so Z = X U holds for the rotated loadings too.
    """
    X, vec = build_tfidf(texts, min_df=min_df)
    k = int(min(k, min(X.shape) - 1))
    svd = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=10, random_state=seed)
    Z = svd.fit_transform(X)                     # = V_k S_k = X U_k
    U = svd.components_.T.copy()                 # = U_k (terms x k)
    R = np.eye(k - 1)
    if rotate and k > 2:
        R = varimax(U[:, 1:])
        U[:, 1:] = U[:, 1:] @ R
        Z[:, 1:] = Z[:, 1:] @ R
    # SVD signs are arbitrary: orient each axis so its largest-|loading| term is positive
    flip = np.sign(U[np.abs(U).argmax(axis=0), np.arange(k)])
    flip[flip == 0] = 1
    U *= flip
    Z *= flip
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    doc_scores = Z / np.where(norms > 0, norms, 1)[:, None]
    terms = vec.get_feature_names_out()
    return LSAModel(vec, svd, terms, svd.singular_values_, svd.explained_variance_ratio_, U, R,
                    label_poles(U, terms), doc_scores)


def label_stability(texts: pd.Series, model: LSAModel, n_boot: int = 5, frac: float = 0.8,
                    min_cos: float = 0.6, seed: int = config.SEED) -> pd.DataFrame:
    """How often each labelled pole of ``model`` reappears when refitting on subsamples.

    A pole "reappears" when some pole of the refitted model carries the same
    label and its term-loading vector (on the shared vocabulary) has cosine
    similarity >= ``min_cos`` with the original.
    """
    rng = np.random.default_rng(seed)
    labelled = model.poles[~model.poles["label"].isin([OTHER, "General"])]
    hits = np.zeros(len(labelled))
    pos = {t: i for i, t in enumerate(model.terms)}
    for b in range(n_boot):
        idx = rng.choice(len(texts), int(frac * len(texts)), replace=False)
        m = fit_lsa(texts.iloc[idx].reset_index(drop=True), k=model.U.shape[1], seed=seed + b)
        shared = [t for t in m.terms if t in pos]
        a_idx = [pos[t] for t in shared]
        b_pos = {t: i for i, t in enumerate(m.terms)}
        b_idx = [b_pos[t] for t in shared]
        for r, (_, p) in enumerate(labelled.iterrows()):
            u = np.maximum(p["sign"] * model.U[a_idx, p["component"]], 0)
            best = 0.0
            for _, q in m.poles[m.poles["label"] == p["label"]].iterrows():
                v = np.maximum(q["sign"] * m.U[b_idx, q["component"]], 0)
                den = np.linalg.norm(u) * np.linalg.norm(v)
                best = max(best, float(u @ v / den) if den else 0.0)
            hits[r] += best >= min_cos
    out = labelled[["component", "sign", "label", "seed_share", "purity"]].copy()
    out["stability"] = hits / n_boot
    return out


def label_poles(U: np.ndarray, terms: np.ndarray, n_top: int = 15,
                seeds: dict[str, list[str]] = config.TOPICS, min_share: float = 0.15,
                min_purity: float = 0.6, general_share: float = 0.9) -> pd.DataFrame:
    """Name the positive and negative pole of every component.

    For a pole, ``share_k`` is the fraction of the pole's loading energy
    (sum of squared positive loadings) on terms containing a seed word of issue
    k. The pole takes the best issue's name when that share is at least
    ``min_share`` and the best issue holds at least ``min_purity`` of all seed
    energy on the pole; otherwise it is ``Other``. The first component of an
    uncentred LSA is usually the mean-document axis: when its loading energy is
    at least ``general_share`` one-signed *and* no single issue dominates it,
    it is labelled ``General`` (overall discourse intensity) rather than forced
    onto an issue.
    """
    seed_sets = {k: set(v) for k, v in seeds.items()}
    words = [set(t.replace("#", "").replace("$", "").split()) for t in terms]
    member = {k: np.array([bool(w & sset) for w in words]) for k, sset in seed_sets.items()}
    rows = []
    for j in range(U.shape[1]):
        u = U[:, j]
        e = u ** 2
        one_signed = max(e[u > 0].sum(), e[u < 0].sum()) / e.sum() >= general_share
        for sign in (+1, -1):
            pe = np.maximum(sign * u, 0) ** 2
            tot = pe.sum()
            shares = {k: pe[m].sum() / tot if tot > 0 else 0.0 for k, m in member.items()}
            best = max(shares, key=shares.get)
            seed_tot = sum(shares.values())
            purity = shares[best] / seed_tot if seed_tot > 0 else 0.0
            if j == 0 and one_signed and purity < min_purity:
                label = "General"
            elif shares[best] >= min_share and purity >= min_purity:
                label = best
            else:
                label = OTHER
            idx = np.argsort(sign * u)[::-1][:n_top]
            rows.append({"component": j, "sign": sign, "label": label,
                         "seed_share": shares[best], "purity": purity,
                         "top_terms": ", ".join(terms[i] for i in idx if sign * u[i] > 0)})
    return pd.DataFrame(rows)


def svd_check(X: sparse.spmatrix, k: int) -> np.ndarray:
    """Singular values from ARPACK (scipy.sparse.linalg.svds), descending, for cross-checks."""
    from scipy.sparse.linalg import svds

    s = svds(X.astype(float), k=k, return_singular_vectors=False)
    return np.sort(s)[::-1]


def topic_weights(model: LSAModel, issues: list[str] | None = None,
                  power: float = 2.0) -> pd.DataFrame:
    """Documents x issue loading weights: positive part of each labelled pole, summed by issue.

    A document's weight on an issue is how far its (cosine-normalised) projection
    points along the poles carrying that issue's label; negative projections
    contribute nothing, so each pole is a one-sided topic. Raising the loading to
    ``power`` (2 = squared cosine, the share of the document's energy on that pole)
    keeps the small loadings that shared campaign vocabulary creates on every
    pole from leaking one issue's tone into another's.
    """
    issues = issues or list(config.TOPICS)
    W = pd.DataFrame(0.0, index=range(model.doc_scores.shape[0]), columns=issues)
    for _, p in model.poles.iterrows():
        if p["label"] in issues:
            W[p["label"]] += np.maximum(p["sign"] * model.doc_scores[:, p["component"]],
                                        0) ** power
    return W


def dominant_topic(W: pd.DataFrame, min_weight: float = 0.01) -> pd.Series:
    best = W.idxmax(axis=1)
    return best.where(W.max(axis=1) >= min_weight, OTHER)


# ---------------------------------------------------------------------------
# Daily projection, salience and sentiment
# ---------------------------------------------------------------------------
@dataclass
class DailyLSA:
    beta: pd.DataFrame        # date x component: projection of the day's document onto U_k
    salience: pd.DataFrame    # date x issue: share of issue loading in that day's projection
    sentiment: pd.DataFrame   # date x issue: S_k,t = (Pos - Neg) / Total, loading-weighted
    mass: pd.DataFrame        # date x issue: Total_k,t (sum of loading weights)
    n_docs: pd.Series


def daily_projection(model: LSAModel, texts: pd.Series, created_at: pd.Series,
                     calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """beta_t = v_t U_k: the day's summed TF-IDF vector projected on the term loadings."""
    X = model.vectorizer.transform(texts.fillna("").astype(str))
    day = to_trading_day(created_at, calendar)
    ok = day.notna().values
    days = pd.DatetimeIndex(day[ok])
    codes, uniq = pd.factorize(days)
    G = sparse.csr_matrix((np.ones(ok.sum()), (codes, np.arange(ok.sum()))),
                          shape=(len(uniq), ok.sum()))
    V = G @ X[np.where(ok)[0]]                      # day x terms
    beta = pd.DataFrame(V @ model.U, index=uniq).sort_index().reindex(calendar)
    beta.index.name = "date"
    return beta


def daily_sentiment(model: LSAModel, docs: pd.DataFrame, calendar: pd.DatetimeIndex,
                    scorer: ToneScorer | None = None) -> DailyLSA:
    """Loading-weighted, topic-conditioned sentiment and salience per trading day.

    ``docs`` needs ``text`` and ``created_at`` and must be the documents the model
    was fitted on, in the same order. Each document is scored with the lexicon of
    its dominant issue, then contributes to every issue in proportion to its
    loading on that issue:  Pos_k,t = sum_d w_dk 1[pol_d > 0], etc.
    """
    scorer = scorer or ToneScorer()
    W = topic_weights(model)
    dom = dominant_topic(W)
    pol = np.array([scorer.polarity(scorer.compound(t, k))
                    for t, k in zip(docs["text"].values, dom.values)])
    day = to_trading_day(docs["created_at"].reset_index(drop=True), calendar)
    W.index = range(len(W))
    pos = W.mul(pol > 0, axis=0).groupby(day.values).sum()
    neg = W.mul(pol < 0, axis=0).groupby(day.values).sum()
    tot = W.groupby(day.values).sum()
    sent = ((pos - neg) / tot.replace(0, np.nan)).reindex(calendar)
    sal = tot.div(tot.sum(axis=1).replace(0, np.nan), axis=0).reindex(calendar)
    n = pd.Series(1, index=range(len(W))).groupby(day.values).sum().reindex(calendar, fill_value=0)
    beta = daily_projection(model, docs["text"], docs["created_at"], calendar)
    for f in (sent, sal):
        f.index.name = "date"
    return DailyLSA(beta, sal, sent, tot.reindex(calendar), n)


# ---------------------------------------------------------------------------
# SVD of the signal-return cross-covariance
# ---------------------------------------------------------------------------
@dataclass
class CovarianceSVD:
    U: pd.DataFrame            # signals x k: signal combinations (the "alpha factors")
    S: np.ndarray              # singular values
    V: pd.DataFrame            # targets x k: the target portfolios each factor predicts
    p_values: np.ndarray       # permutation p-value of each singular value
    factors: pd.DataFrame      # date x k: f_t = s_t U_k, available for every date
    train_end: pd.Timestamp
    lag: int


def covariance_svd(signals: pd.DataFrame, targets: pd.DataFrame, k: int = 3, lag: int = 1,
                   train_frac: float = 0.6, n_perm: int = 500,
                   seed: int = config.SEED) -> CovarianceSVD:
    """C = (1/T) sum_t s_{t-lag}' dy_t = U S V^T, estimated on the training window only.

    Both blocks are standardised with training-window moments, so C is a
    cross-correlation matrix and no future information enters the factors used
    later out of sample. Pairing s_{t-lag} with dy_t makes the factors
    predictive rather than contemporaneous. Each singular value is compared with
    its distribution when the signal dates are circularly shifted (which keeps
    the autocorrelation of both blocks and breaks only their alignment).
    """
    s = signals.shift(lag)
    j = pd.concat([s, targets], axis=1, keys=["s", "y"]).dropna()
    n_train = int(len(j) * train_frac)
    tr = j.iloc[:n_train]
    mu_s, sd_s = tr["s"].mean(), tr["s"].std().replace(0, 1)
    mu_y, sd_y = tr["y"].mean(), tr["y"].std().replace(0, 1)
    Zs, Zy = ((tr["s"] - mu_s) / sd_s).values, ((tr["y"] - mu_y) / sd_y).values
    C = Zs.T @ Zy / len(tr)
    Uc, S, Vt = np.linalg.svd(C, full_matrices=False)
    k = int(min(k, len(S)))
    rng = np.random.default_rng(seed)
    exceed = np.zeros(len(S))
    for _ in range(n_perm):
        shift = rng.integers(5, len(Zs) - 5)
        Sp = np.linalg.svd(np.roll(Zs, shift, axis=0).T @ Zy / len(tr), compute_uv=False)
        exceed += Sp >= S
    pvals = (exceed + 1) / (n_perm + 1)
    U = pd.DataFrame(Uc[:, :k], index=signals.columns, columns=[f"F{i + 1}" for i in range(k)])
    V = pd.DataFrame(Vt[:k].T, index=targets.columns, columns=U.columns)
    # orient: largest-|loading| target positive
    for c in U.columns:
        sgn = np.sign(V[c].iloc[V[c].abs().values.argmax()]) or 1
        U[c] *= sgn
        V[c] *= sgn
    factors = ((signals - mu_s) / sd_s) @ U
    return CovarianceSVD(U, S[:k], V, pvals[:k], factors, tr.index[-1], lag)
