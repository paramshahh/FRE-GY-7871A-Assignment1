"""Analysis layer: tone measures, the trading calendar, event windows, filters.

Everything in this module is written for Assignment 1 and is deliberately kept
out of the notebook so the notebook can stay readable. The pipeline modules
(`edgar`, `parse`, `lexicons`, `market`) are the instructor's and are unchanged.

Conventions used throughout, stated once here:

*   A **window [a, b]** in trading-day offsets relative to day 0 is a
    buy-and-hold return measured from the close on day ``a - 1`` to the close on
    day ``b``. The brief specifies the [0, +3] return as "measured from the close
    on day -1", and the same convention is applied to the [-60, -6] control
    return so the two are comparable.
*   **Realised volatility** over [a, b] is the sample standard deviation of the
    daily simple returns on days a..b inclusive, annualised by sqrt(252). The
    return on day ``a`` uses the close on day ``a - 1``.
*   Lexicon membership is checked on uppercase tokens, which is what
    ``src.parse.tokenize`` emits and the format the LM dictionary ships in.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
MARKET_CLOSE_HOUR = 16          # 16:00 America/New_York
EASTERN = "America/New_York"


# ---------------------------------------------------------------------------
# 1. Tone measures
# ---------------------------------------------------------------------------
def doc_frequency(
    counts_iter: Iterable[Mapping[str, int]],
    word_list: set[str],
) -> tuple[Counter, int]:
    """Document frequency of each lexicon word, and N, over ONE corpus.

    ``df(word)`` is the number of documents in which the word appears at least
    once. ``N`` is the number of documents. Both must be computed on the same
    corpus that is later scored and later regressed -- mixing corpora between
    the tf.idf step and the regression step is explicitly penalised.

    Only lexicon words are counted; general vocabulary never enters df.
    """
    df: Counter = Counter()
    n_docs = 0
    for counts in counts_iter:
        n_docs += 1
        df.update(w for w in word_list if counts.get(w, 0) > 0)
    return df, n_docs


def proportional_score(
    counts: Mapping[str, int],
    n_words: int,
    word_list: set[str],
) -> float:
    """Share of the document's words that are on the list. LM's simple weighting."""
    if not n_words:
        return np.nan
    hits = sum(c for w, c in counts.items() if w in word_list)
    return hits / n_words


def tfidf_score(
    counts: Mapping[str, int],
    n_words: int,
    n_distinct: int,
    word_list: set[str],
    doc_freq: Mapping[str, int],
    n_docs: int,
) -> float:
    """Loughran-McDonald equation (1), summed over the lexicon words present.

        tfidf(w, j) = [(1 + ln tf(w, j)) / (1 + ln a_j)] * ln(N / df(w))

    ``a_j`` is the average word count *within document j*: total words over
    distinct words. It is a per-document quantity, not a corpus-wide one.

    Words with tf = 0 are skipped (no ln 0). Words with df = 0 cannot occur --
    a word present in this document has df >= 1 by construction -- but words
    with df = N are kept and contribute exactly 0, which is the correct
    behaviour: a word in every filing carries no cross-sectional information.

    All logs are natural. Using log base 10 gives 0.368 for the d1 self-check
    instead of 0.8480.
    """
    if not n_words or not n_distinct:
        return np.nan
    a_j = n_words / n_distinct
    denom = 1.0 + np.log(a_j)
    if denom <= 0:
        return np.nan
    total = 0.0
    for word in word_list:
        tf = counts.get(word, 0)
        if tf <= 0:
            continue
        df = doc_freq.get(word, 0)
        if df <= 0:
            continue
        total += ((1.0 + np.log(tf)) / denom) * np.log(n_docs / df)
    return total


def selfcheck_tfidf() -> pd.DataFrame:
    """Reproduce the three-document worked example from the README.

    Word list is {LOSS, RISK}. GAIN is a distractor: it is in every document and
    affects a_j and the denominator of the proportional measure, but is never
    itself scored. Expected tf.idf: 0.8480, 0.2885, 0.5026.
    """
    docs = {
        "d1": "LOSS LOSS RISK GAIN".split(),
        "d2": "LOSS GAIN GAIN".split(),
        "d3": "RISK RISK RISK GAIN".split(),
    }
    word_list = {"LOSS", "RISK"}
    counts = {k: Counter(v) for k, v in docs.items()}
    df, n_docs = doc_frequency(counts.values(), word_list)

    rows = []
    for name, toks in docs.items():
        c = counts[name]
        n_words, n_distinct = len(toks), len(set(toks))
        rows.append({
            "doc": name,
            "words": " ".join(toks),
            "n_words": n_words,
            "n_distinct": n_distinct,
            "a_j": n_words / n_distinct,
            "proportional": proportional_score(c, n_words, word_list),
            "tfidf": tfidf_score(c, n_words, n_distinct, word_list, df, n_docs),
        })
    out = pd.DataFrame(rows).set_index("doc")
    out.attrs["doc_freq"] = dict(df)
    out.attrs["N"] = n_docs
    return out


# ---------------------------------------------------------------------------
# 2. Trading calendar and the day-0 rule
# ---------------------------------------------------------------------------
def trading_calendar(prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DatetimeIndex:
    """The trading days actually present in the downloaded benchmark series.

    The brief asks for the calendar to come from the data rather than an
    external market-calendar package, so holidays and early closes are whatever
    the SPY price history says they are.
    """
    if benchmark not in prices.columns:
        raise KeyError(f"{benchmark} not in prices; have {list(prices.columns)[:10]}")
    idx = prices.index[prices[benchmark].notna()]
    return pd.DatetimeIndex(sorted(pd.DatetimeIndex(idx).normalize().unique()))


def effective_filing_date(
    filing_date: pd.Timestamp,
    acceptance_datetime: pd.Timestamp,
) -> pd.Timestamp:
    """The later of the filing date and the (close-adjusted) acceptance date.

    ``acceptance_datetime`` is UTC. It is converted to America/New_York first;
    an acceptance at or after 16:00 Eastern is not tradable that day, so the
    acceptance date is pushed forward one calendar day. Filings with no
    acceptance timestamp fall back to the filing date alone.
    """
    fdate = pd.Timestamp(filing_date).normalize()
    if acceptance_datetime is None or pd.isna(acceptance_datetime):
        return fdate
    acc = pd.Timestamp(acceptance_datetime)
    acc = acc.tz_localize("UTC") if acc.tzinfo is None else acc.tz_convert("UTC")
    acc_local = acc.tz_convert(EASTERN)
    acc_date = acc_local.normalize().tz_localize(None)
    if acc_local.hour >= MARKET_CLOSE_HOUR:
        acc_date += pd.Timedelta(days=1)
    return max(fdate, acc_date)


def next_trading_day(date: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp | None:
    """First trading day on or after ``date``; None if past the end of the data."""
    pos = calendar.searchsorted(pd.Timestamp(date).normalize(), side="left")
    if pos >= len(calendar):
        return None
    return calendar[pos]


def assign_day0(meta: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Add day-0 columns, and the naive filing-date-only comparison.

    Returns the frame with:
        effective_date    the later of filing date and adjusted acceptance date
        day0              first trading day on/after effective_date
        day0_naive        first trading day on/after the filing date alone
        day0_moved        True where the acceptance rule moved day 0
        acc_after_close   True where acceptance was at/after 16:00 Eastern
    """
    out = meta.copy()
    eff, naive, after_close = [], [], []
    for fdate, acc in zip(out["filing_date"], out["acceptance_datetime"]):
        eff.append(effective_filing_date(fdate, acc))
        naive.append(pd.Timestamp(fdate).normalize())
        if acc is None or pd.isna(acc):
            after_close.append(False)
        else:
            a = pd.Timestamp(acc)
            a = a.tz_localize("UTC") if a.tzinfo is None else a.tz_convert("UTC")
            after_close.append(a.tz_convert(EASTERN).hour >= MARKET_CLOSE_HOUR)
    out["effective_date"] = pd.DatetimeIndex(eff)
    out["acc_after_close"] = after_close
    out["day0"] = [next_trading_day(d, calendar) for d in eff]
    out["day0_naive"] = [next_trading_day(d, calendar) for d in naive]
    out["day0_moved"] = out["day0"] != out["day0_naive"]
    return out


# ---------------------------------------------------------------------------
# 3. Event-window returns and realised volatility
# ---------------------------------------------------------------------------
class PricePanel:
    """Positional access to one price/volume panel indexed by trading day.

    Windows are expressed as integer offsets from day 0, so everything reduces
    to integer slicing of a per-ticker array once day 0 is located in the
    calendar. Tickers with no price history are reported as missing rather than
    silently dropped.
    """

    def __init__(self, prices: pd.DataFrame, calendar: pd.DatetimeIndex,
                 volume: pd.DataFrame | None = None):
        self.calendar = calendar
        self.prices = prices.reindex(calendar)
        self.volume = volume.reindex(calendar) if volume is not None else None
        self._pos = {d: i for i, d in enumerate(calendar)}

    def position(self, day0: pd.Timestamp) -> int | None:
        return self._pos.get(pd.Timestamp(day0).normalize())

    def _series(self, ticker: str) -> np.ndarray | None:
        if ticker not in self.prices.columns:
            return None
        arr = self.prices[ticker].to_numpy(dtype=float)
        return None if np.all(np.isnan(arr)) else arr

    def price_at(self, ticker: str, day0: pd.Timestamp, offset: int) -> float:
        arr = self._series(ticker)
        i = self.position(day0)
        if arr is None or i is None:
            return np.nan
        j = i + offset
        if j < 0 or j >= len(arr):
            return np.nan
        return arr[j]

    def n_days_available(self, ticker: str, day0: pd.Timestamp) -> tuple[int, int]:
        """Count of non-missing daily returns strictly before / after day 0.

        Used by filter 5. A return on day t needs prices on t and t-1, so the
        count before day 0 is the number of usable consecutive price pairs in
        the history preceding day 0.
        """
        arr = self._series(ticker)
        i = self.position(day0)
        if arr is None or i is None:
            return 0, 0
        before = arr[:i]
        after = arr[i + 1:]
        n_before = int(np.sum(~np.isnan(before[1:]) & ~np.isnan(before[:-1]))) if len(before) > 1 else 0
        n_after = int(np.sum(~np.isnan(after[1:]) & ~np.isnan(after[:-1]))) if len(after) > 1 else 0
        return n_before, n_after

    def window_return(self, ticker: str, day0: pd.Timestamp, a: int, b: int) -> float:
        """Buy-and-hold return over [a, b], measured from the close on day a-1."""
        p0 = self.price_at(ticker, day0, a - 1)
        p1 = self.price_at(ticker, day0, b)
        if not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0:
            return np.nan
        return p1 / p0 - 1.0

    def excess_return(self, ticker: str, day0: pd.Timestamp, a: int, b: int,
                      benchmark: str = "SPY") -> float:
        r = self.window_return(ticker, day0, a, b)
        m = self.window_return(benchmark, day0, a, b)
        if not np.isfinite(r) or not np.isfinite(m):
            return np.nan
        return r - m

    def realised_vol(self, ticker: str, day0: pd.Timestamp, a: int, b: int,
                     min_obs: int = 20) -> float:
        """Annualised sd of daily simple returns on days a..b inclusive."""
        arr = self._series(ticker)
        i = self.position(day0)
        if arr is None or i is None:
            return np.nan
        lo, hi = i + a - 1, i + b
        if lo < 0 or hi >= len(arr):
            return np.nan
        seg = arr[lo:hi + 1]
        rets = seg[1:] / seg[:-1] - 1.0
        rets = rets[np.isfinite(rets)]
        if len(rets) < min_obs:
            return np.nan
        return float(np.std(rets, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    def mean_dollar_volume(self, ticker: str, day0: pd.Timestamp, a: int, b: int) -> float:
        """Average daily price x volume over [a, b], for the liquidity control."""
        if self.volume is None or ticker not in self.volume.columns:
            return np.nan
        i = self.position(day0)
        if i is None:
            return np.nan
        lo, hi = max(i + a, 0), i + b
        if hi >= len(self.calendar) or lo > hi:
            return np.nan
        p = self.prices[ticker].to_numpy(dtype=float)[lo:hi + 1]
        v = self.volume[ticker].to_numpy(dtype=float)[lo:hi + 1]
        dv = p * v
        dv = dv[np.isfinite(dv)]
        return float(np.mean(dv)) if len(dv) else np.nan


# ---------------------------------------------------------------------------
# 4. Filter waterfall
# ---------------------------------------------------------------------------
class Waterfall:
    """Running record of a filter sequence, for Table 1.

    Every filter that is applied gets a row. A filter applied but not reported
    is explicitly penalised by the brief, so the intended usage is that the
    only way to drop rows is through ``.apply``.
    """

    def __init__(self, label: str, n_start: int, note: str = ""):
        self.rows = [{"step": label, "removed": np.nan, "remaining": n_start, "note": note}]

    def apply(self, df: pd.DataFrame, mask: pd.Series, label: str,
              note: str = "") -> pd.DataFrame:
        """Keep rows where ``mask`` is True; record how many that removed."""
        mask = mask.fillna(False).astype(bool)
        kept = df[mask]
        self.rows.append({
            "step": label,
            "removed": int(len(df) - len(kept)),
            "remaining": int(len(kept)),
            "note": note,
        })
        return kept

    def record(self, label: str, removed, remaining, note: str = "") -> None:
        """Record a step computed elsewhere (e.g. the universe-level counts)."""
        self.rows.append({"step": label, "removed": removed,
                          "remaining": remaining, "note": note})

    def to_frame(self) -> pd.DataFrame:
        out = pd.DataFrame(self.rows)
        return out[["step", "removed", "remaining", "note"]]


def one_per_firm_quarter(df: pd.DataFrame,
                         firm_col: str = "cik",
                         date_col: str = "filing_date",
                         quarter_col: str = "cal_quarter") -> pd.Series:
    """Boolean mask keeping the earliest filing per firm per calendar quarter.

    Ties on filing date are broken by accession number so the choice is
    deterministic and reproducible; the brief says keep the earliest filing date
    and drop the others rather than averaging them.
    """
    order = df.sort_values([firm_col, quarter_col, date_col, "accession"])
    keep_idx = order.groupby([firm_col, quarter_col], sort=False).head(1).index
    return df.index.isin(keep_idx)


# ---------------------------------------------------------------------------
# 5. Regression helpers
# ---------------------------------------------------------------------------
def newey_west_trend(y: pd.Series, lags: int | None = None) -> dict:
    """Regress a quarterly series on a time trend; OLS and Newey-West t-stats.

    ~20 quarterly observations of a persistent series regressed on time will
    produce a large t-statistic whether or not anything is happening, because
    the residuals are autocorrelated. Newey-West is required here, not optional.
    Default lag truncation is the usual 4*(T/100)^(2/9) rule.
    """
    import statsmodels.api as sm

    y = y.dropna()
    t = np.arange(len(y), dtype=float)
    X = sm.add_constant(t)
    ols = sm.OLS(y.to_numpy(dtype=float), X).fit()
    if lags is None:
        lags = int(np.floor(4 * (len(y) / 100.0) ** (2.0 / 9.0)))
    nw = sm.OLS(y.to_numpy(dtype=float), X).fit(
        cov_type="HAC", cov_kwds={"maxlags": max(lags, 1)}
    )
    return {
        "n_obs": int(len(y)),
        "slope_per_quarter": float(ols.params[1]),
        "slope_per_year": float(ols.params[1] * 4),
        "t_ols": float(ols.tvalues[1]),
        "t_nw": float(nw.tvalues[1]),
        "nw_lags": int(max(lags, 1)),
        "r2": float(ols.rsquared),
    }


def cluster_ols(formula: str, data: pd.DataFrame, cluster_col: str):
    """OLS with standard errors clustered on ``cluster_col`` (firm, by default).

    Clustering on the firm is the relevant choice here: the residuals of
    repeated filings by the same company are correlated, and the panel is wide
    in firms relative to its length in quarters.
    """
    import statsmodels.formula.api as smf

    sub = data.dropna(subset=[c for c in [cluster_col] if c in data.columns])
    model = smf.ols(formula, data=sub)
    used = sub.loc[model.data.row_labels]
    return model.fit(cov_type="cluster",
                     cov_kwds={"groups": used[cluster_col].astype(str)})
