"""Data-quality audit for the market panel and the prediction-market series.

Every check returns a small frame so the audit can be read as a report
(``audit_data.py`` writes DATA_AUDIT.md) and asserted on in tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def ohlc_violations(ohlc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Bars where Low > min(Open, Close), High < max(Open, Close), or a price <= 0."""
    rows = []
    for t, d in ohlc.items():
        lo_bad = d["Low"] > d[["Open", "Close"]].min(axis=1) * (1 + 1e-6)
        hi_bad = d["High"] < d[["Open", "Close"]].max(axis=1) * (1 - 1e-6)
        nonpos = (d[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
        for date in d.index[lo_bad | hi_bad | nonpos]:
            rows.append({"ticker": t, "date": date.date(), "low_bad": bool(lo_bad[date]),
                         "high_bad": bool(hi_bad[date]), "nonpositive": bool(nonpos[date])})
    return pd.DataFrame(rows, columns=["ticker", "date", "low_bad", "high_bad", "nonpositive"])


def return_outliers(returns: pd.DataFrame, z: float = 5.0) -> pd.DataFrame:
    """Returns more than ``z`` robust (MAD) standard deviations from the median."""
    rows = []
    for c in returns.columns:
        r = returns[c].dropna()
        mad = 1.4826 * (r - r.median()).abs().median()
        if mad == 0:
            continue
        zs = (r - r.median()) / mad
        for d in zs.index[zs.abs() > z]:
            rows.append({"series": c, "date": d.date(), "return": r[d], "robust_z": zs[d]})
    return pd.DataFrame(rows, columns=["series", "date", "return", "robust_z"])


def stale_prices(close: pd.DataFrame, run: int = 2) -> pd.DataFrame:
    """Series with ``run`` or more consecutive identical closes."""
    rows = []
    for c in close.columns:
        s = close[c].dropna()
        same = s.diff().eq(0)
        n = int(same.sum())
        if n >= run - 1 and n:
            rows.append({"series": c, "unchanged_days": n,
                         "dates": ", ".join(str(d.date()) for d in s.index[same][:5])})
    return pd.DataFrame(rows, columns=["series", "unchanged_days", "dates"])


def cross_source_returns(primary: dict[str, pd.DataFrame], secondary: dict[str, pd.DataFrame],
                         tol: float = 0.003) -> pd.DataFrame:
    """Compare daily close-to-close returns between two vendors (e.g. Yahoo vs Stooq).

    Differences on ex-dividend days are expected when one source is adjusted and
    the other is not; anything else above ``tol`` is a data error in one feed.
    """
    rows = []
    for t in primary:
        if t not in secondary:
            continue
        a = np.log(primary[t]["Close"]).diff()
        b = np.log(secondary[t]["Close"]).diff()
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        if j.empty:
            continue
        diff = (j["a"] - j["b"]).abs()
        rows.append({"ticker": t, "days": len(j), "corr": j["a"].corr(j["b"]),
                     "mean_abs_diff_bp": 1e4 * diff.mean(),
                     "days_over_tol": int((diff > tol).sum()),
                     "worst_day": str(diff.idxmax().date()), "worst_diff_bp": 1e4 * diff.max()})
    return pd.DataFrame(rows)


def fred_checks(fred: pd.DataFrame, jump_bp: float = 25) -> pd.DataFrame:
    rows = []
    for c in fred.columns:
        s = fred[c].dropna()
        jumps = s.diff().abs() * 100
        rows.append({"series": c, "obs": len(s), "min": s.min(), "max": s.max(),
                     f"days_jump_gt_{int(jump_bp)}bp": int((jumps > jump_bp).sum()),
                     "largest_jump_bp": jumps.max()})
    return pd.DataFrame(rows).set_index("series")


# ---------------------------------------------------------------------------
# Prediction markets
# ---------------------------------------------------------------------------
def quote_staleness(long: pd.DataFrame) -> pd.DataFrame:
    """Hours between the quote used and that day's 16:00 ET close, per venue and chamber."""
    d = long.dropna(subset=["prob_dem", "quote_ts"]).copy()
    close = (d["date"] + pd.Timedelta(hours=config.MARKET_CLOSE_HOUR)) \
        .dt.tz_localize(config.TZ).dt.tz_convert("UTC")
    d["age_h"] = (close - pd.to_datetime(d["quote_ts"], utc=True)).dt.total_seconds() / 3600
    g = d.groupby(["venue", "chamber"])["age_h"]
    return pd.DataFrame({"days": g.size(), "median_age_h": g.median(), "p95_age_h": g.quantile(.95),
                         "max_age_h": g.max(), "days_older_than_24h": g.apply(lambda x: (x > 24).sum())})


def venue_agreement(long: pd.DataFrame, a: str = "polymarket", b: str = "kalshi",
                    big_gap: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Level and change agreement between two venues; list days where they disagree."""
    wide = long.pivot_table(index="date", columns=["chamber", "venue"], values="prob_dem")
    summ, gaps = [], []
    for ch in wide.columns.get_level_values(0).unique():
        if a not in wide[ch] or b not in wide[ch]:
            continue
        x, y = wide[ch][a], wide[ch][b]
        j = pd.concat([x.rename(a), y.rename(b)], axis=1).dropna()
        dj = j.diff().dropna()
        gap = (j[a] - j[b]).abs()
        summ.append({"chamber": ch, "days": len(j), "level_corr": j[a].corr(j[b]),
                     "change_corr": dj[a].corr(dj[b]), "mean_abs_gap_pp": 100 * gap.mean(),
                     "max_gap_pp": 100 * gap.max(), f"days_gap_gt_{int(100 * big_gap)}pp":
                     int((gap > big_gap).sum())})
        for d in gap.index[gap > big_gap]:
            gaps.append({"chamber": ch, "date": d.date(), a: j.loc[d, a], b: j.loc[d, b]})
    return pd.DataFrame(summ).set_index("chamber"), pd.DataFrame(gaps)


def large_moves(long: pd.DataFrame, consensus: pd.DataFrame, threshold: float = 0.04) -> pd.DataFrame:
    """Consensus moves above ``threshold`` and whether each venue moved the same way."""
    wide = long.pivot_table(index="date", columns=["chamber", "venue"], values="prob_dem")
    rows = []
    for ch in consensus.columns:
        dc = consensus[ch].diff()
        for d in dc.index[dc.abs() > threshold]:
            row = {"chamber": ch, "date": d.date(), "dProb_pp": 100 * dc[d]}
            for v in wide[ch].columns:
                dv = wide[ch][v].diff().get(d, np.nan)
                row[f"d_{v}_pp"] = 100 * dv
            rows.append(row)
    return pd.DataFrame(rows)
