"""Election prediction markets: Polymarket, Kalshi and PredictIt.

Every venue is reduced to the same long table

    date | venue | chamber | prob_dem

where ``prob_dem`` is the implied probability that Democrats control the
chamber after the 3 Nov 2026 midterms, with the overround removed across the
D and R contracts. A weighted consensus is built per chamber and trading day,
and its first difference ``dProb`` is what the econometrics consume.

Contracts (identified Oct 2026; override in ``config`` if a venue relists):

* Polymarket: binary markets "Will the Democratic/Republican Party control the
  House|Senate after the 2026 Midterm elections?". Located through Gamma
  ``/public-search``; daily history from CLOB ``/prices-history`` keyed on the
  YES token id.
* Kalshi: event ``CONTROLH-2026`` (markets ``CONTROLH-2026-D``/``-R``) and
  ``CONTROLS-2026`` (``CONTROLS-2026-D``/``-R``). Daily candles from
  ``/series/{series}/markets/{ticker}/candlesticks?period_interval=1440``.
* PredictIt: markets 8157 (House) and 8155 (Senate). The documented API is a
  snapshot (``/api/marketdata/markets/{id}``); daily OHLC comes from the chart
  CSV behind the market page (``/Resource/DownloadMarketChartData``, 90 days
  max). Each run saves the CSV under ``data/raw/predictit/`` so history
  accumulates across runs, and appends the live snapshot to a log.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config
from .data_loader import to_trading_day, trading_calendar

log = logging.getLogger(__name__)

POLY_GAMMA = "https://gamma-api.polymarket.com"
POLY_CLOB = "https://clob.polymarket.com"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
PREDICTIT_MARKET = "https://www.predictit.org/api/marketdata/markets/{id}"
PREDICTIT_CSV = "https://www.predictit.org/Resource/DownloadMarketChartData"
PREDICTIT_DIR = config.RAW_DIR / "predictit"


def party_of(label: str) -> str | None:
    """'Democratic Party' -> 'D'; 'Republicans' -> 'R'; anything else -> None."""
    s = f" {(label or '').lower()} "
    if re.search(r"\b(democrat\w*|dem)\b", s):
        return "D"
    if re.search(r"\b(republican\w*|gop|rep)\b", s):
        return "R"
    return None


def chamber_of(text: str) -> str | None:
    s = (text or "").lower()
    if "house" in s:
        return "House"
    if "senate" in s:
        return "Senate"
    return None


def remove_overround(prices: dict[str, float]) -> dict[str, float]:
    """Normalise outcome prices to probabilities that sum to one."""
    tot = sum(v for v in prices.values() if v is not None and np.isfinite(v))
    if tot <= 0:
        return {k: np.nan for k in prices}
    return {k: v / tot for k, v in prices.items()}


def _get(session: requests.Session, url: str, params: dict | None = None,
         retries: int = 5, backoff: float = 2.0):
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 10 * (i + 1))))
                raise requests.HTTPError("rate limited", response=r)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if i == retries - 1:
                raise
            time.sleep(backoff ** (i + 1))
    raise RuntimeError("unreachable")


def _two_party_prob(series: dict[str, pd.Series]) -> pd.Series:
    """P(D) normalised over the D and R contracts (D alone if R is absent)."""
    if "D" not in series or series["D"].empty:
        raise LookupError("No Democratic contract found")
    # D and R quotes are stamped independently: carry each side's last price forward
    df = pd.DataFrame(series).sort_index().ffill()
    if "R" in df and df["R"].notna().any():
        return (df["D"] / (df["D"] + df["R"])).dropna()
    return df["D"].dropna()


# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------
def _as_list(x):
    return json.loads(x) if isinstance(x, str) else (x or [])


def find_polymarket_tokens(search_payload: dict, chamber: str) -> dict[str, str]:
    """Party -> YES token id for the 2026 control markets of one chamber."""
    tokens = {}
    for ev in search_payload.get("events", []) or []:
        for m in ev.get("markets", []) or []:
            q = m.get("question", "")
            if "2026" not in q or "control" not in q.lower() or chamber_of(q) != chamber:
                continue
            party = party_of(q)
            ids = _as_list(m.get("clobTokenIds"))
            outcomes = [o.lower() for o in _as_list(m.get("outcomes"))]
            if party and ids:
                yes = outcomes.index("yes") if "yes" in outcomes else 0
                tokens.setdefault(party, ids[yes])
    return tokens


def parse_polymarket_history(payload: dict) -> pd.Series:
    hist = payload.get("history", []) or []
    if not hist:
        return pd.Series(dtype=float)
    s = pd.Series({pd.to_datetime(h["t"], unit="s", utc=True): float(h["p"]) for h in hist})
    return s.sort_index()


def _windows(start: str, end: str, days: int) -> list[tuple[int, int]]:
    """[start, end + 1 day) split into consecutive windows of at most ``days`` days (unix s)."""
    lo = pd.Timestamp(start, tz=config.TZ).tz_convert("UTC")
    hi = (pd.Timestamp(end, tz=config.TZ) + pd.Timedelta(days=1)).tz_convert("UTC")
    out = []
    while lo < hi:
        nxt = min(lo + pd.Timedelta(days=days), hi)
        out.append((int(lo.timestamp()), int(nxt.timestamp())))
        lo = nxt
    return out


def fetch_polymarket(chamber: str, session: requests.Session, start: str = config.START,
                     end: str = config.END) -> pd.Series:
    """Hourly P(Dem) from the CLOB price history (windows of 14 days, the API maximum)."""
    payload = _get(session, f"{POLY_GAMMA}/public-search",
                   {"q": f"control the {chamber} after the 2026 Midterm elections"})
    tokens = find_polymarket_tokens(payload, chamber)
    if not tokens:
        raise LookupError(f"Polymarket {chamber} control markets not found")
    series = {}
    for party, tok in tokens.items():
        parts = []
        for lo, hi in _windows(start, end, 14):
            h = parse_polymarket_history(_get(session, f"{POLY_CLOB}/prices-history",
                                              {"market": tok, "startTs": lo, "endTs": hi,
                                               "fidelity": 60}))
            # the API appends the latest point to every window: keep only in-window points
            parts.append(h[(h.index >= pd.Timestamp(lo, unit="s", tz="UTC"))
                           & (h.index < pd.Timestamp(hi, unit="s", tz="UTC"))])
        s = pd.concat(parts).sort_index()
        series[party] = s[~s.index.duplicated(keep="last")]
    return _two_party_prob(series)


# ---------------------------------------------------------------------------
# Kalshi
# ---------------------------------------------------------------------------
def _num(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _close_dollars(field: dict | None) -> float | None:
    """Kalshi candle field -> close in dollars (``close_dollars`` or ``close`` in cents)."""
    field = field or {}
    v = _num(field.get("close_dollars"))
    if v is None and _num(field.get("close")) is not None:
        v = _num(field.get("close")) / 100.0
    return v


def parse_kalshi_candles(payload: dict, max_spread: float = 0.05) -> pd.Series:
    """YES price at the end of each candle, stamped one second before the bar closes.

    The bid/ask midpoint is used when the spread is at most ``max_spread`` (it is
    the price at the bar's close); otherwise, or with no two-sided quote, the
    last trade.
    """
    rows = {}
    for c in payload.get("candlesticks", []) or []:
        ts = pd.to_datetime(c["end_period_ts"] - 1, unit="s", utc=True)
        b, a = _close_dollars(c.get("yes_bid")), _close_dollars(c.get("yes_ask"))
        if a is not None and b is not None and 0 <= a - b <= max_spread and a > 0:
            px = (a + b) / 2
        else:
            px = _close_dollars(c.get("price"))
        if px is not None:
            rows[ts] = px
    return pd.Series(rows, dtype=float).sort_index()


def fetch_kalshi(chamber: str, session: requests.Session, start: str = config.START,
                 end: str = config.END) -> pd.Series:
    """Hourly P(Dem) from Kalshi candlesticks, requested ~100 days (< 5,000 candles) at a time."""
    series_ticker = config.KALSHI_SERIES[chamber]
    series = {}
    for party in ("D", "R"):
        ticker = f"{config.KALSHI_EVENTS[chamber]}-{party}"
        parts = []
        for lo, hi in _windows(start, end, 100):
            payload = _get(session,
                           f"{KALSHI}/series/{series_ticker}/markets/{ticker}/candlesticks",
                           {"start_ts": lo, "end_ts": hi, "period_interval": 60})
            parts.append(parse_kalshi_candles(payload))
            time.sleep(2.0)  # public tier rate limit
        s = pd.concat(parts).sort_index()
        series[party] = s[~s.index.duplicated(keep="last")]
    return _two_party_prob(series)


# ---------------------------------------------------------------------------
# PredictIt
# ---------------------------------------------------------------------------
def parse_predictit_snapshot(payload: dict) -> dict[str, float]:
    """Party -> last trade price from a PredictIt market snapshot."""
    out = {}
    for c in payload.get("contracts", []) or []:
        party = party_of(c.get("name") or c.get("shortName", ""))
        if party and c.get("lastTradePrice") is not None:
            out[party] = float(c["lastTradePrice"])
    return out


def update_predictit(chamber: str, session: requests.Session) -> None:
    """Save today's 90-day chart CSV and append the live snapshot to the log."""
    PREDICTIT_DIR.mkdir(parents=True, exist_ok=True)
    mid = config.PREDICTIT_MARKET_IDS[chamber]
    r = session.get(PREDICTIT_CSV, params={"marketid": mid, "timespan": "90d"}, timeout=30)
    r.raise_for_status()
    if r.text.startswith("ContractName"):
        stamp = pd.Timestamp.now(tz=config.TZ).strftime("%Y%m%d")
        (PREDICTIT_DIR / f"{chamber.lower()}_{stamp}.csv").write_text(r.text, encoding="utf-8")
    probs = remove_overround(parse_predictit_snapshot(
        _get(session, PREDICTIT_MARKET.format(id=mid))))
    if "D" in probs:
        path = PREDICTIT_DIR / "snapshot_log.csv"
        pd.DataFrame([{"ts": pd.Timestamp.now(tz="UTC"), "chamber": chamber,
                       "prob_dem": probs["D"]}]).to_csv(
            path, mode="a", header=not path.exists(), index=False)


def parse_predictit_csv(path: Path) -> pd.DataFrame:
    """A PredictIt per-market CSV export -> date, party, close price."""
    df = pd.read_csv(path)
    cols = {c.lower().replace(" ", ""): c for c in df.columns}
    date_c = cols.get("date")
    px_c = cols.get("closeshareprice") or cols.get("close")
    name_c = cols.get("contractname") or cols.get("contract")
    if not (date_c and px_c and name_c):
        raise ValueError(f"Unrecognised PredictIt export: {path.name}")
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_c], format="mixed").dt.tz_localize(None),
        "party": df[name_c].map(party_of),
        "price": pd.to_numeric(df[px_c].astype(str).str.replace("$", "", regex=False),
                               errors="coerce"),
    })
    return out.dropna()


def load_predictit_history(chamber: str) -> pd.Series:
    """History from manual CSV exports (``predictit/<chamber>_*.csv``) plus the snapshot log."""
    parts = []
    for f in sorted(PREDICTIT_DIR.glob(f"{chamber.lower()}_*.csv")):
        df = parse_predictit_csv(f)
        wide = df.pivot_table(index="date", columns="party", values="price", aggfunc="last")
        if "D" in wide:
            s = wide["D"] / wide.sum(axis=1) if "R" in wide else wide["D"]
            # a daily bar closes at midnight ET, after the 16:00 equity close: stamp it at
            # 23:59 so it is information for the next session, never the same one
            s.index = (s.index.normalize() + pd.Timedelta(hours=23, minutes=59)) \
                .tz_localize(config.TZ).tz_convert("UTC")
            parts.append(s)
    log_path = PREDICTIT_DIR / "snapshot_log.csv"
    if log_path.exists():
        snap = pd.read_csv(log_path)
        snap = snap[snap["chamber"] == chamber]
        if len(snap):
            parts.append(pd.Series(snap["prob_dem"].values,
                                   index=pd.to_datetime(snap["ts"], utc=True)))
    if not parts:
        raise LookupError("No PredictIt history in data/raw/predictit/")
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="last")]


# ---------------------------------------------------------------------------
# Daily alignment and consensus
# ---------------------------------------------------------------------------
def to_daily(series: pd.Series, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Last quote strictly before each 16:00 ET close, with that quote's timestamp."""
    if series.empty:
        return pd.DataFrame({"prob_dem": np.nan,
                             "quote_ts": pd.Series(pd.NaT, index=calendar,
                                                   dtype="datetime64[ns, UTC]")}, index=calendar)
    s = series.dropna().sort_index()
    days = to_trading_day(pd.Series(s.index, index=s.index), calendar)
    df = pd.DataFrame({"prob_dem": s.values, "quote_ts": s.index, "date": days.values})
    return df.dropna(subset=["date"]).groupby("date")[["prob_dem", "quote_ts"]].last() \
        .reindex(calendar)


def consensus(long: pd.DataFrame, weights: dict[str, float] = config.VENUE_WEIGHTS) -> pd.DataFrame:
    """Weighted P(Dem) per chamber and day, weights renormalised over reporting venues."""
    wide = long.pivot_table(index="date", columns=["chamber", "venue"], values="prob_dem")
    out = {}
    for ch in wide.columns.get_level_values(0).unique():
        sub = wide[ch]
        w = pd.Series({v: weights.get(v, 0.0) for v in sub.columns})
        num = (sub.fillna(0) * w).sum(axis=1)
        den = (sub.notna() * w).sum(axis=1)
        out[ch] = num / den.replace(0, np.nan)
    df = pd.DataFrame(out)
    df.index.name = "date"
    return df


@dataclass
class MarketProbabilities:
    by_venue: pd.DataFrame   # long: date, venue, chamber, prob_dem
    consensus: pd.DataFrame  # date x chamber, P(Dem control)
    dprob: pd.DataFrame      # first differences, probability points
    coverage: pd.DataFrame   # venue x chamber: days with a quote
    errors: dict


CACHE = config.RAW_DIR / "prediction_markets_long.csv"


def load_prediction_markets(start: str = config.START, end: str = config.END,
                            refresh: bool = True, max_ffill: int = 2) -> MarketProbabilities:
    """Fetch every venue, cache the long table, and build the daily consensus.

    A venue that fails is skipped and its error recorded. If no venue returns
    data and nothing is cached, this raises: there is no simulated fallback.
    """
    cal = trading_calendar(start, end)
    errors: dict[str, str] = {}
    rows = []
    if refresh:
        s = requests.Session()
        s.headers["User-Agent"] = "nyu-fre7871-assignment5/1.0 (academic research)"
        for ch in config.CHAMBERS:
            try:
                update_predictit(ch, s)
            except Exception as exc:
                errors[f"predictit_download_{ch}"] = repr(exc)
        fetchers = {
            "polymarket": lambda ch: fetch_polymarket(ch, s, start, end),
            "kalshi": lambda ch: fetch_kalshi(ch, s, start, end),
            "predictit": load_predictit_history,
        }
        for venue, fn in fetchers.items():
            for ch in config.CHAMBERS:
                try:
                    daily = to_daily(fn(ch), cal)
                    rows.append(pd.DataFrame({"date": cal, "venue": venue, "chamber": ch,
                                              "prob_dem": daily["prob_dem"].values,
                                              "quote_ts": daily["quote_ts"].values}))
                except Exception as exc:
                    errors[f"{venue}_{ch}"] = repr(exc)
                    log.warning("%s %s unavailable: %s", venue, ch, exc)
        if CACHE.exists():  # a venue that failed this run keeps its cached history
            old = pd.read_csv(CACHE, parse_dates=["date"])
            old["quote_ts"] = pd.to_datetime(old["quote_ts"], utc=True)
            got = {(r["venue"].iloc[0], r["chamber"].iloc[0]) for r in rows}
            keep = old[[(v, c) not in got for v, c in zip(old["venue"], old["chamber"])]]
            if len(keep):
                rows.append(keep)
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(CACHE, index=False)
    if not rows:
        if not CACHE.exists():
            raise ConnectionError(
                "No prediction-market data: every venue failed and there is no cache. "
                f"Errors: {errors}")
        rows = [pd.read_csv(CACHE, parse_dates=["date"])]
        rows[0]["quote_ts"] = pd.to_datetime(rows[0]["quote_ts"], utc=True)
    long = pd.concat(rows, ignore_index=True)
    long = long[(long["date"] >= pd.Timestamp(start)) & (long["date"] <= pd.Timestamp(end))]
    cons = consensus(long.dropna(subset=["prob_dem"])).reindex(cal).ffill(limit=max_ffill)
    coverage = long.dropna(subset=["prob_dem"]).groupby(["venue", "chamber"]).size() \
        .unstack(fill_value=0)
    return MarketProbabilities(long, cons, cons.diff(), coverage, errors)
