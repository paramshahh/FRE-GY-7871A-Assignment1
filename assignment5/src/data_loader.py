"""Market data: Yahoo Finance and FRED downloads, trading calendar, timestamp cleaning.

Missing trading dates are handled explicitly rather than silently:

* Prices are aligned to an NYSE calendar. A sector ETF with no print on a
  trading day gets a NaN return for that day and the next one; returns are
  never computed across a gap as if it were one day.
* FRED Treasury yields follow the bond-market calendar (closed on Columbus and
  Veterans Day when equities trade). Levels are forward-filled at most
  ``max_ffill`` days and the number of filled cells is recorded.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)

from . import config

log = logging.getLogger(__name__)
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
STOOQ_CSV = "https://stooq.com/q/d/l/?s={symbol}&i=d"


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """NYSE full-day closures (the rules in force since 2022)."""

    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def trading_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """NYSE trading days in [start, end]."""
    hol = NYSEHolidayCalendar().holidays(start, end)
    days = pd.bdate_range(start, end)
    return days[~days.isin(hol)]


def to_trading_day(ts: pd.Series, calendar: pd.DatetimeIndex,
                   close_hour: int = config.MARKET_CLOSE_HOUR) -> pd.Series:
    """Map timestamps to the first trading day whose close they precede.

    Naive timestamps are taken as UTC. Anything stamped at or after the close
    (Eastern) or on a non-trading day rolls forward to the next session, so a
    Saturday tweet is information for Monday's return, never Friday's.
    Timestamps after the last calendar day map to NaT.
    """
    ts = pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert(config.TZ)
    day = ts.dt.tz_localize(None).dt.normalize()
    after_close = ts.dt.hour >= close_hour
    day = day.where(~after_close, day + pd.Timedelta(days=1))
    cal = pd.DatetimeIndex(calendar).sort_values()
    pos = cal.searchsorted(day.values, side="left")
    out = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    ok = (pos < len(cal)) & day.notna().values
    out.iloc[np.where(ok)[0]] = cal[pos[ok]]
    return out


def clean_index(df: pd.DataFrame) -> pd.DataFrame:
    """Tz-naive, normalised, sorted, de-duplicated DatetimeIndex."""
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert(config.TZ).tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "date"
    return out


@dataclass
class AlignmentReport:
    """What alignment to the trading calendar did to each series."""

    missing: dict[str, int] = field(default_factory=dict)
    filled: dict[str, int] = field(default_factory=dict)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"missing_days": self.missing, "forward_filled": self.filled})


def align_to_calendar(df: pd.DataFrame, calendar: pd.DatetimeIndex, max_ffill: int = 0,
                      report: AlignmentReport | None = None) -> pd.DataFrame:
    """Reindex to the trading calendar; optionally forward-fill short gaps in levels."""
    out = clean_index(df).reindex(calendar)
    out.index.name = "date"
    for c in out.columns:
        n_missing = int(out[c].isna().sum())
        if report is not None:
            report.missing[c] = n_missing
        if max_ffill > 0:
            before = out[c].isna().sum()
            out[c] = out[c].ffill(limit=max_ffill)
            if report is not None:
                report.filled[c] = int(before - out[c].isna().sum())
    return out


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------
def download_yahoo(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Daily OHLC per ticker from Yahoo Finance via yfinance (auto-adjusted)."""
    import yfinance as yf

    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=start, end=end_excl, auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    if raw is None or raw.empty:
        raise ConnectionError("yfinance returned no data")
    out = {}
    for t in tickers:
        sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        sub = sub[["Open", "High", "Low", "Close"]].dropna(how="all")
        if sub.empty:
            log.warning("No Yahoo data for %s", t)
            continue
        out[t] = clean_index(sub)
    if not out:
        raise ConnectionError("yfinance returned no usable tickers")
    return out


def download_fred(series: list[str], start: str, end: str,
                  session: requests.Session | None = None) -> pd.DataFrame:
    """FRED series via the public CSV endpoint (no API key). '.' means missing."""
    s = session or requests.Session()
    frames = []
    for sid in series:
        r = s.get(FRED_CSV.format(series=sid), timeout=30)
        r.raise_for_status()
        frames.append(parse_fred_csv(r.text, sid))
    df = pd.concat(frames, axis=1)
    return df.loc[start:end]


def parse_fred_csv(text: str, series: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).rename(columns={df.columns[-1]: series})
    return clean_index(df[[series]].astype(float))


# ---------------------------------------------------------------------------
# Derived series
# ---------------------------------------------------------------------------
def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns; NaN wherever either endpoint is missing (no gap-spanning returns)."""
    return np.log(close).diff()


def excess_returns(returns: pd.DataFrame, benchmark: str = config.BENCHMARK) -> pd.DataFrame:
    """R_sector,t - R_SPY,t for every column except the benchmark."""
    cols = [c for c in returns.columns if c != benchmark]
    return returns[cols].sub(returns[benchmark], axis=0)


def range_volatility(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """Intraday range volatility: 2 (H - L) / (H + L)."""
    return 2.0 * (high - low) / (high + low)


@dataclass
class MarketPanel:
    close: pd.DataFrame
    returns: pd.DataFrame
    excess: pd.DataFrame
    vol: pd.DataFrame
    macro: pd.DataFrame          # SPY return, VIX level/change, DGS2/DGS3MO level/change
    alignment: pd.DataFrame
    source: str


def build_market_panel(ohlc: dict[str, pd.DataFrame], fred: pd.DataFrame,
                       calendar: pd.DatetimeIndex, source: str) -> MarketPanel:
    """Assemble aligned returns, excess returns, range vol and macro controls."""
    rep = AlignmentReport()
    fields = {}
    for f in ("High", "Low", "Close"):
        wide = pd.DataFrame({t: d[f] for t, d in ohlc.items()})
        fields[f] = align_to_calendar(wide, calendar, max_ffill=0,
                                      report=rep if f == "Close" else None)
    close = fields["Close"]
    etfs = [t for t in config.ASSETS if t in close.columns]
    rets = log_returns(close[etfs + [config.BENCHMARK]])
    excess = excess_returns(rets)
    vol = range_volatility(fields["High"][etfs], fields["Low"][etfs])

    fred_al = align_to_calendar(fred, calendar, max_ffill=3, report=rep)
    macro = pd.DataFrame(index=calendar)
    macro["SPY_ret"] = rets[config.BENCHMARK]
    if "^VIX" in close.columns:
        macro["VIX"] = close["^VIX"]
        macro["dVIX"] = close["^VIX"].diff()
    for sid in fred_al.columns:
        macro[sid] = fred_al[sid]
        macro[f"d{sid}"] = fred_al[sid].diff()
    macro["slope_2y_3m"] = macro.get("DGS2", np.nan) - macro.get("DGS3MO", np.nan)
    macro.index.name = "date"
    return MarketPanel(close, rets, excess, vol, macro, rep.as_frame(), source)


def download_stooq(tickers: list[str], start: str, end: str,
                   session: requests.Session | None = None) -> dict[str, pd.DataFrame]:
    """Backup price source: Stooq daily CSV (unadjusted). ``^VIX`` is not on Stooq."""
    s = session or requests.Session()
    out = {}
    for t in tickers:
        if t.startswith("^"):
            continue
        r = s.get(STOOQ_CSV.format(symbol=f"{t.lower()}.us"), timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")
        if {"High", "Low", "Close"} <= set(df.columns):
            out[t] = clean_index(df[["Open", "High", "Low", "Close"]]).loc[start:end]
    if not out:
        raise ConnectionError("Stooq returned no usable tickers")
    return out


def _cache_ohlc(ohlc: dict[str, pd.DataFrame]) -> None:
    long = pd.concat({t: d for t, d in ohlc.items()}, names=["ticker", "date"]).reset_index()
    long.to_csv(config.RAW_DIR / "prices_ohlc.csv", index=False)


def _read_cached_ohlc(start: str, end: str) -> dict[str, pd.DataFrame]:
    long = pd.read_csv(config.RAW_DIR / "prices_ohlc.csv", parse_dates=["date"])
    return {t: clean_index(g.set_index("date").drop(columns="ticker")).loc[start:end]
            for t, g in long.groupby("ticker")}


def load_market_panel(start: str = config.START, end: str = config.END,
                      refresh: bool = True) -> MarketPanel:
    """Download (or read the cached copy of) prices and FRED yields and build the panel.

    Order of preference for prices: Yahoo Finance, then Stooq for any ticker
    Yahoo did not return, then the on-disk cache from the last successful run.
    There is no simulated fallback: if no real data can be obtained this raises.
    """
    cal = trading_calendar(start, end)
    tickers = config.ASSETS + [config.BENCHMARK] + list(config.YAHOO_MACRO.values())
    fred_path = config.RAW_DIR / "fred.csv"
    source = "cache"
    if refresh:
        ohlc: dict[str, pd.DataFrame] = {}
        try:
            ohlc = download_yahoo(tickers, start, end)
            source = "yahoo"
        except Exception as exc:
            log.warning("Yahoo download failed: %s", exc)
        missing = [t for t in tickers if t not in ohlc]
        if missing:
            try:
                ohlc.update(download_stooq(missing, start, end))
                source = "yahoo+stooq" if source == "yahoo" else "stooq"
            except Exception as exc:
                log.warning("Stooq download failed: %s", exc)
        if ohlc:
            _cache_ohlc(ohlc)
        try:
            download_fred(list(config.FRED_SERIES.values()), start, end).to_csv(fred_path)
        except Exception as exc:
            log.warning("FRED download failed: %s", exc)
    cache = config.RAW_DIR / "prices_ohlc.csv"
    if not cache.exists() or not fred_path.exists():
        raise ConnectionError(
            "No market data available: downloads failed and there is no cache in "
            f"{config.RAW_DIR}. Check network access to query1.finance.yahoo.com, "
            "stooq.com and fred.stlouisfed.org.")
    ohlc = _read_cached_ohlc(start, end)
    absent = [t for t in config.ASSETS + [config.BENCHMARK] if t not in ohlc]
    if absent:
        raise ValueError(f"Price data missing for {absent}")
    fred = clean_index(pd.read_csv(fred_path, index_col=0, parse_dates=True))
    return build_market_panel(ohlc, fred, cal, source=source)
