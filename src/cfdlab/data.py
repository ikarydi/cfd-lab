"""Daily OHLC loading, cleaning and caching.

Every source is normalised to a DataFrame indexed by a tz-naive DatetimeIndex
(one row per trading day) with float columns open, high, low, close.
"""
from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

COLUMNS = ["open", "high", "low", "close"]
CACHE_DIR = Path("data/raw")


@dataclass
class CleanReport:
    rows_in: int
    rows_out: int
    duplicates_dropped: int = 0
    missing_dropped: int = 0
    ohlc_repaired: int = 0
    gaps_over_5_days: list[tuple[str, str]] = field(default_factory=list)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "date"
    return df[COLUMNS].astype(float)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleanReport]:
    """Sort, drop duplicate dates and missing rows, fix impossible OHLC, report gaps."""
    report = CleanReport(rows_in=len(df), rows_out=0)
    df = df.sort_index()
    dup = df.index.duplicated(keep="last")
    report.duplicates_dropped = int(dup.sum())
    df = df[~dup]
    bad = df[COLUMNS].isna().any(axis=1) | (df[COLUMNS] <= 0).any(axis=1)
    report.missing_dropped = int(bad.sum())
    df = df[~bad].copy()
    # Some free feeds report high/low that exclude the open or close.
    hi = df[COLUMNS].max(axis=1)
    lo = df[COLUMNS].min(axis=1)
    repaired = (df["high"] < hi) | (df["low"] > lo)
    report.ohlc_repaired = int(repaired.sum())
    df["high"], df["low"] = hi, lo
    gaps = df.index.to_series().diff().dt.days
    for end, days in gaps[gaps > 5].items():
        start = df.index[df.index.get_loc(end) - 1]
        report.gaps_over_5_days.append((start.date().isoformat(), end.date().isoformat()))
    report.rows_out = len(df)
    return df, report


def load_csv(path: str | Path) -> pd.DataFrame:
    return _normalise(pd.read_csv(path))


def fetch_stooq(symbol: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode()
    if not text.lower().startswith("date"):
        raise RuntimeError(f"Stooq returned no data for {symbol!r}: {text[:80]!r}")
    return _normalise(pd.read_csv(io.StringIO(text)))


def fetch_yfinance(symbol: str, start: str = "2005-01-01") -> pd.DataFrame:
    import yfinance as yf  # optional dependency

    df = yf.download(symbol, start=start, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol!r}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return _normalise(df)


def cache_path(instrument: str) -> Path:
    return CACHE_DIR / f"{instrument}.parquet"


def save(instrument: str, df: pd.DataFrame) -> Path:
    path = cache_path(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def load(instrument: str) -> pd.DataFrame:
    path = cache_path(instrument)
    if not path.exists():
        raise FileNotFoundError(f"No cached data for {instrument}; run `cfdlab fetch {instrument}` first")
    return pd.read_parquet(path)
