"""Synthetic OHLC for tests and offline demos. Never use it to judge a strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_walk(n: int = 3000, start: str = "2010-01-01", drift: float = 0.0003,
                vol: float = 0.011, seed: int = 0, price: float = 1000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    rets = rng.normal(drift, vol, n)
    close = price * np.exp(np.cumsum(rets))
    open_ = np.r_[price, close[:-1]] * np.exp(rng.normal(0, vol / 4, n))
    span = np.abs(rng.normal(0, vol / 2, n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close},
                        index=pd.DatetimeIndex(dates, name="date"))
