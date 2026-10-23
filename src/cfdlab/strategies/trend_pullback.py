"""Trend-pullback: buy dips in an uptrend (and optionally sell rallies in a downtrend).

Trend filter: close above/below EMA(ema_len).
Pullback:     RSI(rsi_len) below rsi_entry (long) or above 100 - rsi_entry (short).
Exit:         ATR initial stop and ATR trailing stop (handled by the backtester),
              plus a close back through the EMA.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from ..indicators import atr, ema, rsi


@dataclass(frozen=True)
class TrendPullback:
    ema_len: int = 200
    rsi_len: int = 3
    rsi_entry: float = 20.0
    atr_len: int = 14
    stop_atr: float = 2.0
    trail_atr: float = 3.0
    allow_short: bool = False

    @property
    def params(self) -> dict:
        return asdict(self)

    def with_params(self, **kw) -> "TrendPullback":
        return replace(self, **kw)

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        trend = ema(df["close"], self.ema_len)
        r = rsi(df["close"], self.rsi_len)
        up = df["close"] > trend
        down = df["close"] < trend
        entry = np.where(up & (r < self.rsi_entry), 1, 0)
        if self.allow_short:
            entry = np.where(down & (r > 100 - self.rsi_entry), -1, entry)
        # Exit on a close through the EMA against the position; the backtester
        # applies `exit` only to the side it contradicts.
        out = pd.DataFrame(index=df.index)
        out["entry"] = pd.Series(entry, index=df.index).where(trend.notna() & r.notna(), 0).astype(int)
        out["exit_long"] = down.fillna(False)
        out["exit_short"] = up.fillna(False)
        out["atr"] = atr(df, self.atr_len)
        return out
