"""Strategies turn OHLC into per-bar signals; the backtester does all execution.

A strategy exposes `params` (a dict), `stop_atr`, `trail_atr`, and
`signals(df)` returning a DataFrame aligned to df with columns:
  entry  +1 go long, -1 go short, 0 nothing (decided at this bar's close)
  exit   True to close any open position at the next open
  atr    ATR used for stop distances
"""
from .trend_pullback import TrendPullback

REGISTRY = {"trend_pullback": TrendPullback}
