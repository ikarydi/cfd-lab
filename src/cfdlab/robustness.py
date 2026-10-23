"""Robustness checks: parameter perturbation, execution delay, walk-forward, Monte Carlo."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import backtest, metrics
from .costs import CostModel

TUNABLE = ("ema_len", "rsi_len", "rsi_entry", "atr_len", "stop_atr", "trail_atr")


def _perturbed(value, factor):
    out = value * factor
    return max(1, int(round(out))) if isinstance(value, int) and not isinstance(value, bool) else out


def perturbation(df, strategy, costs: CostModel, risk, start, end, pct: float = 0.20) -> pd.DataFrame:
    """Move each tunable parameter to x(1-pct) and x(1+pct), one at a time."""
    rows = []
    for name in TUNABLE:
        if name not in strategy.params:
            continue
        for factor in (1 - pct, 1.0, 1 + pct):
            s = strategy.with_params(**{name: _perturbed(strategy.params[name], factor)})
            m = metrics.summary(backtest.run(df, s, costs, risk, start, end))
            rows.append({"param": name, "factor": factor, "value": s.params[name],
                         "sharpe": m["sharpe"], "cagr": m["cagr"], "max_drawdown": m["max_drawdown"]})
    return pd.DataFrame(rows)


def delay_test(df, strategy, costs, risk, start, end, delays=(0, 1)) -> pd.DataFrame:
    rows = []
    for d in delays:
        m = metrics.summary(backtest.run(df, strategy, costs, risk, start, end, delay=d))
        rows.append({"delay": d, "sharpe": m["sharpe"], "cagr": m["cagr"], "max_drawdown": m["max_drawdown"]})
    return pd.DataFrame(rows)


def walk_forward(df, strategy, costs, risk, start, end, grid: dict, train_years=5, test_years=1):
    """Rolling re-optimisation: pick the best train-window Sharpe, trade the next window.

    Returns (oos_returns, choices). Out-of-sample returns are stitched in order.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    oos, choices = [], []
    t0 = start
    while True:
        train_end = t0 + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        test_end = min(train_end + pd.DateOffset(years=test_years), end)
        if train_end >= end:
            break
        best, best_sr = None, -np.inf
        for combo in combos:
            s = strategy.with_params(**combo)
            sr = metrics.sharpe(backtest.run(df, s, costs, risk, t0, train_end).returns)
            if sr > best_sr:
                best, best_sr = combo, sr
        res = backtest.run(df, strategy.with_params(**best), costs, risk, train_end + pd.Timedelta(days=1), test_end)
        oos.append(res.returns.iloc[1:])
        choices.append({"train": f"{t0.date()}..{train_end.date()}", "test_end": str(test_end.date()),
                        "train_sharpe": best_sr, "oos_sharpe": metrics.sharpe(res.returns), **best})
        t0 = t0 + pd.DateOffset(years=test_years)
    return (pd.concat(oos) if oos else pd.Series(dtype=float)), pd.DataFrame(choices), len(combos)


def monte_carlo(result: backtest.Result, n: int = 2000, seed: int = 7) -> dict:
    """Bootstrap trade returns to see how bad drawdowns could plausibly be."""
    r = np.array([t.r_return for t in result.trades])
    if len(r) < 5:
        return {"trades": len(r), "note": "too few trades for a meaningful bootstrap"}
    rng = np.random.default_rng(seed)
    samples = rng.choice(r, size=(n, len(r)), replace=True)
    paths = np.cumprod(1 + samples, axis=1)
    peaks = np.maximum.accumulate(paths, axis=1)
    mdd = (1 - paths / peaks).max(axis=1)
    final = paths[:, -1] - 1
    q = lambda a: {f"p{p}": float(np.percentile(a, p)) for p in (5, 50, 95)}
    return {"trades": len(r), "max_drawdown": q(mdd), "net_return": q(final),
            "prob_loss": float((final < 0).mean())}
