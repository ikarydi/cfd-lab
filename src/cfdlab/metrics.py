"""Performance metrics from the plan, plus the Deflated Sharpe Ratio."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from .backtest import Result

PERIODS = 252
_N = NormalDist()


def sharpe(returns: pd.Series) -> float:
    sd = returns.std(ddof=1)
    return float(returns.mean() / sd * math.sqrt(PERIODS)) if sd > 0 else 0.0


def sortino(returns: pd.Series) -> float:
    downside = np.sqrt((returns.clip(upper=0) ** 2).mean())
    return float(returns.mean() / downside * math.sqrt(PERIODS)) if downside > 0 else 0.0


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    return float(-drawdown(equity).min())


def longest_recovery_days(equity: pd.Series) -> int:
    """Longest calendar stretch spent below a previous equity high (open-ended if unrecovered)."""
    peak_dates = equity.index[equity >= equity.cummax()]
    bounds = list(peak_dates) + [equity.index[-1]]
    return int(max((b - a).days for a, b in zip(bounds[:-1], bounds[1:])) if len(bounds) > 1 else 0)


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0 or equity.iloc[0] <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def deflated_sharpe(returns: pd.Series, n_trials: int, trial_sharpes_var: float | None = None) -> float:
    """Probability that the true Sharpe is above zero after testing n_trials variants.

    Bailey & Lopez de Prado (2014). Uses per-period Sharpe internally.
    `trial_sharpes_var` is the variance of the per-period Sharpe across trials;
    if omitted it falls back to the estimator's own variance (a lenient default).
    """
    T = len(returns)
    sd = returns.std(ddof=1)
    if T < 3 or sd == 0:
        return 0.0
    sr = returns.mean() / sd
    skew = float(returns.skew())
    kurt = float(returns.kurt()) + 3
    sr_var_est = (1 - skew * sr + (kurt - 1) / 4 * sr**2) / (T - 1)
    var = trial_sharpes_var if trial_sharpes_var is not None else sr_var_est
    if n_trials > 1:
        g = 0.5772156649
        sr0 = math.sqrt(var) * ((1 - g) * _N.inv_cdf(1 - 1 / n_trials) + g * _N.inv_cdf(1 - 1 / (n_trials * math.e)))
    else:
        sr0 = 0.0
    return float(_N.cdf((sr - sr0) / math.sqrt(sr_var_est)))


def summary(result: Result) -> dict:
    eq, r = result.equity, result.returns
    tf = result.trades_frame()
    wins = tf[tf.net_pnl > 0] if len(tf) else tf
    losses = tf[tf.net_pnl <= 0] if len(tf) else tf
    gross_win = wins.net_pnl.sum() if len(tf) else 0.0
    gross_loss = -losses.net_pnl.sum() if len(tf) else 0.0
    mdd = max_drawdown(eq)
    g = cagr(eq)
    in_market_days = sum(
        ((t.exit_date - t.entry_date).days or 1) for t in result.trades
    )
    span_days = max((eq.index[-1] - eq.index[0]).days, 1)
    yearly = eq.resample("YE").last().pct_change()
    yearly.iloc[0] = eq.resample("YE").last().iloc[0] / eq.iloc[0] - 1
    return {
        "start": eq.index[0].date().isoformat(),
        "end": eq.index[-1].date().isoformat(),
        "net_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "cagr": g,
        "max_drawdown": mdd,
        "longest_recovery_days": longest_recovery_days(eq),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "calmar": g / mdd if mdd > 0 else 0.0,
        "trades": len(tf),
        "win_rate": len(wins) / len(tf) if len(tf) else 0.0,
        "avg_win": float(wins.net_pnl.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.net_pnl.mean()) if len(losses) else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
        "exposure": min(in_market_days / span_days, 1.0),
        "turnover_per_year": float(sum(2 * t.units * t.entry_price for t in result.trades) / eq.mean() / (span_days / 365.25)),
        "costs": {k: round(v, 2) for k, v in result.costs.items()},
        "costs_pct_of_start_equity": float(sum(result.costs.values()) / eq.iloc[0]),
        "blocked_entries": result.blocked_entries,
        "by_year": {str(k.year): round(float(v), 4) for k, v in yearly.items()},
    }
