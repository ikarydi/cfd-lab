"""Single-instrument daily backtester for CFDs.

Timing (no look-ahead): a signal is decided at bar t's close and filled at the
open of bar t+1+delay. Within each bar the order of events is:
  1. pending exit fills at the open
  2. pending entry fills at the open (unless a loss limit blocks new entries)
  3. stop check: a gap through the stop fills at the open, otherwise at the stop
  4. at the close: mark to market, charge overnight financing up to the next
     bar's date, ratchet the trailing stop, check loss limits, queue orders
Indicators are computed on the full history; trading is restricted to
[start, end], and any open position is closed at the last bar's close.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.0075
    daily_loss_limit: float = 0.02
    weekly_loss_limit: float = 0.05
    monthly_loss_limit: float = 0.10
    max_leverage: float = 3.0

    @classmethod
    def from_gates(cls, gates: dict) -> "RiskConfig":
        return cls(**gates["risk"])


@dataclass
class Trade:
    side: int
    entry_date: pd.Timestamp
    entry_price: float
    units: float
    stop: float
    equity_at_entry: float
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    reason: str = ""
    gross_pnl: float = 0.0
    commission: float = 0.0
    financing: float = 0.0
    spread_slippage: float = 0.0

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.commission - self.financing

    @property
    def r_return(self) -> float:
        """Net P&L as a fraction of equity when the trade was opened."""
        return self.net_pnl / self.equity_at_entry


@dataclass
class Result:
    equity: pd.Series
    trades: list[Trade]
    costs: dict = field(default_factory=dict)
    blocked_entries: int = 0
    params: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)

    def trades_frame(self) -> pd.DataFrame:
        rows = [
            {**{k: getattr(t, k) for k in t.__dataclass_fields__}, "net_pnl": t.net_pnl, "r_return": t.r_return}
            for t in self.trades
        ]
        return pd.DataFrame(rows)


def run(
    df: pd.DataFrame,
    strategy,
    costs: CostModel,
    risk: RiskConfig = RiskConfig(),
    start=None,
    end=None,
    delay: int = 0,
    initial_equity: float = 10_000.0,
) -> Result:
    sig = strategy.signals(df)
    if delay:
        sig["entry"] = sig["entry"].shift(delay, fill_value=0)
        sig["exit_long"] = sig["exit_long"].shift(delay, fill_value=False)
        sig["exit_short"] = sig["exit_short"].shift(delay, fill_value=False)

    mask = np.ones(len(df), dtype=bool)
    if start is not None:
        mask &= df.index >= pd.Timestamp(start)
    if end is not None:
        mask &= df.index <= pd.Timestamp(end)
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        raise ValueError("Trading window has fewer than two bars")

    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    dates = df.index
    entry_sig = sig["entry"].to_numpy()
    exit_long = sig["exit_long"].to_numpy()
    exit_short = sig["exit_short"].to_numpy()
    atr_v = sig["atr"].to_numpy()

    cash = initial_equity  # realised equity; open P&L is added when marking
    pos: Trade | None = None
    pending_entry = 0
    pending_exit = False
    trades: list[Trade] = []
    totals = dict(commission=0.0, financing=0.0, spread_slippage=0.0)
    blocked = 0
    equity_curve = np.empty(len(idx))
    period_start = {"week": (None, initial_equity), "month": (None, initial_equity)}
    prev_close_equity = initial_equity
    block_until: pd.Timestamp | None = None

    def close_position(i: int, raw_price: float, reason: str) -> None:
        nonlocal cash, pos
        fill = costs.fill_price(raw_price, -pos.side)
        notional = pos.units * fill
        comm = costs.commission(notional)
        pos.exit_date, pos.exit_price, pos.reason = dates[i], fill, reason
        pos.gross_pnl = pos.side * pos.units * (fill - pos.entry_price)
        pos.commission += comm
        pos.spread_slippage += pos.units * abs(raw_price - fill)
        cash += pos.gross_pnl - comm
        totals["commission"] += comm
        totals["spread_slippage"] += pos.units * abs(raw_price - fill)
        trades.append(pos)
        pos = None

    for n, i in enumerate(idx):
        d = dates[i]
        # 1. pending exit
        if pos is not None and pending_exit:
            close_position(i, o[i], "signal_exit")
        pending_exit = False

        # 2. pending entry
        if pos is None and pending_entry != 0:
            if block_until is not None and d < block_until:
                blocked += 1
            elif np.isfinite(atr_v[i - 1]) and atr_v[i - 1] > 0:
                side = pending_entry
                fill = costs.fill_price(o[i], side)
                stop_dist = strategy.stop_atr * atr_v[i - 1]
                equity_now = cash
                units = equity_now * risk.risk_per_trade / stop_dist
                units = min(units, equity_now * risk.max_leverage / fill)
                comm = costs.commission(units * fill)
                cash -= comm
                totals["commission"] += comm
                totals["spread_slippage"] += units * abs(fill - o[i])
                pos = Trade(side, d, fill, units, fill - side * stop_dist, equity_now,
                            commission=comm, spread_slippage=units * abs(fill - o[i]))
        pending_entry = 0

        # 3. stop check
        if pos is not None:
            if pos.side > 0 and o[i] <= pos.stop:
                close_position(i, o[i], "stop_gap")
            elif pos.side > 0 and l[i] <= pos.stop:
                close_position(i, pos.stop, "stop")
            elif pos.side < 0 and o[i] >= pos.stop:
                close_position(i, o[i], "stop_gap")
            elif pos.side < 0 and h[i] >= pos.stop:
                close_position(i, pos.stop, "stop")

        last = n == len(idx) - 1
        if last and pos is not None:
            close_position(i, c[i], "end_of_window")

        # 4. close-of-bar bookkeeping
        if pos is not None:
            next_d = dates[idx[n + 1]]
            days = max((next_d - d).days, 1)
            fin = costs.financing(pos.units * c[i], pos.side, days)
            cash -= fin
            pos.financing += fin
            totals["financing"] += fin
            trail = c[i] - pos.side * strategy.trail_atr * atr_v[i]
            if np.isfinite(trail):
                pos.stop = max(pos.stop, trail) if pos.side > 0 else min(pos.stop, trail)

        open_pnl = 0.0 if pos is None else pos.side * pos.units * (c[i] - pos.entry_price)
        equity = cash + open_pnl
        equity_curve[n] = equity

        # loss limits: stop opening new positions for the rest of the period
        week, month = d.isocalendar()[:2], (d.year, d.month)
        if period_start["week"][0] != week:
            period_start["week"] = (week, prev_close_equity)
        if period_start["month"][0] != month:
            period_start["month"] = (month, prev_close_equity)
        if not last:
            nd = dates[idx[n + 1]]
            if equity / prev_close_equity - 1 <= -risk.daily_loss_limit:
                block_until = max(block_until or nd, nd + pd.Timedelta(days=1))
            if equity / period_start["week"][1] - 1 <= -risk.weekly_loss_limit:
                block_until = max(block_until or nd, (d + pd.offsets.Week(weekday=0)).normalize())
            if equity / period_start["month"][1] - 1 <= -risk.monthly_loss_limit:
                block_until = max(block_until or nd, (d + pd.offsets.MonthBegin(1)).normalize())
        prev_close_equity = equity

        if not last:
            if pos is not None:
                pending_exit = bool(exit_long[i]) if pos.side > 0 else bool(exit_short[i])
            else:
                pending_entry = int(entry_sig[i])

    return Result(
        equity=pd.Series(equity_curve, index=dates[idx], name="equity"),
        trades=trades,
        costs=totals,
        blocked_entries=blocked,
        params=dict(strategy.params, delay=delay),
    )
