from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from cfdlab import backtest, data, metrics, robustness, splits
from cfdlab.backtest import RiskConfig
from cfdlab.costs import ZERO, CostModel
from cfdlab.strategies import TrendPullback
from cfdlab.synthetic import random_walk


def bars(rows, start="2024-01-01"):
    """rows: list of (open, high, low, close) on consecutive business days."""
    idx = pd.bdate_range(start, periods=len(rows), name="date")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


@dataclass(frozen=True)
class Scripted:
    """Test strategy: fixed entries and a constant ATR."""
    entries: tuple
    exits: tuple = ()
    atr_value: float = 10.0
    stop_atr: float = 1.0
    trail_atr: float = 100.0

    @property
    def params(self):
        return {}

    def signals(self, df):
        out = pd.DataFrame(index=df.index)
        out["entry"] = [self.entries[i] if i < len(self.entries) else 0 for i in range(len(df))]
        ex = [i in self.exits for i in range(len(df))]
        out["exit_long"] = ex
        out["exit_short"] = ex
        out["atr"] = self.atr_value
        return out


NO_LIMITS = RiskConfig(risk_per_trade=0.01, daily_loss_limit=1, weekly_loss_limit=1, monthly_loss_limit=1, max_leverage=100)


def test_signal_fills_at_next_open_not_same_close():
    df = bars([(100, 101, 99, 100), (105, 106, 104, 105), (106, 107, 105, 106)])
    res = backtest.run(df, Scripted(entries=(1,)), ZERO, NO_LIMITS)
    t = res.trades[0]
    assert t.entry_date == df.index[1] and t.entry_price == 105


def test_delay_shifts_fill_by_one_bar():
    df = bars([(100, 101, 99, 100), (105, 106, 104, 105), (110, 111, 109, 110), (110, 111, 109, 110)])
    res = backtest.run(df, Scripted(entries=(1,)), ZERO, NO_LIMITS, delay=1)
    assert res.trades[0].entry_date == df.index[2] and res.trades[0].entry_price == 110


def test_position_sized_by_stop_distance():
    df = bars([(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100)])
    res = backtest.run(df, Scripted(entries=(1,), atr_value=5, stop_atr=2), ZERO, NO_LIMITS)
    # risk 1% of 10,000 = 100; stop distance 10 -> 10 units
    assert res.trades[0].units == pytest.approx(10)


def test_gap_through_stop_fills_at_open():
    df = bars([(100, 101, 99, 100), (100, 101, 99, 100), (80, 82, 79, 81), (81, 82, 80, 81)])
    res = backtest.run(df, Scripted(entries=(1,)), ZERO, NO_LIMITS)
    t = res.trades[0]
    assert t.reason == "stop_gap" and t.exit_price == 80  # worse than the 90 stop


def test_intraday_stop_fills_at_stop():
    df = bars([(100, 101, 99, 100), (100, 101, 99, 100), (95, 96, 85, 92), (92, 93, 91, 92)])
    res = backtest.run(df, Scripted(entries=(1,)), ZERO, NO_LIMITS)
    assert res.trades[0].reason == "stop" and res.trades[0].exit_price == 90


def test_short_stop_above_entry():
    df = bars([(100, 101, 99, 100), (100, 101, 99, 100), (105, 112, 104, 108), (108, 109, 107, 108)])
    res = backtest.run(df, Scripted(entries=(-1,)), ZERO, NO_LIMITS)
    t = res.trades[0]
    assert t.side == -1 and t.exit_price == 110 and t.gross_pnl < 0


def test_financing_counts_weekend_days():
    # Friday entry, held over the weekend to Monday, exits Tuesday at the open.
    idx = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"], name="date")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=idx)
    costs = CostModel(financing_long=0.036, day_basis=360)
    res = backtest.run(df, Scripted(entries=(1,), exits=(2,)), costs, NO_LIMITS)
    t = res.trades[0]
    # 10 units * 100 notional = 1000; Fri->Mon 3 days + Mon->Tue 1 day = 4 days at 0.01%/day
    assert t.financing == pytest.approx(1000 * 0.036 / 360 * 4)


def test_spread_and_commission_reduce_pnl():
    df = bars([(100, 101, 99, 100)] * 4)
    costs = CostModel(spread_bps=10, commission_pct=0.01)
    res = backtest.run(df, Scripted(entries=(1,), exits=(2,)), costs, NO_LIMITS)
    t = res.trades[0]
    assert t.entry_price == pytest.approx(100.05) and t.exit_price == pytest.approx(99.95)
    assert t.net_pnl < 0 and res.equity.iloc[-1] < 10_000


def test_daily_loss_limit_blocks_next_entry():
    risk = RiskConfig(risk_per_trade=0.05, daily_loss_limit=0.02, weekly_loss_limit=1, monthly_loss_limit=1, max_leverage=100)
    # Entry on bar 1, stopped out on bar 2 for a 5% loss, new signal on bar 2 blocked on bar 3.
    df = bars([(100, 101, 99, 100), (100, 101, 99, 100), (95, 96, 85, 92), (92, 93, 91, 92), (92, 93, 91, 92)])
    res = backtest.run(df, Scripted(entries=(1, 0, 1)), ZERO, risk)
    assert len(res.trades) == 1 and res.blocked_entries == 1


def test_no_lookahead_future_data_does_not_change_past():
    df = random_walk(1500, seed=3)
    s = TrendPullback()
    full = backtest.run(df, s, ZERO, end=df.index[1200])
    cut = backtest.run(df.iloc[:1201], s, ZERO)
    pd.testing.assert_series_equal(full.equity, cut.equity)


def test_trend_pullback_runs_and_costs_hurt():
    df = random_walk(3000, seed=1)
    s = TrendPullback()
    free = metrics.summary(backtest.run(df, s, ZERO))
    costly = metrics.summary(backtest.run(df, s, CostModel(spread_bps=2, commission_pct=0.01, financing_long=0.054)))
    assert free["trades"] > 10
    assert costly["net_return"] < free["net_return"]


def test_metrics_on_known_curve():
    eq = pd.Series([100, 110, 99, 120], index=pd.bdate_range("2024-01-01", periods=4), dtype=float)
    assert metrics.max_drawdown(eq) == pytest.approx(0.1)


def test_deflated_sharpe_falls_with_more_trials():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 2000))
    assert metrics.deflated_sharpe(r, 100) < metrics.deflated_sharpe(r, 1)


def test_test_period_is_locked(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "ACCESS_LOG", tmp_path / "access.log")
    g = {"splits": {"test": ["2023-01-01", "2026-09-30"]}}
    with pytest.raises(splits.TestPeriodLocked):
        splits.window("test", g)
    assert splits.window("test", g, unlock_test=True, note="final") == ("2023-01-01", "2026-09-30")
    assert "final" in (tmp_path / "access.log").read_text()


def test_clean_drops_duplicates_and_repairs_ohlc():
    df = bars([(100, 99, 98, 101), (100, 101, 99, 100)])
    df = pd.concat([df, df.iloc[[1]]])
    out, rep = data.clean(df)
    assert rep.duplicates_dropped == 1 and rep.ohlc_repaired == 1
    assert out.iloc[0]["high"] == 101


def test_robustness_helpers_run():
    df = random_walk(2500, seed=2)
    s = TrendPullback()
    pert = robustness.perturbation(df, s, ZERO, RiskConfig(), None, None)
    assert set(pert.factor) == {0.8, 1.0, 1.2}
    mc = robustness.monte_carlo(backtest.run(df, s, ZERO))
    assert mc["max_drawdown"]["p95"] >= mc["max_drawdown"]["p5"]
