# cfd-lab

Cost-realistic daily research harness for CFD strategies on S&P 500, NASDAQ 100,
DAX 40 and EUR/USD, built so a strategy is proven before anything is automated.
Execution (IBKR) and signals (TradingView) come later, only after a strategy
passes the gates in [`gates.yaml`](gates.yaml).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,yfinance]"
pytest

cfdlab fetch                       # all four instruments from Stooq (or --source yfinance)
cfdlab backtest SPX                # trend-pullback on the train period
cfdlab robustness SPX              # ±20% parameter moves, 1-bar delay, Monte Carlo
cfdlab walkforward SPX             # rolling re-optimisation + deflated Sharpe
cfdlab backtest --period validation
```

The test period is locked. Touching it needs `--unlock-test --note "why"`, and
every unlock is appended to `logs/test_access.log`, which is committed. Run it
once, for the pre-registered evaluation:

```bash
cfdlab backtest --period test --unlock-test --note "research gate"
cfdlab robustness --period test --unlock-test --note "research gate"
cfdlab gate reports/backtest_trend_pullback_test.json --robustness reports/robustness_trend_pullback_test.json
```

## What the backtester models

- **Timing:** signals are decided at the close and filled at the next open
  (`--delay N` adds bars). Indicators warm up on earlier data; trading is
  limited to the chosen window and positions close at its end.
- **Costs** (per instrument, `config/instruments.yaml`): half the spread plus
  slippage on every fill, commission as a percent of notional, and overnight
  financing charged on notional for every calendar day held (weekends count 3).
- **Stops:** ATR initial stop and ATR trailing stop. A gap through the stop fills
  at the open, not at the stop.
- **Sizing:** risk per trade as a fraction of equity divided by stop distance,
  capped by max leverage.
- **Loss limits:** breaching the daily, weekly or monthly limit blocks new
  entries for the rest of that period. Blocked entries are counted.

## Metrics

Net return, CAGR, max drawdown, longest time below a high, Sharpe, Sortino,
Calmar, win rate, average win and loss, profit factor, exposure, turnover,
costs by type, returns by year, and the Deflated Sharpe Ratio for walk-forward
searches.

## Before trusting any result

1. Verify every value marked `VERIFY` in `config/instruments.yaml` against IBKR
   Ireland's CFD pages and your account (spreads, financing benchmark and
   markup, short credit, FX costs).
2. Free index data has no CFD session quirks (DAX CFD hours, US overnight
   prices). Check a sample against IBKR's own CFD history.
3. Set `registered_at` and `status: registered` in `gates.yaml` and commit it
   before the first real backtest.

## Layout

```
src/cfdlab/
  data.py         Stooq / yfinance / CSV loaders, cleaning report, Parquet cache
  costs.py        spread, slippage, commission, daily financing
  indicators.py   EMA, RSI, ATR (causal)
  strategies/     trend_pullback.py (first candidate)
  backtest.py     daily event loop, stops, sizing, loss limits
  metrics.py      performance metrics and Deflated Sharpe
  robustness.py   perturbation, delay, walk-forward, Monte Carlo
  splits.py       train / validation / locked test windows
  gates.py        research gate evaluation
  cli.py          `cfdlab` command
```
