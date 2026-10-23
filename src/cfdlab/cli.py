"""Command line: fetch data, run a backtest, run robustness checks, evaluate gates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from . import backtest, data, gates, metrics, robustness, splits
from .costs import CONFIG, CostModel
from .strategies import REGISTRY

REPORTS = Path("reports")


def _instruments() -> dict:
    return yaml.safe_load(CONFIG.read_text())["instruments"]


def cmd_fetch(args) -> None:
    cfg = _instruments()
    for inst in args.instruments or list(cfg):
        if args.csv:
            raw = data.load_csv(args.csv)
        elif args.source == "yfinance":
            raw = data.fetch_yfinance(cfg[inst]["yfinance"])
        else:
            raw = data.fetch_stooq(cfg[inst]["stooq"])
        df, rep = data.clean(raw)
        path = data.save(inst, df)
        print(f"{inst}: {rep.rows_out} rows {df.index[0].date()}..{df.index[-1].date()} -> {path}")
        print(f"  dropped {rep.duplicates_dropped} duplicates, {rep.missing_dropped} bad rows; "
              f"repaired {rep.ohlc_repaired} OHLC; {len(rep.gaps_over_5_days)} gaps over 5 days")


def _setup(args):
    g = splits.load_gates()
    start, end = splits.window(args.period, g, unlock_test=args.unlock_test, note=args.note)
    risk = backtest.RiskConfig.from_gates(g)
    strategy = REGISTRY[args.strategy]()
    if args.params:
        strategy = strategy.with_params(**json.loads(args.params))
    return g, start, end, risk, strategy


def _write(name: str, payload) -> Path:
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def cmd_backtest(args) -> None:
    g, start, end, risk, strategy = _setup(args)
    out = {}
    for inst in args.instruments or list(_instruments()):
        res = backtest.run(data.load(inst), strategy, CostModel.from_config(inst), risk, start, end, delay=args.delay)
        out[inst] = metrics.summary(res)
        m = out[inst]
        print(f"{inst:7s} {args.period}: net {m['net_return']:+.1%}  CAGR {m['cagr']:+.1%}  "
              f"Sharpe {m['sharpe']:.2f}  maxDD {m['max_drawdown']:.1%}  trades {m['trades']}  "
              f"costs {m['costs_pct_of_start_equity']:.1%}")
    print("report:", _write(f"backtest_{args.strategy}_{args.period}", {"params": strategy.params, "results": out}))


def cmd_robustness(args) -> None:
    g, start, end, risk, strategy = _setup(args)
    pct = g["research_gate"]["robustness"]["param_perturbation"]
    out = {}
    for inst in args.instruments or list(_instruments()):
        df, costs = data.load(inst), CostModel.from_config(inst)
        pert = robustness.perturbation(df, strategy, costs, risk, start, end, pct)
        delay = robustness.delay_test(df, strategy, costs, risk, start, end)
        mc = robustness.monte_carlo(backtest.run(df, strategy, costs, risk, start, end))
        out[inst] = {
            "min_perturbed_sharpe": float(pert.sharpe.min()),
            "delay1_sharpe": float(delay.set_index("delay").sharpe[1]),
            "perturbation": pert.to_dict("records"),
            "delay": delay.to_dict("records"),
            "monte_carlo": mc,
        }
        print(f"{inst:7s} worst Sharpe under ±{pct:.0%}: {out[inst]['min_perturbed_sharpe']:.2f}   "
              f"with 1-bar delay: {out[inst]['delay1_sharpe']:.2f}   "
              f"MC maxDD p95: {mc.get('max_drawdown', {}).get('p95', float('nan')):.1%}")
    print("report:", _write(f"robustness_{args.strategy}_{args.period}", out))


def cmd_walkforward(args) -> None:
    g, start, end, risk, strategy = _setup(args)
    grid = json.loads(args.grid)
    for inst in args.instruments or list(_instruments()):
        oos, choices, n = robustness.walk_forward(data.load(inst), strategy, CostModel.from_config(inst),
                                                   risk, start, end, grid)
        dsr = metrics.deflated_sharpe(oos, n_trials=n)
        print(f"{inst:7s} walk-forward OOS Sharpe {metrics.sharpe(oos):.2f}  "
              f"deflated Sharpe prob {dsr:.2f} ({n} variants)")
        print(choices.to_string(index=False))


def cmd_gate(args) -> None:
    bt = json.loads(Path(args.backtest).read_text())["results"]
    rb = json.loads(Path(args.robustness).read_text()) if args.robustness else {}
    verdict = gates.research_gate(bt, rb, splits.load_gates())
    print(json.dumps(verdict, indent=2))
    sys.exit(0 if verdict["passed"] else 1)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="cfdlab")
    sub = p.add_subparsers(required=True)

    f = sub.add_parser("fetch", help="download and clean daily data into data/raw")
    f.add_argument("instruments", nargs="*")
    f.add_argument("--source", choices=["stooq", "yfinance"], default="stooq")
    f.add_argument("--csv", help="load one instrument from a local CSV instead")
    f.set_defaults(func=cmd_fetch)

    def common(sp):
        sp.add_argument("instruments", nargs="*")
        sp.add_argument("--strategy", default="trend_pullback", choices=list(REGISTRY))
        sp.add_argument("--period", default="train", choices=["train", "validation", "test"])
        sp.add_argument("--params", help='JSON overrides, e.g. \'{"rsi_entry": 15}\'')
        sp.add_argument("--unlock-test", action="store_true", help="required to touch the test period")
        sp.add_argument("--note", default="", help="reason, written to the test access log")

    b = sub.add_parser("backtest", help="run one strategy on one period")
    common(b)
    b.add_argument("--delay", type=int, default=0)
    b.set_defaults(func=cmd_backtest)

    r = sub.add_parser("robustness", help="parameter perturbation, delay and Monte Carlo")
    common(r)
    r.set_defaults(func=cmd_robustness)

    w = sub.add_parser("walkforward", help="rolling re-optimisation with deflated Sharpe")
    common(w)
    w.add_argument("--grid", default='{"rsi_entry": [10, 20, 30], "stop_atr": [1.5, 2.0, 3.0]}')
    w.set_defaults(func=cmd_walkforward)

    gt = sub.add_parser("gate", help="check a test-period report against gates.yaml")
    gt.add_argument("backtest")
    gt.add_argument("--robustness")
    gt.set_defaults(func=cmd_gate)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except (splits.TestPeriodLocked, FileNotFoundError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
