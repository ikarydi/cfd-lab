"""Evaluate test-period results against the pre-registered research gate."""
from __future__ import annotations


def research_gate(per_instrument: dict[str, dict], robustness: dict[str, dict], gates: dict) -> dict:
    """per_instrument: {inst: metrics.summary(...)} on the test period.
    robustness: {inst: {"min_perturbed_sharpe": x, "delay1_sharpe": y}}.
    """
    g = gates["research_gate"]
    rb = g["robustness"]
    checks = {}
    portfolio_sharpe = sum(m["sharpe"] for m in per_instrument.values()) / len(per_instrument)
    worst_dd = max(m["max_drawdown"] for m in per_instrument.values())
    profitable = sum(m["net_return"] > 0 for m in per_instrument.values())
    checks["net_sharpe"] = (portfolio_sharpe, portfolio_sharpe >= g["min_net_sharpe"])
    checks["max_drawdown"] = (worst_dd, worst_dd < g["max_drawdown"])
    checks["profitable_instruments"] = (profitable, profitable >= g["min_profitable_instruments"])
    if robustness:
        worst_pert = min(r["min_perturbed_sharpe"] for r in robustness.values())
        worst_delay = min(r["delay1_sharpe"] for r in robustness.values())
        checks["param_perturbation"] = (worst_pert, worst_pert >= rb["min_sharpe_under_perturbation"])
        checks["one_bar_delay"] = (worst_delay, worst_delay >= rb["one_bar_delay_min_sharpe"])
    return {"passed": all(ok for _, ok in checks.values()),
            "checks": {k: {"value": v, "pass": ok} for k, (v, ok) in checks.items()}}
