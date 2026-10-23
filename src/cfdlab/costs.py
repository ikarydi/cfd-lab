"""CFD cost model: spread, slippage, commission and daily overnight financing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG = Path("config/instruments.yaml")


@dataclass(frozen=True)
class CostModel:
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    commission_pct: float = 0.0
    financing_long: float = 0.0
    financing_short: float = 0.0
    day_basis: int = 360

    def fill_price(self, price: float, side: int) -> float:
        """Price actually paid (side=+1 buy) or received (side=-1 sell)."""
        adverse = (self.spread_bps / 2 + self.slippage_bps) / 1e4
        return price * (1 + side * adverse)

    def commission(self, notional: float) -> float:
        return abs(notional) * self.commission_pct / 100

    def financing(self, notional: float, direction: int, days: int) -> float:
        """Overnight cost for holding `days` calendar days. Positive means you pay."""
        rate = self.financing_long if direction > 0 else self.financing_short
        return abs(notional) * rate / self.day_basis * days

    @classmethod
    def from_config(cls, instrument: str, path: str | Path = CONFIG) -> "CostModel":
        cfg = yaml.safe_load(Path(path).read_text())["instruments"][instrument]
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in cfg.items() if k in fields})


ZERO = CostModel()
