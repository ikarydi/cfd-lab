"""Train / validation / test windows, with the test window locked.

Reading the test window requires an explicit unlock and is appended to
logs/test_access.log, so every look at untouched data leaves a record.
"""
from __future__ import annotations

import datetime as dt
import getpass
from pathlib import Path

import yaml

GATES = Path("gates.yaml")
ACCESS_LOG = Path("logs/test_access.log")


class TestPeriodLocked(PermissionError):
    pass


def load_gates(path: str | Path = GATES) -> dict:
    return yaml.safe_load(Path(path).read_text())


def window(name: str, gates: dict | None = None, unlock_test: bool = False, note: str = "") -> tuple[str, str]:
    gates = gates or load_gates()
    start, end = (str(x) for x in gates["splits"][name])
    if name == "test":
        if not unlock_test:
            raise TestPeriodLocked(
                "The test period is untouched data. Pass --unlock-test (CLI) or unlock_test=True "
                "only for the final, pre-registered evaluation."
            )
        ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ACCESS_LOG.open("a") as f:
            f.write(f"{dt.datetime.now(dt.UTC).isoformat()}\t{getpass.getuser()}\t{note}\n")
    return start, end
