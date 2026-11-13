"""Weekly archive of ESPN's FPI / résumé table (FPI, strength of record, schedule strength, game control).

ESPN serves only the current values, and for a finished season only the end-of-season ones, so past weeks
cannot be recovered later. Each `cfp update` therefore appends the table as it stands to a committed CSV.
Once a season or two of weekly snapshots exist they can be tested as committee-model inputs without look-ahead.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cfp.config import DATA

SNAP_PATH = DATA / "snapshots" / "espn_fpi_weekly.csv"
KEY = ["season", "fetched"]          # `fetched` is ESPN's own last-updated stamp for the table


def load(path: Path = SNAP_PATH) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def append(fpi: pd.DataFrame, path: Path = SNAP_PATH, now: pd.Timestamp | None = None) -> int:
    """Store `fpi` (one season's table as just fetched) unless that ESPN update is already archived.
    Returns the number of rows added."""
    if fpi.empty:
        return 0
    now = now or pd.Timestamp.now(tz="UTC")
    new = fpi.copy()
    new["fetched"] = new["fetched"].fillna(str(now.date()))    # no stamp from ESPN: one snapshot per day
    old = load(path)
    if len(old):
        seen = set(map(tuple, old[KEY].drop_duplicates().to_numpy()))
        new = new[[k not in seen for k in map(tuple, new[KEY].to_numpy())]]
    if new.empty:
        return 0
    new.insert(2, "captured_utc", str(now)[:19])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([old, new], ignore_index=True).to_csv(path, index=False)
    return len(new)
