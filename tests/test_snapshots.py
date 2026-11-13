import pandas as pd

from cfp.ingest import snapshots


def _table(stamp, fpi):
    return pd.DataFrame({"season": [2026, 2026], "team_id": [1, 2], "fetched": [stamp, stamp], "fpi_fpi": fpi})


def test_snapshot_is_appended_once_per_espn_update(tmp_path):
    p = tmp_path / "snap.csv"
    assert snapshots.append(_table("2026-11-10T09:00Z", [20.0, 10.0]), p) == 2
    assert snapshots.append(_table("2026-11-10T09:00Z", [20.0, 10.0]), p) == 0      # same ESPN update: no duplicate
    assert snapshots.append(_table("2026-11-17T09:00Z", [21.0, 9.0]), p) == 2
    d = snapshots.load(p)
    assert len(d) == 4 and d["fetched"].nunique() == 2
    assert d[d.fetched == "2026-11-10T09:00Z"]["fpi_fpi"].tolist() == [20.0, 10.0]  # earlier week is untouched


def test_snapshot_without_espn_stamp_is_keyed_by_day(tmp_path):
    p = tmp_path / "snap.csv"
    t = _table(None, [1.0, 2.0])
    now = pd.Timestamp("2026-11-15 18:00", tz="UTC")
    assert snapshots.append(t, p, now=now) == 2
    assert snapshots.append(t, p, now=now + pd.Timedelta(hours=3)) == 0
    assert snapshots.append(t, p, now=now + pd.Timedelta(days=7)) == 2
