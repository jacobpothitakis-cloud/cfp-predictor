"""Read access to the DuckDB database."""
from __future__ import annotations

from functools import lru_cache

import duckdb
import pandas as pd

from cfp.config import DB_PATH


def con() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise RuntimeError("data/cfp.duckdb not found. Run `cfp update` first.")
    return duckdb.connect(str(DB_PATH), read_only=True)


def table(name: str) -> pd.DataFrame:
    c = con()
    try:
        return c.execute(f"SELECT * FROM {name}").df()
    finally:
        c.close()


def has_table(name: str) -> bool:
    c = con()
    try:
        return name in set(c.execute("SHOW TABLES").df()["name"])
    finally:
        c.close()


def overrides() -> dict:
    import yaml
    from cfp.config import DATA
    p = DATA / "schedule_overrides.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


@lru_cache(maxsize=1)
def games() -> pd.DataFrame:
    g = table("games")
    g["start_date"] = pd.to_datetime(g["start_date"], utc=True)
    g["game_day"] = pd.to_datetime(g["game_day"])
    # fill placeholder ("TBD") opponents from data/schedule_overrides.yaml; drop unresolved placeholders
    mem = table("membership")
    tbd = (g["away_team"] == "TBD") | (g["home_team"] == "TBD")
    for fx in overrides().get("flex_games", []):
        row = g[tbd & (g["season"] == fx["season"]) & (g["home_team"] == fx["home"]) & ~g["completed"]]
        aw = mem[(mem["season"] == fx["season"]) & (mem["team"] == fx["away"])]
        if len(row) and len(aw):
            i, a = row.index[0], aw.iloc[0]
            g.loc[i, ["away_id", "away_team", "away_conf", "away_fbs", "away_power", "conf_game"]] = \
                [int(a.team_id), a.team, a.conf, True, bool(a.is_power), bool(fx.get("conference_game", False))]
    still = (g["away_team"] == "TBD") | (g["home_team"] == "TBD")
    return g[~still].reset_index(drop=True)


@lru_cache(maxsize=1)
def membership() -> pd.DataFrame:
    return table("membership")


@lru_cache(maxsize=1)
def rankings() -> pd.DataFrame:
    r = table("rankings")
    r["poll_date"] = pd.to_datetime(r["poll_date"], utc=True)
    return r


@lru_cache(maxsize=1)
def releases() -> pd.DataFrame:
    r = table("releases")
    r["cutoff"] = pd.to_datetime(r["cutoff"], utc=True)
    return r


def cfp_ranking(season: int, cutoff: pd.Timestamp) -> pd.DataFrame:
    r = rankings()
    return r[(r["poll"] == "cfp") & (r["season"] == season) & (r["poll_date"] == cutoff)].sort_values("rank")


def team_names() -> dict[int, str]:
    m = membership().sort_values("season")
    return dict(zip(m.team_id, m.team))


def clear_cache() -> None:
    games.cache_clear(); membership.cache_clear(); rankings.cache_clear(); releases.cache_clear()
