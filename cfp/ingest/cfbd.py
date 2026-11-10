"""CollegeFootballData API (primary source). Free tier has a monthly call cap, so every response
is cached permanently for completed seasons and the call budget is checked before each request.

Team ids and game ids in CFBD are the same integers ESPN uses, which is what lets the two
sources be reconciled game by game.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from cfp.config import CACHE, CFBD_BASE, CURRENT_SEASON, cfbd_key
from cfp.ingest import http

MIN_CALLS_LEFT = 25  # refuse to spend the last few calls of the month


def _remaining() -> int | None:
    log = CACHE / "fetch_log.jsonl"
    if not log.exists():
        return None
    last = None
    for line in log.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("namespace") == "cfbd" and d.get("remaining") is not None:
            last = d["remaining"]
    try:
        return int(last) if last is not None else None
    except ValueError:
        return None


def get(path: str, params: dict[str, Any], name: str, *, season: int | None = None,
        max_age_s: float | None = None, force: bool = False) -> Any:
    url = f"{CFBD_BASE}{path}"
    age = max_age_s if (season is not None and season >= CURRENT_SEASON) else None
    cached = http.cache_path("cfbd", name, url, params, "json")
    if not cached.exists() or force:
        left = _remaining()
        if left is not None and left < MIN_CALLS_LEFT:
            raise http.FetchError(f"CFBD monthly budget nearly exhausted ({left} calls left); not calling {path}")
    return http.get(url, params, namespace="cfbd", name=name, max_age_s=age, force=force,
                    headers={"Authorization": f"Bearer {cfbd_key()}"})


def games(season: int, season_type: str) -> pd.DataFrame:
    d = get("/games", {"year": season, "seasonType": season_type}, f"games_{season}_{season_type}",
            season=season, max_age_s=3600)
    return pd.json_normalize(d)


def fbs_teams(season: int) -> pd.DataFrame:
    return pd.json_normalize(get("/teams/fbs", {"year": season}, f"teams_fbs_{season}", season=season,
                                 max_age_s=30 * 86400))


def rankings(season: int, season_type: str) -> pd.DataFrame:
    d = get("/rankings", {"year": season, "seasonType": season_type}, f"rankings_{season}_{season_type}",
            season=season, max_age_s=3600)
    rows = []
    for wk in d:
        for poll in wk.get("polls", []):
            for r in poll.get("ranks", []):
                rows.append({"season": wk["season"], "season_type": wk["seasonType"], "week": wk["week"],
                             "poll": poll["poll"], "rank": r.get("rank"), "team": r.get("school"),
                             "team_id": r.get("teamId"), "conference": r.get("conference"),
                             "points": r.get("points"), "first_place": r.get("firstPlaceVotes")})
    return pd.DataFrame(rows)


def lines(season: int, season_type: str) -> pd.DataFrame:
    d = get("/lines", {"year": season, "seasonType": season_type}, f"lines_{season}_{season_type}",
            season=season, max_age_s=3600)
    rows = []
    for g in d:
        for ln in g.get("lines", []):
            rows.append({"game_id": g["id"], "provider": ln.get("provider"), "spread": ln.get("spread"),
                         "spread_open": ln.get("spreadOpen"), "over_under": ln.get("overUnder"),
                         "home_ml": ln.get("homeMoneyline"), "away_ml": ln.get("awayMoneyline")})
    return pd.DataFrame(rows)


def simple(path: str, season: int, name: str, **extra: Any) -> pd.DataFrame:
    return pd.json_normalize(get(path, {"year": season, **extra}, f"{name}_{season}", season=season,
                                 max_age_s=6 * 3600))


def sp(season: int) -> pd.DataFrame:
    return simple("/ratings/sp", season, "sp")


def fpi(season: int) -> pd.DataFrame:
    return simple("/ratings/fpi", season, "fpi")


def srs(season: int) -> pd.DataFrame:
    return simple("/ratings/srs", season, "srs")


def elo(season: int) -> pd.DataFrame:
    return simple("/ratings/elo", season, "elo")


def advanced_game_stats(season: int) -> pd.DataFrame:
    return simple("/stats/game/advanced", season, "adv_game")


def ppa_games(season: int) -> pd.DataFrame:
    return simple("/ppa/games", season, "ppa_games")


def pregame_wp(season: int) -> pd.DataFrame:
    return simple("/metrics/wp/pregame", season, "wp_pregame")


def talent(season: int) -> pd.DataFrame:
    return simple("/talent", season, "talent")
