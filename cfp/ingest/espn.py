"""ESPN public JSON endpoints: scoreboards, weekly rankings (CFP / AP / Coaches),
conference membership by season, and the FPI table (FPI, strength of record, SOS, game control).

No key needed. Everything goes through the disk cache. Completed past seasons are cached forever;
current-season pages are refreshed when older than `max_age_s`.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable

import pandas as pd

from cfp.config import CURRENT_SEASON, ESPN_BASE
from cfp.ingest import http

CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
STANDINGS = "https://site.api.espn.com/apis/v2/sports/football/college-football/standings"
FPI = "https://site.web.api.espn.com/apis/fitt/v3/sports/football/college-football/powerindex"
POLL_IDS = {"21": "cfp", "1": "ap", "2": "coaches"}


FORCE_CURRENT = False   # set by `cfp update` to re-fetch every current-season page


def _age(season: int, max_age_s: float | None) -> float | None:
    if season < CURRENT_SEASON:
        return None
    return 0.0 if FORCE_CURRENT else max_age_s


# ----------------------------------------------------------------------------- scoreboard
def scoreboard(season: int, season_type: int, week: int, *, max_age_s: float | None = 3600,
               force: bool = False) -> dict[str, Any]:
    return http.get(
        f"{ESPN_BASE}/scoreboard",
        {"dates": season, "seasontype": season_type, "week": week, "groups": 80, "limit": 400},
        namespace="espn", name=f"scoreboard_{season}_{season_type}_{week:02d}",
        max_age_s=_age(season, max_age_s), force=force,
    )


def _int(x: Any) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def parse_events(payload: dict[str, Any], season: int, season_type: int, week: int) -> list[dict]:
    rows = []
    for ev in payload.get("events", []):
        if not ev.get("competitions"):
            continue
        comp = ev["competitions"][0]
        st = comp["status"]["type"]
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        if "home" not in sides or "away" not in sides:
            continue
        h, a = sides["home"], sides["away"]
        completed = bool(st.get("completed"))
        hls = [_int(x.get("value")) for x in h.get("linescores", [])]
        als = [_int(x.get("value")) for x in a.get("linescores", [])]
        notes = "; ".join(n.get("headline", "") for n in comp.get("notes", []) if n.get("headline"))
        odds = (comp.get("odds") or [{}])[0] or {}
        rows.append({
            "game_id": int(ev["id"]),
            "season": season,
            "season_type": "regular" if season_type == 2 else "postseason",
            "week": week,
            "start_date": ev["date"],
            "neutral": bool(comp.get("neutralSite")),
            "conf_game": bool(comp.get("conferenceCompetition")),
            "home_id": int(h["team"]["id"]),
            "home_team": h["team"].get("location") or h["team"].get("displayName"),
            "away_id": int(a["team"]["id"]),
            "away_team": a["team"].get("location") or a["team"].get("displayName"),
            "home_pts": _int(h.get("score")) if completed else None,
            "away_pts": _int(a.get("score")) if completed else None,
            "home_ls": json.dumps(hls) if completed else None,
            "away_ls": json.dumps(als) if completed else None,
            "periods": max(len(hls), len(als)) if completed else None,
            "completed": completed,
            "status": st.get("name"),
            "home_rank_at_game": _rank(h),
            "away_rank_at_game": _rank(a),
            "notes": notes,
            "venue": (comp.get("venue") or {}).get("fullName"),
            "venue_city": ((comp.get("venue") or {}).get("address") or {}).get("city"),
            "espn_spread_home": _spread_home(odds, h, a),
            "espn_over_under": odds.get("overUnder"),
        })
    return rows


def _rank(c: dict) -> int | None:
    r = (c.get("curatedRank") or {}).get("current")
    return r if isinstance(r, int) and 1 <= r <= 25 else None


def _spread_home(odds: dict, h: dict, a: dict) -> float | None:
    """Home-team spread (negative = home favoured) when ESPN shows one."""
    sp = odds.get("spread")
    if sp is None:
        return None
    try:
        sp = float(sp)
    except (TypeError, ValueError):
        return None
    if (odds.get("homeTeamOdds") or {}).get("favorite"):
        return -abs(sp)
    if (odds.get("awayTeamOdds") or {}).get("favorite"):
        return abs(sp)
    return sp


def season_games(season: int, *, max_age_s: float | None = 3600) -> pd.DataFrame:
    rows: list[dict] = []
    for wk in range(1, 17):
        rows += parse_events(scoreboard(season, 2, wk, max_age_s=max_age_s), season, 2, wk)
    rows += parse_events(scoreboard(season, 3, 1, max_age_s=max_age_s), season, 3, 1)
    df = pd.DataFrame(rows)
    if df.empty:
        raise http.FetchError(f"ESPN returned no games for {season}")
    return df.drop_duplicates("game_id", keep="last").reset_index(drop=True)


# ----------------------------------------------------------------------------- rankings
def _core(url: str, name: str, season: int, max_age_s: float | None) -> Any:
    return http.get(url.replace("http://", "https://"), namespace="espn_core", name=name,
                    max_age_s=_age(season, max_age_s))


def week_rankings(season: int, season_type: int, week: int, *, max_age_s: float | None = 3600) -> list[dict]:
    base = f"{CORE}/seasons/{season}/types/{season_type}/weeks/{week}/rankings"
    try:
        idx = _core(base, f"rkidx_{season}_{season_type}_{week:02d}", season, max_age_s)
    except http.FetchError:
        return []
    rows = []
    for item in idx.get("items", []):
        m = re.search(r"/rankings/(\d+)", item["$ref"])
        if not m or m.group(1) not in POLL_IDS:
            continue
        pid = m.group(1)
        d = _core(f"{base}/{pid}", f"rk_{season}_{season_type}_{week:02d}_{pid}", season, max_age_s)
        for r in d.get("ranks", []):
            tm = re.search(r"/teams/(\d+)", r["team"]["$ref"])
            rows.append({
                "season": season, "season_type": "regular" if season_type == 2 else "postseason",
                "week": week, "poll": POLL_IDS[pid], "poll_date": d.get("date"),
                "rank": int(r["current"]), "team_id": int(tm.group(1)),
                "record": r.get("record", {}).get("summary"),
                "points": r.get("points"), "previous": r.get("previous"),
            })
    return rows


def season_rankings(season: int, *, max_age_s: float | None = 3600) -> pd.DataFrame:
    rows: list[dict] = []
    for wk in range(1, 17):
        rows += week_rankings(season, 2, wk, max_age_s=max_age_s)
    rows += week_rankings(season, 3, 1, max_age_s=max_age_s)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- membership
def season_membership(season: int, *, max_age_s: float | None = 86400) -> pd.DataFrame:
    """FBS teams with conference and division for one season (from the standings tree)."""
    d = http.get(STANDINGS, {"season": season, "group": 80}, namespace="espn",
                 name=f"standings_{season}", max_age_s=_age(season, max_age_s))
    rows: list[dict] = []

    def walk(node: dict, conf: str | None, conf_abbr: str | None, division: str | None) -> None:
        for e in (node.get("standings") or {}).get("entries", []):
            t = e["team"]
            rows.append({"season": season, "team_id": int(t["id"]), "team": t.get("location"),
                         "display_name": t.get("displayName"), "abbr": t.get("abbreviation"),
                         "conference": conf, "conf_abbr": conf_abbr, "division": division})
        for ch in node.get("children", []) or []:
            if conf is None:
                walk(ch, ch.get("name"), ch.get("abbreviation"), None)
            else:
                walk(ch, conf, conf_abbr, ch.get("name"))

    walk(d, None, None, None)
    df = pd.DataFrame(rows)
    if df.empty:
        raise http.FetchError(f"ESPN standings returned no FBS teams for {season}")
    return df.drop_duplicates("team_id")


# ----------------------------------------------------------------------------- FPI
def season_fpi(season: int, *, max_age_s: float | None = 6 * 3600) -> pd.DataFrame:
    """FPI table. For past seasons this is the END-OF-SEASON value (look-ahead if used in training)."""
    d = http.get(FPI, {"region": "us", "lang": "en", "season": season, "limit": 200},
                 namespace="espn", name=f"fpi_{season}", max_age_s=_age(season, max_age_s))
    cats = {c["name"]: c.get("names") or [] for c in d.get("categories", [])}
    rows = []
    for t in d.get("teams", []):
        row: dict[str, Any] = {"season": season, "team_id": int(t["team"]["id"]),
                               "fetched": d.get("lastUpdated")}
        for c in t.get("categories", []):
            names = cats.get(c["name"], [])
            for nm, val in zip(names, c.get("values") or c.get("totals") or []):
                try:
                    row[f"{c['name']}_{nm}".lower().replace("/", "_")] = float(val)
                except (TypeError, ValueError):
                    pass
        rows.append(row)
    return pd.DataFrame(rows)


def all_seasons(seasons: Iterable[int]) -> dict[str, pd.DataFrame]:
    g, r, m, f = [], [], [], []
    for s in seasons:
        g.append(season_games(s)); r.append(season_rankings(s)); m.append(season_membership(s))
        try:
            f.append(season_fpi(s))
        except http.FetchError:
            pass
    return {"games": pd.concat(g, ignore_index=True), "rankings": pd.concat(r, ignore_index=True),
            "membership": pd.concat(m, ignore_index=True),
            "fpi": pd.concat(f, ignore_index=True) if f else pd.DataFrame()}
