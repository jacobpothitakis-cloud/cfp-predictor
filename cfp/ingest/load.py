"""Build data/cfp.duckdb from the cached source responses.

Raw tables per source (espn_*, cfbd_*) are kept as fetched. The modelling tables
(`games`, `rankings`, `membership`) are unified views of them: CFBD is primary where it has been
ingested, ESPN fills in and cross-checks. Any final-score disagreement lands in `score_conflicts`.
"""
from __future__ import annotations

import os

import duckdb
import numpy as np
import pandas as pd

from cfp.config import CURRENT_SEASON, DB_PATH, FIRST_SEASON
from cfp.ingest import cfbd, espn, http

POWER_BY_ERA = {
    # conference abbreviations (ESPN) treated as "power" by season
    "p5": {"acc", "big10", "big12", "pac12", "sec"},
    "p4": {"acc", "big10", "big12", "sec"},
}
NOTRE_DAME_ID = 87


def power_set(season: int) -> set[str]:
    return POWER_BY_ERA["p5"] if season <= 2023 else POWER_BY_ERA["p4"]


def _norm_abbr(a: str | None) -> str | None:
    return a.lower().replace(" ", "").replace("-", "") if isinstance(a, str) else None


def build_membership(seasons: list[int]) -> pd.DataFrame:
    m = pd.concat([espn.season_membership(s) for s in seasons], ignore_index=True)
    m["conf"] = m["conf_abbr"].map(_norm_abbr)
    m["is_power"] = [(c in power_set(s)) or (t == NOTRE_DAME_ID) for c, s, t in zip(m.conf, m.season, m.team_id)]
    m["is_independent"] = m["conference"].str.contains("Independent", case=False, na=False)
    return m


def build_espn_games(seasons: list[int]) -> pd.DataFrame:
    g = pd.concat([espn.season_games(s) for s in seasons], ignore_index=True)
    g["start_date"] = pd.to_datetime(g["start_date"], utc=True)
    return g


def unify_games(eg: pd.DataFrame, mem: pd.DataFrame, cg: pd.DataFrame | None,
                lines: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = eg.copy()
    g["source"] = "espn"
    conflicts = pd.DataFrame(columns=["game_id", "season", "home_team", "away_team", "espn_home", "espn_away",
                                      "cfbd_home", "cfbd_away"])
    if cg is not None and len(cg):
        c = cg.rename(columns={"id": "game_id"}).copy()
        c = c[c["game_id"].isin(g["game_id"])]
        keep = ["game_id", "homePoints", "awayPoints", "homePregameElo", "awayPregameElo",
                "homePostgameWinProbability", "awayPostgameWinProbability", "excitementIndex",
                "neutralSite", "conferenceGame", "completed"]
        c = c[[k for k in keep if k in c.columns]]
        g = g.merge(c, on="game_id", how="left", suffixes=("", "_cfbd"))
        both = g["home_pts"].notna() & g["homePoints"].notna()
        bad = both & ((g["home_pts"] != g["homePoints"]) | (g["away_pts"] != g["awayPoints"]))
        conflicts = g.loc[bad, ["game_id", "season", "home_team", "away_team", "home_pts", "away_pts",
                                "homePoints", "awayPoints"]].rename(
            columns={"home_pts": "espn_home", "away_pts": "espn_away", "homePoints": "cfbd_home",
                     "awayPoints": "cfbd_away"})
        # CFBD is primary where it has a final score
        has = g["homePoints"].notna() & g["awayPoints"].notna()
        g.loc[has, "home_pts"] = g.loc[has, "homePoints"]
        g.loc[has, "away_pts"] = g.loc[has, "awayPoints"]
        g.loc[has, "source"] = "cfbd"
        g = g.rename(columns={"homePregameElo": "home_elo_pre", "awayPregameElo": "away_elo_pre",
                              "homePostgameWinProbability": "home_postgame_wp",
                              "excitementIndex": "excitement"})
    for col in ("home_elo_pre", "away_elo_pre", "home_postgame_wp", "excitement"):
        if col not in g.columns:
            g[col] = np.nan
    # closing line: CFBD consensus (median across books) if ingested, else ESPN's displayed spread
    g["spread_home"] = g["espn_spread_home"]
    g["spread_source"] = np.where(g["spread_home"].notna(), "espn", None)
    if lines is not None and len(lines):
        ln = lines.dropna(subset=["spread"]).copy()
        ln["spread"] = pd.to_numeric(ln["spread"], errors="coerce")
        med = ln.groupby("game_id")["spread"].median().rename("cfbd_spread")
        g = g.merge(med, on="game_id", how="left")
        has = g["cfbd_spread"].notna()
        g.loc[has, "spread_home"] = g.loc[has, "cfbd_spread"]
        g.loc[has, "spread_source"] = "cfbd"
    mm = mem[["season", "team_id", "conf", "division", "is_power"]]
    for side in ("home", "away"):
        g = g.merge(mm.rename(columns={"team_id": f"{side}_id", "conf": f"{side}_conf",
                                       "division": f"{side}_division", "is_power": f"{side}_power"}),
                    on=["season", f"{side}_id"], how="left")
        g[f"{side}_fbs"] = g[f"{side}_conf"].notna()
        g[f"{side}_power"] = g[f"{side}_power"].fillna(False).astype(bool)
    g = g[g["home_fbs"] | g["away_fbs"]].copy()
    g["overtime"] = g["periods"].fillna(0) > 4
    note = g["notes"].fillna("").str.lower()
    g["is_conf_champ"] = (g["season_type"] == "regular") & note.str.contains("championship") & \
        ~note.str.contains("playoff|bowl")
    # US game day (kickoffs after midnight UTC belong to the previous calendar day)
    g["game_day"] = (g["start_date"] - pd.Timedelta(hours=8)).dt.date
    return g.sort_values(["start_date", "game_id"]).reset_index(drop=True), conflicts


def build_rankings(seasons: list[int], games: pd.DataFrame) -> pd.DataFrame:
    """All polls. ESPN omits the date on some early seasons; those are derived from the schedule:
    the "week N" ranking is the one released on the Tuesday before week N's Saturday (08:00 UTC,
    matching ESPN's own convention), and a postseason-week ranking is the Sunday after the last
    regular-season Saturday."""
    r = pd.concat([espn.season_rankings(s) for s in seasons], ignore_index=True)
    r["poll_date"] = pd.to_datetime(r["poll_date"], utc=True)
    reg = games[games["season_type"] == "regular"].copy()
    reg["day"] = (reg["start_date"] - pd.Timedelta(hours=8)).dt.normalize()
    sat = reg[reg["day"].dt.day_of_week == 5].groupby(["season", "week"])["day"].agg(lambda x: x.mode().iloc[0])
    last_sat = reg[reg["day"].dt.day_of_week == 5].groupby("season")["day"].max()

    def derive(season: int, stype: str, week: int) -> pd.Timestamp:
        if stype == "postseason" or (season, week) not in sat.index:
            return last_sat[season] + pd.Timedelta(days=1, hours=8)
        return sat[(season, week)] - pd.Timedelta(days=4) + pd.Timedelta(hours=8)

    miss = r["poll_date"].isna()
    r["date_derived"] = miss
    if miss.any():
        r.loc[miss, "poll_date"] = [derive(a, b, c) for a, b, c in
                                    zip(r.loc[miss, "season"], r.loc[miss, "season_type"], r.loc[miss, "week"])]
        # AP / Coaches come out on Sunday, two days before the committee's Tuesday
        ap = miss & (r["poll"] != "cfp")
        r.loc[ap, "poll_date"] = r.loc[ap, "poll_date"] - pd.Timedelta(days=2)
        # the last committee ranking of a season is Selection Day: the Sunday after the title games
        ccg = reg[reg["notes"].fillna("").str.contains("Championship")].groupby("season")["day"].max()
        last_wk = r[r["poll"] == "cfp"].groupby("season")["week"].transform("max")
        fin = miss & (r["poll"] == "cfp") & (r["season_type"] == "regular")
        fin = fin & (r["week"] == last_wk.reindex(r.index)) & r["season"].isin(ccg.index)
        r.loc[fin, "poll_date"] = r.loc[fin, "season"].map(ccg) + pd.Timedelta(days=1, hours=8)
    return r


def build_releases(rk: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct CFP committee ranking. A ranking that ESPN lists twice (same 25 teams in
    the same order under two week labels) is kept once."""
    c = rk[rk["poll"] == "cfp"].sort_values(["season", "poll_date", "season_type", "week", "rank"])
    rows, seen = [], set()
    for (season, stype, week), grp in c.groupby(["season", "season_type", "week"], sort=False):
        sig = (season, tuple(grp.sort_values("rank")["team_id"]))
        if sig in seen:
            continue
        seen.add(sig)
        rows.append({"season": season, "season_type": stype, "week": week, "cutoff": grp["poll_date"].iloc[0],
                     "date_derived": bool(grp["date_derived"].iloc[0])})
    rel = pd.DataFrame(rows).sort_values(["season", "cutoff"]).reset_index(drop=True)
    rel["release_no"] = rel.groupby("season").cumcount() + 1
    rel["n_releases"] = rel.groupby("season")["release_no"].transform("max")
    # the last ranking of a completed season is the Selection Day ranking
    rel["is_final"] = (rel["release_no"] == rel["n_releases"]) & (rel["season"] < CURRENT_SEASON)
    return rel


def build(use_cfbd: bool | None = None, seasons: list[int] | None = None) -> dict[str, int]:
    seasons = seasons or list(range(FIRST_SEASON, CURRENT_SEASON + 1))
    if use_cfbd is None:
        use_cfbd = bool(os.environ.get("CFBD_API_KEY", "").strip())
    mem = build_membership(seasons)
    eg = build_espn_games(seasons)
    rk = build_rankings(seasons, eg)
    fpi = pd.concat([espn.season_fpi(s) for s in seasons], ignore_index=True)
    tables: dict[str, pd.DataFrame] = {"espn_games": eg, "membership": mem, "espn_rankings": rk, "espn_fpi": fpi}
    cg = lines = None
    if use_cfbd:
        cgs, lns, extra = [], [], {k: [] for k in ("sp", "srs", "elo", "cfbd_fpi", "adv", "ppa", "wp", "cfbd_rank")}
        for s in seasons:
            for st in ("regular", "postseason"):
                cgs.append(cfbd.games(s, st)); lns.append(cfbd.lines(s, st))
                extra["cfbd_rank"].append(cfbd.rankings(s, st))
            for key, fn in (("sp", cfbd.sp), ("srs", cfbd.srs), ("elo", cfbd.elo), ("cfbd_fpi", cfbd.fpi),
                            ("adv", cfbd.advanced_game_stats), ("ppa", cfbd.ppa_games), ("wp", cfbd.pregame_wp)):
                df = fn(s)
                if "year" not in df.columns and "season" not in df.columns:
                    df["season"] = s
                extra[key].append(df)
        cg = pd.concat(cgs, ignore_index=True)
        lines = pd.concat(lns, ignore_index=True)
        tables["cfbd_games"] = cg
        tables["cfbd_lines"] = lines
        names = {"sp": "cfbd_sp", "srs": "cfbd_srs", "elo": "cfbd_elo", "cfbd_fpi": "cfbd_fpi",
                 "adv": "cfbd_adv_game", "ppa": "cfbd_ppa_game", "wp": "cfbd_pregame_wp",
                 "cfbd_rank": "cfbd_rankings"}
        for k, v in extra.items():
            tables[names[k]] = pd.concat([d for d in v if len(d)], ignore_index=True)
    games, conflicts = unify_games(eg, mem, cg, lines)
    tables["games"] = games
    tables["score_conflicts"] = conflicts
    tables["rankings"] = rk
    tables["releases"] = build_releases(rk)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    counts = {}
    for name, df in tables.items():
        df = df.copy()
        for c in df.columns:  # nested lists/dicts are stored as JSON text
            if df[c].map(lambda x: isinstance(x, (list, dict))).any():
                df[c] = df[c].map(lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else str(x))
        con.register("_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
        con.unregister("_df")
        counts[name] = len(df)
    con.close()
    return counts


if __name__ == "__main__":
    try:
        for k, v in build().items():
            print(f"{k:18s} {v:>8,d}")
    except http.FetchError as e:
        raise SystemExit(f"INGEST STOPPED: {e}")
