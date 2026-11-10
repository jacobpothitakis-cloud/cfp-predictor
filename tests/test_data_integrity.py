"""Data-integrity tests. They run against data/cfp.duckdb and are skipped if it has not been built."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from cfp.config import CURRENT_SEASON, DB_PATH, FIRST_SEASON

pytestmark = pytest.mark.data
if not DB_PATH.exists():
    pytest.skip("database not built; run `cfp update`", allow_module_level=True)

from cfp import data  # noqa: E402


@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    return data.games()


@pytest.fixture(scope="module")
def mem() -> pd.DataFrame:
    return data.membership()


def test_no_duplicate_game_ids(games):
    assert not games["game_id"].duplicated().any()


def test_no_duplicate_matchups_same_day(games):
    g = games[games["completed"]]
    key = g.apply(lambda r: (r.season, str(r.game_day), min(r.home_id, r.away_id), max(r.home_id, r.away_id)), axis=1)
    assert not key.duplicated().any()


def test_every_final_has_a_score(games):
    g = games[games["completed"]]
    assert g["home_pts"].notna().all() and g["away_pts"].notna().all()
    assert (g["home_pts"] >= 0).all() and (g["away_pts"] >= 0).all()


def test_no_ties_in_completed_games(games):
    g = games[games["completed"]]
    assert (g["home_pts"] != g["away_pts"]).all()


def test_line_scores_sum_to_final(games):
    g = games[games["completed"] & games["home_ls"].notna()]
    bad = 0
    for r in g.itertuples():
        hl, al = json.loads(r.home_ls), json.loads(r.away_ls)
        if hl and al and None not in hl and None not in al:
            bad += (sum(hl) != r.home_pts) or (sum(al) != r.away_pts)
    assert bad / max(len(g), 1) < 0.005, f"{bad} games whose quarter scores do not sum to the final"


def test_every_season_present(games):
    assert set(games["season"]) == set(range(FIRST_SEASON, CURRENT_SEASON + 1))


def test_fbs_team_counts(mem):
    n = mem.groupby("season")["team_id"].nunique()
    assert n.between(125, 140).all(), n.to_dict()
    assert n[CURRENT_SEASON] == 138  # the committee chair's own count on Nov 3, 2026


def test_each_fbs_team_has_a_full_regular_season(games, mem):
    """Completed seasons: every FBS team played 11-14 regular-season games (2020 excepted: COVID)."""
    reg = games[(games["season_type"] == "regular") & games["completed"] & ~games["is_conf_champ"]]
    long = pd.concat([reg[["season", "home_id"]].rename(columns={"home_id": "team_id"}),
                      reg[["season", "away_id"]].rename(columns={"away_id": "team_id"})])
    cnt = long.groupby(["season", "team_id"]).size().rename("n").reset_index()
    cnt = mem[["season", "team_id", "team"]].merge(cnt, on=["season", "team_id"], how="left").fillna({"n": 0})
    full = cnt[(cnt["season"] != 2020) & (cnt["season"] < CURRENT_SEASON)]
    bad = full[~full["n"].between(11, 13)]
    # Virginia played 10 games in 2022: its last two were cancelled after the Nov 13 campus shooting.
    bad = bad[~((bad["season"] == 2022) & (bad["team"] == "Virginia"))]
    assert bad.empty, bad.to_string()


def test_current_season_schedule_is_twelve_games(games, mem):
    reg = games[(games["season"] == CURRENT_SEASON) & (games["season_type"] == "regular") & ~games["is_conf_champ"]]
    reg = reg[~reg["status"].isin(["STATUS_CANCELED", "STATUS_POSTPONED"])]
    long = pd.concat([reg["home_id"], reg["away_id"]]).value_counts()
    ids = mem.loc[mem["season"] == CURRENT_SEASON, "team_id"]
    n = long.reindex(ids).fillna(0)
    short = n[~n.between(12, 13)]
    # The rebuilt Pac-12 plays a week-13 "flex" game between members whose pairings are confirmed
    # only six days out (pac-12.com, 2026 schedule release). Until ESPN lists a team's flex game it
    # shows 11 games. Those games are supplied by data/schedule_overrides.yaml.
    pac12 = set(mem.loc[(mem["season"] == CURRENT_SEASON) & (mem["conf"] == "pac12"), "team_id"])
    unexpected = short[[t not in pac12 or v != 11 for t, v in short.items()]]
    assert unexpected.empty, unexpected.to_dict()


def test_records_reconcile_with_second_source(games):
    """W-L computed from our games table vs. the record ESPN prints next to each ranked team in the
    CFP ranking (an independent field of the rankings feed, not derived from the scoreboard)."""
    rk = data.rankings()
    rk = rk[(rk["poll"] == "cfp") & rk["record"].notna()]
    g = games[games["completed"]]
    mism = []
    for (season, date), grp in rk.groupby(["season", "poll_date"]):
        gg = g[(g["season"] == season) & (g["start_date"] < date)]
        w = pd.concat([gg.loc[gg.home_pts > gg.away_pts, "home_id"], gg.loc[gg.away_pts > gg.home_pts, "away_id"]]).value_counts()
        l = pd.concat([gg.loc[gg.home_pts < gg.away_pts, "home_id"], gg.loc[gg.away_pts < gg.home_pts, "away_id"]]).value_counts()
        for r in grp.itertuples():
            ours = f"{int(w.get(r.team_id, 0))}-{int(l.get(r.team_id, 0))}"
            if ours != r.record:
                mism.append((season, str(date)[:10], r.team_id, ours, r.record))
    assert len(mism) <= 0.01 * len(rk), mism[:20]


def test_cfp_ranking_releases(games):
    rk = data.rankings()
    c = rk[rk["poll"] == "cfp"]
    per = c.groupby(["season", "poll_date"])["rank"].agg(["size", "min", "max"])
    assert (per["size"] == 25).all() and (per["min"] == 1).all() and (per["max"] == 25).all(), per[per["size"] != 25]
    n = c.groupby("season")["poll_date"].nunique()
    assert all(n.get(s, 0) >= 4 for s in range(FIRST_SEASON, CURRENT_SEASON)), n.to_dict()


def test_first_2026_ranking_matches_the_published_one():
    """Sanity check of the ingest against the list in the project brief (not a data source)."""
    rk = data.rankings()
    c = rk[(rk["poll"] == "cfp") & (rk["season"] == 2026)]
    first = c[c["poll_date"] == c["poll_date"].min()].sort_values("rank")
    names = data.team_names()
    got = [names[t] for t in first["team_id"]]
    assert got[:12] == ["Miami", "Georgia", "Utah", "Indiana", "Alabama", "Kentucky", "Texas", "UCLA", "BYU",
                        "Ole Miss", "Notre Dame", "Ohio State"], got


def test_score_conflicts_are_reported_not_hidden():
    c = data.table("score_conflicts")
    assert len(c) < 30, c.head(30).to_string()
