"""SeasonState: everything known about one season as of a cutoff time, as plain arrays.

FBS teams occupy indices 0..n-1. Index n is the pooled FCS pseudo-team. The same structure
feeds the historical feature build (one "simulation") and the Monte Carlo (many).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from cfp import data, ratings as ratings_mod
from cfp.ratings.model import FCS_ID, ns

G6_CONFS = {"american", "usa", "midam", "mwest", "belt", "pac12"}  # 2026 definition (see docs/committee_protocol.md)
GC_SIGMA = 13.0


def g6_set(season: int) -> set[str]:
    """Non-power conferences. The Pac-12 is a power league through 2023 and Group of Six from 2026;
    in 2024-25 its two remaining members were treated as neither P4 nor an automatic-bid league."""
    return G6_CONFS if season >= 2026 else {"american", "usa", "midam", "mwest", "belt"}


@dataclass
class GameArrays:
    gh: np.ndarray            # home team index (FCS -> n)
    ga: np.ndarray            # away team index
    neutral: np.ndarray
    conf_game: np.ndarray
    ccg: np.ndarray
    day: np.ndarray           # days since Aug 1 of the season
    rank_h: np.ndarray        # poll rank at game time (nan = unranked)
    rank_a: np.ndarray
    pg_wp_h: np.ndarray       # postgame win expectancy, home side (nan if not available)
    gc_h: np.ndarray          # game-control proxy from quarter scores, home side (nan if not available)
    game_id: np.ndarray
    week: np.ndarray

    def __len__(self) -> int:
        return len(self.gh)

    def take(self, m: np.ndarray) -> "GameArrays":
        return GameArrays(**{k: getattr(self, k)[m] for k in self.__dataclass_fields__})

    @staticmethod
    def concat(a: "GameArrays", b: "GameArrays") -> "GameArrays":
        return GameArrays(**{k: np.concatenate([getattr(a, k), getattr(b, k)]) for k in a.__dataclass_fields__})


@dataclass
class SeasonState:
    season: int
    cutoff: pd.Timestamp
    team_ids: np.ndarray
    names: list[str]
    conf: np.ndarray           # conference code per team
    division: np.ndarray
    power: np.ndarray
    indep: np.ndarray
    g6: np.ndarray
    played: GameArrays
    margin: np.ndarray         # (G,) home margin of played games
    home_pts: np.ndarray
    away_pts: np.ndarray
    future: GameArrays
    rating: np.ndarray         # (n+1,) as of cutoff
    ap: np.ndarray             # (n,) latest AP rank available before the cutoff (nan = unranked)
    coaches: np.ndarray
    prev_cfp: np.ndarray       # previous committee ranking this season (nan if none / unranked)
    has_prev_cfp: bool
    preseason_ap: np.ndarray
    cutoff_day: float
    idx: dict[int, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.team_ids)


def game_control_home(home_ls: str | None, away_ls: str | None, margin: float) -> float:
    """Proxy for 'game control' (average in-game win probability) from quarter scores:
    mean of the home side's win probability at the end of Q1, Q2, Q3 and the final result.
    Play-by-play win probability is not in the free data, so this is a documented stand-in."""
    try:
        h, a = json.loads(home_ls), json.loads(away_ls)
        leads = np.cumsum(np.array(h[:3], float) - np.array(a[:3], float))
        if len(leads) < 3 or np.isnan(leads).any():
            raise ValueError
    except Exception:
        return np.nan
    rem = np.array([0.75, 0.5, 0.25])
    wp = norm.cdf(leads / (GC_SIGMA * np.sqrt(rem)))
    return float((wp.sum() + (1.0 if margin > 0 else 0.0)) / 4)


def game_control_from_margin(margin: np.ndarray) -> np.ndarray:
    """Fallback used for simulated games: assume the lead grew linearly to the final margin."""
    q = np.array([0.25, 0.5, 0.75])
    wp = norm.cdf(margin[..., None] * q / (GC_SIGMA * np.sqrt(1 - q)))
    return (wp.sum(-1) + (margin > 0)) / 4


def _arrays(g: pd.DataFrame, idx: dict[int, int], n: int, season_start: pd.Timestamp) -> GameArrays:
    def ix(ids: pd.Series, fbs: pd.Series) -> np.ndarray:
        return np.array([idx[int(t)] if f and int(t) in idx else n for t, f in zip(ids, fbs)], int)

    day = (ns(g["start_date"]) - season_start.value) / 86400e9 if len(g) else np.array([])
    marg = (g["home_pts"] - g["away_pts"]).to_numpy(float) if len(g) else np.array([])
    gc = np.array([game_control_home(h, a, m) for h, a, m in zip(g["home_ls"], g["away_ls"], marg)], float)
    return GameArrays(
        gh=ix(g["home_id"], g["home_fbs"]), ga=ix(g["away_id"], g["away_fbs"]),
        neutral=g["neutral"].to_numpy(bool), conf_game=g["conf_game"].to_numpy(bool),
        ccg=g["is_conf_champ"].to_numpy(bool), day=np.asarray(day, float),
        rank_h=g["home_rank_at_game"].to_numpy(float), rank_a=g["away_rank_at_game"].to_numpy(float),
        pg_wp_h=g["home_postgame_wp"].to_numpy(float), gc_h=gc,
        game_id=g["game_id"].to_numpy(), week=g["week"].to_numpy(),
    )


def _poll_vec(rk: pd.DataFrame, idx: dict[int, int], n: int) -> np.ndarray:
    v = np.full(n, np.nan)
    for t, r in zip(rk["team_id"], rk["rank"]):
        if int(t) in idx:
            v[idx[int(t)]] = r
    return v


def build_state(season: int, cutoff: pd.Timestamp, *, games: pd.DataFrame | None = None,
                R: ratings_mod.Ratings | None = None) -> SeasonState:
    """State of `season` using only games that kicked off before `cutoff` and polls dated before it."""
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    g_all = data.games() if games is None else games
    g = g_all[g_all["season"] == season]
    mem = data.membership()
    m = mem[mem["season"] == season].sort_values("team_id")
    team_ids = m["team_id"].to_numpy(int)
    idx = {int(t): i for i, t in enumerate(team_ids)}
    n = len(team_ids)
    start = pd.Timestamp(f"{season}-08-01", tz="UTC")
    reg = g[g["season_type"] == "regular"]
    done = reg[reg["completed"] & (reg["start_date"] < cutoff)].sort_values(["start_date", "game_id"])
    # games still to be played as of the cutoff (for a past cutoff: the ones that were then in the future)
    fut = reg[(reg["start_date"] >= cutoff) & ~reg["status"].isin(["STATUS_CANCELED", "STATUS_POSTPONED"])]
    fut = fut.sort_values(["start_date", "game_id"])
    R = R or ratings_mod.load()
    tab = R.asof(season, cutoff)
    rating = np.array([tab.get(int(t), R.p.new_fbs_prior) for t in team_ids] + [tab.get(FCS_ID, R.p.fcs_prior)])
    rk = data.rankings()
    rk = rk[rk["season"] == season]

    def latest(poll: str, strict_before: pd.Timestamp) -> pd.DataFrame:
        p = rk[(rk["poll"] == poll) & (rk["poll_date"] < strict_before)]
        return p[p["poll_date"] == p["poll_date"].max()] if len(p) else p

    ap, co, prev = latest("ap", cutoff), latest("coaches", cutoff), latest("cfp", cutoff)
    pre = rk[(rk["poll"] == "ap") & (rk["season_type"] == "regular") & (rk["week"] == 1)]
    conf = m["conf"].to_numpy(str)
    g6 = np.isin(conf, list(g6_set(season)))
    return SeasonState(
        season=season, cutoff=cutoff, team_ids=team_ids, names=m["team"].tolist(), conf=conf,
        division=m["division"].fillna("").to_numpy(str), power=m["is_power"].to_numpy(bool),
        indep=m["is_independent"].to_numpy(bool), g6=g6,
        played=_arrays(done, idx, n, start), margin=(done["home_pts"] - done["away_pts"]).to_numpy(float),
        home_pts=done["home_pts"].to_numpy(float), away_pts=done["away_pts"].to_numpy(float),
        future=_arrays(fut, idx, n, start), rating=rating,
        ap=_poll_vec(ap, idx, n), coaches=_poll_vec(co, idx, n), prev_cfp=_poll_vec(prev, idx, n),
        has_prev_cfp=len(prev) > 0, preseason_ap=_poll_vec(pre, idx, n),
        cutoff_day=(cutoff.value - start.value) / 86400e9, idx=idx,
    )
