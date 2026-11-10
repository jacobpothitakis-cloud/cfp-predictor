"""Layer 1: team strength.

A walk-forward, opponent-adjusted margin rating. For each season and each weekly boundary
(Tuesday 12:00 UTC) we solve a ridge regression on capped scoring margins of the games played
before the boundary, shrunk toward a preseason prior built from last season's final rating.
A game is always predicted with the snapshot taken before it kicked off, so there is no
look-ahead in anything downstream that uses these ratings.

All FCS opponents share one pseudo-team (id -1): a documented proxy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

FCS_ID = -1


@dataclass(frozen=True)
class RatingParams:
    cap: float = 42.0          # margin cap (points)
    hfa: float = 2.6           # home-field advantage (points)
    ridge: float = 3.0         # prior strength, in "games"
    carry: float = 0.74        # share of last season's final rating carried into the prior
    fcs_prior: float = -19.0   # prior rating of the pooled FCS team vs. FBS average 0
    new_fbs_prior: float = -12.0
    sigma: float = 14.0        # sd of margin around the rating difference
    recency: float = 0.0       # per-week extra weight on recent games (0 = none)


def weekly_boundaries(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Tuesdays 12:00 UTC from the Tuesday before `start` through the one after `end`."""
    d = (start - pd.Timedelta(days=(start.weekday() - 1) % 7)).normalize() + pd.Timedelta(hours=12)
    if d > start:
        d -= pd.Timedelta(days=7)
    out = []
    while d <= end + pd.Timedelta(days=7):
        out.append(d)
        d += pd.Timedelta(days=7)
    return out


def ns(x: pd.Series) -> np.ndarray:
    """Timestamps as int64 nanoseconds since the epoch (UTC), whatever the stored resolution."""
    return pd.to_datetime(x, utc=True).dt.as_unit("ns").astype("int64").to_numpy()


def _team_key(g: pd.DataFrame, side: str) -> np.ndarray:
    return np.where(g[f"{side}_fbs"].to_numpy(), g[f"{side}_id"].to_numpy(), FCS_ID)


def solve_ridge(home: np.ndarray, away: np.ndarray, y: np.ndarray, w: np.ndarray, teams: np.ndarray,
                prior: np.ndarray, lam: np.ndarray) -> np.ndarray:
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    A = np.diag(lam.astype(float))
    b = lam * prior
    if len(y):
        hi = np.fromiter((idx[t] for t in home), int, len(home))
        ai = np.fromiter((idx[t] for t in away), int, len(away))
        np.add.at(A, (hi, hi), w); np.add.at(A, (ai, ai), w)
        np.add.at(A, (hi, ai), -w); np.add.at(A, (ai, hi), -w)
        np.add.at(b, hi, w * y); np.add.at(b, ai, -w * y)
    return np.linalg.solve(A, b)


def season_snapshots(g: pd.DataFrame, fbs_ids: np.ndarray, prior: dict[int, float],
                     p: RatingParams) -> pd.DataFrame:
    """Ratings at every weekly boundary of one season. g: that season's games (all rows)."""
    done = g[g["completed"] & g["home_pts"].notna()].sort_values("start_date")
    teams = np.concatenate([np.sort(fbs_ids), [FCS_ID]])
    pr = np.array([prior.get(int(t), p.new_fbs_prior) for t in teams], float)
    pr[-1] = p.fcs_prior
    lam = np.full(len(teams), p.ridge); lam[-1] = p.ridge * 4
    bounds = weekly_boundaries(g["start_date"].min(), g["start_date"].max())
    home = _team_key(done, "home"); away = _team_key(done, "away")
    marg = (done["home_pts"] - done["away_pts"]).to_numpy(float)
    y = np.clip(marg, -p.cap, p.cap) - np.where(done["neutral"].to_numpy(), 0.0, p.hfa)
    t = ns(done["start_date"])
    rows = []
    for bnd in bounds:
        m = t < bnd.value
        w = np.ones(m.sum())
        if p.recency and m.any():
            age = (bnd.value - t[m]) / (7 * 86400e9)
            w = np.exp(-p.recency * age)
        r = solve_ridge(home[m], away[m], y[m], w, teams, pr, lam)
        rows.append(pd.DataFrame({"boundary": bnd, "team_id": teams, "rating": r, "n_games": int(m.sum())}))
    out = pd.concat(rows, ignore_index=True)
    out["season"] = int(g["season"].iloc[0])
    return out


def build_all(games: pd.DataFrame, membership: pd.DataFrame, p: RatingParams = RatingParams()) -> pd.DataFrame:
    """Walk-forward snapshots for every season. Returns season, boundary, team_id, rating."""
    out = []
    prev_final: dict[int, float] = {}
    for season in sorted(games["season"].unique()):
        g = games[games["season"] == season]
        fbs = membership.loc[membership["season"] == season, "team_id"].to_numpy()
        prior = {int(t): p.carry * prev_final[t] for t in fbs if t in prev_final}
        snap = season_snapshots(g, fbs, prior, p)
        out.append(snap)
        last = snap[snap["boundary"] == snap["boundary"].max()]
        prev_final = dict(zip(last["team_id"].astype(int), last["rating"]))
    return pd.concat(out, ignore_index=True)


class Ratings:
    """Lookup of walk-forward ratings as of any timestamp."""

    def __init__(self, snaps: pd.DataFrame, p: RatingParams = RatingParams()):
        self.p = p
        self.snaps = snaps
        self._by_season: dict[int, tuple[np.ndarray, list[dict[int, float]]]] = {}
        snaps = snaps.assign(_b=ns(snaps["boundary"]))
        for season, s in snaps.groupby("season"):
            bs = np.sort(s["_b"].unique())
            tabs = [dict(zip(x["team_id"].astype(int), x["rating"])) for _, x in s.groupby("_b", sort=True)]
            self._by_season[int(season)] = (bs, tabs)

    def asof(self, season: int, when: pd.Timestamp) -> dict[int, float]:
        """Ratings from the last weekly boundary at or before `when`."""
        bs, tabs = self._by_season[season]
        i = int(np.searchsorted(bs, pd.Timestamp(when).value, side="right")) - 1
        return tabs[max(i, 0)]

    def pregame(self, games: pd.DataFrame) -> pd.DataFrame:
        """Adds home_r, away_r, mu (expected home margin), p_home for each game row."""
        hr, ar = np.empty(len(games)), np.empty(len(games))
        hk = _team_key(games, "home"); ak = _team_key(games, "away")
        for n, (season, when) in enumerate(zip(games["season"].to_numpy(), ns(games["start_date"]))):
            tab = self.asof(int(season), pd.Timestamp(when, tz="UTC"))
            hr[n] = tab.get(int(hk[n]), self.p.new_fbs_prior)
            ar[n] = tab.get(int(ak[n]), self.p.new_fbs_prior)
        out = games.copy()
        out["home_r"], out["away_r"] = hr, ar
        out["mu"] = hr - ar + np.where(games["neutral"].to_numpy(), 0.0, self.p.hfa)
        out["p_home"] = win_prob(out["mu"].to_numpy(), self.p.sigma)
        return out


def win_prob(mu: np.ndarray | float, sigma: float) -> np.ndarray:
    return norm.cdf(np.asarray(mu, float) / sigma)


def evaluate(pred: pd.DataFrame, sigma: float | None = None) -> dict[str, float]:
    """Accuracy / log loss / Brier / margin MAE for completed games in `pred` (needs mu, p_home)."""
    d = pred[pred["completed"] & pred["home_pts"].notna() & (pred["home_pts"] != pred["away_pts"])]
    y = (d["home_pts"] > d["away_pts"]).to_numpy(float)
    ph = np.clip(d["p_home"].to_numpy() if sigma is None else win_prob(d["mu"], sigma), 1e-6, 1 - 1e-6)
    return {"n": int(len(d)), "acc": float(((ph > 0.5) == (y > 0.5)).mean()),
            "logloss": float(-(y * np.log(ph) + (1 - y) * np.log(1 - ph)).mean()),
            "brier": float(((ph - y) ** 2).mean()),
            "mae_margin": float(np.abs((d["home_pts"] - d["away_pts"]) - d["mu"]).mean())}


def calibration_table(pred: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    d = pred[pred["completed"] & pred["home_pts"].notna()].copy()
    # fold to the favourite's side so both tails are used
    pf = np.where(d["p_home"] >= 0.5, d["p_home"], 1 - d["p_home"])
    won = np.where(d["p_home"] >= 0.5, d["home_pts"] > d["away_pts"], d["away_pts"] > d["home_pts"])
    cut = pd.cut(pf, np.linspace(0.5, 1.0, bins + 1), include_lowest=True)
    t = pd.DataFrame({"bin": cut, "p": pf, "won": won}).groupby("bin", observed=True).agg(
        n=("won", "size"), predicted=("p", "mean"), actual=("won", "mean")).reset_index()
    return t
