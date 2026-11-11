"""Inputs that need the CollegeFootballData tables: betting-market strength, Elo and play-by-play efficiency.

All are strictly as-of the cutoff: only games that kicked off before it are used. Each column is z-scored
across FBS teams. If the CFBD tables are missing every column is zero, so the model still runs on ESPN data alone.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from cfp import data

EXTERNAL = ["mkt", "elo", "net_ppa", "net_sr", "ats", "ats_week", "rv", "w_cfp", "w_cfp12", "cfp_win_pts", "l_cfp_out"]


@lru_cache(maxsize=1)
def _tables() -> dict[str, pd.DataFrame] | None:
    if not (data.has_table("cfbd_games") and data.has_table("cfbd_adv_game")):
        return None
    cg = data.table("cfbd_games")
    cg["startDate"] = pd.to_datetime(cg["startDate"], utc=True)
    adv = data.table("cfbd_adv_game")[["gameId", "season", "team", "offense.ppa", "defense.ppa",
                                       "offense.successRate", "defense.successRate"]]
    side = pd.concat([cg[["id", "homeTeam", "homeId"]].set_axis(["gameId", "team", "team_id"], axis=1),
                      cg[["id", "awayTeam", "awayId"]].set_axis(["gameId", "team", "team_id"], axis=1)])
    adv = adv.merge(side, on=["gameId", "team"], how="left").merge(
        cg[["id", "startDate"]].rename(columns={"id": "gameId"}), on="gameId", how="left")
    return {"cg": cg, "adv": adv}


def clear() -> None:
    _tables.cache_clear(); _others.cache_clear()


def _z(x: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(x)
    if ok.sum() < 10:
        return np.zeros_like(x)
    out = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    return np.where(ok, out, 0.0)


def _market_rating(g: pd.DataFrame, idx: dict[int, int], n: int, ridge: float = 1.0) -> np.ndarray:
    """Team strength implied by closing spreads of the games played so far (ridge least squares; one pooled
    slot for all non-FBS opponents; a common home-field term)."""
    g = g[g["spread_home"].notna()]
    if len(g) < n:
        return np.full(n, np.nan)
    h = g["home_id"].map(lambda t: idx.get(int(t), n)).to_numpy()
    a = g["away_id"].map(lambda t: idx.get(int(t), n)).to_numpy()
    A = np.zeros((len(g), n + 2))
    r = np.arange(len(g))
    A[r, h] += 1; A[r, a] -= 1
    A[:, n + 1] = (~g["neutral"].to_numpy(bool)).astype(float)
    y = -g["spread_home"].to_numpy(float)
    P = np.eye(n + 2) * ridge; P[n + 1, n + 1] = 1e-6
    b = np.linalg.solve(A.T @ A + P, A.T @ y)
    out = b[:n].copy()
    seen = np.bincount(np.concatenate([h, a]), minlength=n + 1)[:n]
    out[seen == 0] = np.nan
    return out


@lru_cache(maxsize=1)
def _others() -> pd.DataFrame | None:
    if not data.has_table("poll_others"):
        return None
    o = data.table("poll_others")
    o["poll_date"] = pd.to_datetime(o["poll_date"], utc=True)
    return o


def receiving_votes(st) -> np.ndarray:
    """0..1 for teams just outside the AP / Coaches top 25: their poll points as a share of the 25th team's,
    averaged over the two polls (latest poll before the cutoff). Ranked teams and teams with no votes get 0."""
    n = st.n
    out = np.zeros(n)
    o = _others()
    if o is None:
        return out
    idx = {int(t): i for i, t in enumerate(st.team_ids)}
    o = o[(o.season == st.season) & (o.poll_date < st.cutoff)]
    for poll in ("ap", "coaches"):
        p = o[o.poll == poll]
        if p.empty:
            continue
        p = p[p.poll_date == p.poll_date.max()]
        for tid, pts, p25 in zip(p.team_id, p.points, p.points_25th):
            i = idx.get(int(tid))
            if i is not None and p25 > 0:
                out[i] += 0.5 * min(pts / p25, 1.0)
    return out


def committee_wins(st) -> dict[str, np.ndarray]:
    """Results measured against the committee's own previous ranking, the way the chair describes them
    ("three wins over teams in our top 13"). All zero at a season's first ranking."""
    n = st.n
    out = {k: np.zeros(n) for k in ("w_cfp", "w_cfp12", "cfp_win_pts", "l_cfp_out")}
    if not st.has_prev_cfp:
        return out
    rk = np.append(np.where(np.isnan(st.prev_cfp), 99.0, st.prev_cfp), 99.0)
    for h, a, m in zip(st.played.gh, st.played.ga, st.margin):
        if m == 0 or np.isnan(m):
            continue
        w, l = (h, a) if m > 0 else (a, h)
        if w < n:
            if rk[l] <= 25:
                out["w_cfp"][w] += 1; out["cfp_win_pts"][w] += (26 - rk[l]) / 25
            if rk[l] <= 12:
                out["w_cfp12"][w] += 1
        if l < n and rk[w] > 25:
            out["l_cfp_out"][l] += 1
    return out


def features(st) -> pd.DataFrame:
    """One row per team in st.names."""
    n, names = st.n, list(st.names)
    X = pd.DataFrame(0.0, index=names, columns=EXTERNAL)
    X["rv"] = receiving_votes(st)
    for k, v in committee_wins(st).items():
        X[k] = v
    T = _tables()
    if T is None:
        return X
    idx = {int(t): i for i, t in enumerate(st.team_ids)}
    cut = st.cutoff
    g = data.games()
    g = g[(g.season == st.season) & g.completed & (g.start_date <= cut)]
    X["mkt"] = _z(_market_rating(g, idx, n))

    # performance against the closing spread, capped at three touchdowns a game
    cover = np.clip((g.home_pts - g.away_pts + g.spread_home).to_numpy(float), -21, 21)
    s = np.zeros(n); c = np.zeros(n); last = np.full(n, np.nan); last_t = np.zeros(n)
    week_lo = cut - pd.Timedelta(days=8)
    for hid, aid, cv, when in zip(g.home_id, g.away_id, cover, g.start_date):
        if np.isnan(cv):
            continue
        for tid, v in ((hid, cv), (aid, -cv)):
            i = idx.get(int(tid))
            if i is None:
                continue
            s[i] += v; c[i] += 1
            if when > week_lo and when.value >= last_t[i]:
                last[i] = v; last_t[i] = when.value
    X["ats"] = _z(np.where(c > 0, s / np.maximum(c, 1), np.nan))
    X["ats_week"] = np.nan_to_num(last) / 10.0

    cg = T["cg"]
    cg = cg[(cg.season == st.season) & (cg.startDate <= cut) & cg.homePoints.notna()].sort_values("startDate")
    elo = np.full(n, np.nan)
    for hid, aid, he, ae in zip(cg.homeId, cg.awayId, cg.homePostgameElo, cg.awayPostgameElo):
        for tid, e in ((hid, he), (aid, ae)):
            i = idx.get(int(tid)) if pd.notna(tid) else None
            if i is not None and pd.notna(e):
                elo[i] = e
    X["elo"] = _z(elo)

    adv = T["adv"]
    adv = adv[(adv.season == st.season) & (adv.startDate <= cut) & adv.team_id.notna()]
    if len(adv):
        m = adv.assign(net_ppa=adv["offense.ppa"] - adv["defense.ppa"],
                       net_sr=adv["offense.successRate"] - adv["defense.successRate"]).groupby("team_id")[["net_ppa", "net_sr"]].mean()
        for col in ("net_ppa", "net_sr"):
            v = np.full(n, np.nan)
            for tid, val in m[col].items():
                i = idx.get(int(tid))
                if i is not None:
                    v[i] = val
            X[col] = _z(v)

    return X
