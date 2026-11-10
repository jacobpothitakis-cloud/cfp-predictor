"""One team's road to the playoff, read straight from the season simulation.

Everything here is a conditional count over the simulated seasons (no extra simulation): how often the team
makes the field, hosts a first-round game or earns a bye given its own results, who it would play, and which
other games move its odds.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

FAN_TEAMS = ["Iowa"]
MIN_N = 150            # fewer simulated seasons than this is too thin to quote a percentage


def _date(season: int, day: float) -> pd.Timestamp:
    return (pd.Timestamp(f"{season}-08-01") + pd.Timedelta(days=float(day) - 0.34)).normalize()


def _stats(seed_t: np.ndarray, rank_t: np.ndarray | None, mask: np.ndarray, extra: dict) -> dict:
    n = int(mask.sum())
    out = {"n": n}
    if n == 0:
        return out
    s = seed_t[mask]
    out |= {"p_field": float((s > 0).mean()), "p_bye": float(((s >= 1) & (s <= 4)).mean()),
            "p_host": float(((s >= 5) & (s <= 8)).mean()), "p_road": float((s >= 9).mean()),
            "seeds": {k: float((s == k).mean()) for k in range(1, 13)}}
    if rank_t is not None:
        r = rank_t[mask]
        out |= {"median_rank": float(np.median(r)), "rank_p10": float(np.percentile(r, 10)),
                "rank_p90": float(np.percentile(r, 90)), "p_top12": float((r <= 12).mean())}
    for k, v in extra.items():
        out[k] = float(v[mask].mean())
    return out


def breakdown(sim, team: str) -> dict | None:
    """sim: a full SimResult (needs sim.raw, so call it before the result is trimmed for saving)."""
    st, raw = sim.st, sim.raw
    if team not in list(st.names) or "homewin" not in raw:
        return None
    t = list(st.names).index(team)
    fut, n = st.future, st.n
    seed, hw, counted = raw["seed"], raw["homewin"], raw["counted"]
    N = seed.shape[0]
    cand = list(raw["cand"])
    rank_t = raw["rank"][:, cand.index(t)] if t in cand else None
    seed_t = seed[:, t]
    rounds_t = raw["rounds"][:, t]
    extra = {"p_title_game": raw["ccg"][:, t], "p_conf_champ": raw["champ"][:, t],
             "p_quarterfinal": rounds_t >= 2, "p_semifinal": rounds_t >= 3, "p_champion": rounds_t >= 5}
    name = lambda i: st.names[i] if i < n else "FCS opponent"   # noqa: E731

    # ---- the team's own remaining games
    own = [g for g in np.where(counted)[0] if t in (fut.gh[g], fut.ga[g])]
    own.sort(key=lambda g: fut.day[g])
    won = {g: (hw[:, g] if fut.gh[g] == t else ~hw[:, g]) for g in own}
    games = []
    for g in own:
        home = fut.gh[g] == t
        opp = int(fut.ga[g] if home else fut.gh[g])
        w = won[g]
        games.append({"g": int(g), "date": _date(st.season, fut.day[g]), "home": bool(home), "neutral": bool(fut.neutral[g]),
                      "opp": name(opp), "opp_idx": opp, "p_win": float(w.mean()),
                      "if_win": _stats(seed_t, rank_t, w, extra), "if_lose": _stats(seed_t, rank_t, ~w, extra)})

    # ---- every combination of results
    scenarios = []
    for combo in product([True, False], repeat=len(own)):
        m = np.ones(N, bool)
        for g, res in zip(own, combo):
            m &= won[g] if res else ~won[g]
        lost = [games[i]["opp"] for i, res in enumerate(combo) if not res]
        label = "Win out" if not lost else ("Lose all remaining games" if len(lost) == len(own) and len(own) > 1
                                            else "Lose only to " + lost[0] if len(lost) == 1 else "Lose to " + " and ".join(lost))
        scenarios.append({"label": label, "n_losses": len(lost), "share": float(m.mean()), **_stats(seed_t, rank_t, m, extra)})
    scenarios.sort(key=lambda s: (s["n_losses"], -s["share"]))
    winout = np.ones(N, bool)
    for g in own:
        winout &= won[g]

    # ---- first-round opponents: 5 hosts 12, 6 hosts 11, 7 hosts 10, 8 hosts 9
    def opponents(mask: np.ndarray) -> list[dict]:
        rows = np.where(mask)[0]
        if len(rows) == 0:
            return []
        want = 17 - seed_t[rows]
        opp = (seed[rows] == want[:, None]).argmax(1)
        cnt = np.bincount(opp, minlength=n)
        return [{"team": st.names[i], "share": float(cnt[i] / len(rows))} for i in np.argsort(-cnt)[:6] if cnt[i] > 0]

    hosting = (seed_t >= 5) & (seed_t <= 8)
    road = seed_t >= 9

    # ---- other teams' games: how much each one moves this team's odds
    def rooting(base: np.ndarray, target: np.ndarray) -> list[dict]:
        out = []
        nb = int(base.sum())
        for g in np.where(counted)[0]:
            if g in own:
                continue
            h = hw[base, g]
            nh = int(h.sum())
            if nh < MIN_N or nb - nh < MIN_N:
                continue
            y = target[base]
            ph, pa = float(y[h].mean()), float(y[~h].mean())
            p = float(y.mean())
            se = float(np.sqrt(max(p * (1 - p), 1e-9) * (1 / nh + 1 / (nb - nh))))
            d = ph - pa
            hi, ai = int(fut.gh[g]), int(fut.ga[g])
            out.append({"game": f"{name(ai)} at {name(hi)}", "date": _date(st.season, fut.day[g]), "home": name(hi), "away": name(ai),
                        "p_home": float(hw[:, g].mean()), "if_home": ph, "if_away": pa, "swing": d, "z": d / se,
                        "root_for": name(hi) if d > 0 else name(ai), "p_root": float(hw[:, g].mean()) if d > 0 else float(1 - hw[:, g].mean())})
        out.sort(key=lambda r: -abs(r["z"]))
        return out

    in_field = seed_t > 0
    return {
        "team": team, "idx": int(t), "n_sims": int(N),
        "now": _stats(seed_t, rank_t, np.ones(N, bool), extra),
        "p_win_out": float(winout.mean()),
        "win_out": _stats(seed_t, rank_t, winout, extra),
        "games": games, "scenarios": scenarios,
        "host_opponents": opponents(hosting), "road_opponents": opponents(road),
        "host_opponents_win_out": opponents(hosting & winout),
        "rooting": rooting(np.ones(N, bool), in_field),
        "rooting_win_out": rooting(winout, in_field) if winout.sum() >= 4 * MIN_N else [],
        "rooting_host_win_out": rooting(winout, hosting) if winout.sum() >= 4 * MIN_N else [],
    }
