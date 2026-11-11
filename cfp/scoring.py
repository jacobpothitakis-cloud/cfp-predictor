"""Score stored pre-release predictions against the committee's actual ranking, next to two yardsticks:
the AP poll that was current when the prediction was made, and the committee's own previous ranking left unchanged.

A team a source did not list is counted as rank 31 for every source alike (the stored prediction holds 30 teams).
"""
from __future__ import annotations

import pandas as pd

from cfp import data
from cfp.config import REPORTS

PRED_DIR = REPORTS / "predictions"
UNLISTED = 31


def _week_result(g: pd.DataFrame, tid: int, lo: pd.Timestamp | None, hi: pd.Timestamp) -> str:
    """What the team did between the previous ranking and this one, from the stored games."""
    m = g[((g.home_id == tid) | (g.away_id == tid)) & g.completed & (g.start_date <= hi)]
    if lo is not None:
        m = m[m.start_date > lo]
    else:
        m = m[m.start_date > hi - pd.Timedelta(days=7)]
    out = []
    for r in m.sort_values("start_date").itertuples():
        home = r.home_id == tid
        us, them = (r.home_pts, r.away_pts) if home else (r.away_pts, r.home_pts)
        opp = r.away_team if home else r.home_team
        out.append(f"{'beat' if us > them else 'lost to'} {opp} {int(us)}-{int(them)}" + ("" if home or r.neutral else " (away)"))
    return "; ".join(out) if out else "did not play"


def scored_releases(season: int) -> list[dict]:
    """One entry per committee ranking this season that has a prediction stored before its release."""
    rel = data.releases()
    rel = rel[rel["season"] == season].sort_values("release_no")
    names = data.team_names()
    rk = data.rankings()
    g = data.games()
    g = g[g.season == season]
    out, prev_cut = [], None
    for r in rel.itertuples():
        f = PRED_DIR / f"{season}_release_{r.release_no}.csv"
        act = data.cfp_ranking(season, r.cutoff)
        entry = {"release": int(r.release_no), "date": str(r.cutoff)[:10], "status": "scored", "cutoff": r.cutoff}
        this_prev, prev_cut = prev_cut, r.cutoff
        if not f.exists():
            out.append(entry | {"status": "no prediction"})
            continue
        p = pd.read_csv(f)
        made = pd.Timestamp(p["games_through_utc"].iloc[0], tz="UTC")
        if made > r.cutoff + pd.Timedelta(hours=20):
            out.append(entry | {"status": "made after release"})
            continue
        ap = rk[(rk.poll == "ap") & (rk.season == season) & (rk.poll_date <= made)]
        ap = ap[ap.poll_date == ap.poll_date.max()] if len(ap) else ap
        ap_rank = {names[int(t)]: int(k) for t, k in zip(ap.team_id, ap["rank"])}
        prev = data.cfp_ranking(season, this_prev) if this_prev is not None else act.iloc[0:0]
        prev_rank = {names[int(t)]: int(k) for t, k in zip(prev.team_id, prev["rank"])}
        pr = dict(zip(p["team"], p["rank"]))
        d = pd.DataFrame({"team": [names[int(t)] for t in act.team_id], "team_id": act.team_id.astype(int).to_numpy(),
                          "actual": act["rank"].astype(int).to_numpy(), "record": act["record"].to_numpy()})
        d["model"] = d.team.map(pr).fillna(UNLISTED).astype(int)
        d["ap"] = d.team.map(ap_rank).fillna(UNLISTED).astype(int)
        d["prev"] = d.team.map(prev_rank).fillna(UNLISTED).astype(int)
        for c in ("model", "ap", "prev"):
            d[f"err_{c}"] = d[c] - d["actual"]
        d["week"] = [_week_result(g, t, this_prev, r.cutoff) for t in d.team_id]
        d["low"] = d.team.map(dict(zip(p["team"], p["low"])))
        d["high"] = d.team.map(dict(zip(p["team"], p["high"])))
        d["in_range"] = (d.actual >= d.low) & (d.actual <= d.high)
        d = d.sort_values("actual").reset_index(drop=True)

        def summary(col: str, ranks: dict) -> dict:
            e = d[f"err_{col}"].abs()
            top = lambda k: len({t for t, v in ranks.items() if v <= k} & set(d[d.actual <= k].team))   # noqa: E731
            return {"mae": float(e.mean()), "within1": float((e <= 1).mean()), "within2": float((e <= 2).mean()),
                    "exact": int((e == 0).sum()), "top4": top(4), "top12": top(12),
                    "top25": len({t for t, v in ranks.items() if v <= 25} & set(d.team))}

        entry |= {"made": str(made)[:16], "ap_date": str(ap.poll_date.max())[:10] if len(ap) else "", "table": d,
                  "model": summary("model", pr), "ap": summary("ap", ap_rank),
                  "prev": summary("prev", prev_rank) if prev_rank else None,
                  "closer_model": int((d.err_model.abs() < d.err_ap.abs()).sum()),
                  "closer_ap": int((d.err_model.abs() > d.err_ap.abs()).sum()),
                  "in_range": float(d.in_range.mean())}
        out.append(entry)
    return out
