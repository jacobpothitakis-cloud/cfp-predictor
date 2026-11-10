"""Calibration of simulated playoff odds against history.

For each season 2014-2025: stand at the first committee ranking, train the committee models WITHOUT that
season, simulate the rest of the season, and compare each team's simulated chance of making the field
with whether it did. Conference races use the generic tiebreaker ladder with that season's divisions
(the leagues' rules have changed since), and title games are treated as neutral-site.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfp import data
from cfp.committee import backtest as bt
from cfp.committee import dataset
from cfp.config import REPORTS
from cfp.simulate.season import simulate


def run(n_sims: int = 1000, releases: tuple[int, ...] = (1, 3)) -> str:
    rels = list(dataset.all_releases())
    g = data.games()
    rows = []
    for season in range(2014, 2026):
        sr = sorted([r for r in rels if r.season == season], key=lambda r: r.release_no)
        final = sr[-1]
        act_order = np.argsort(np.where(np.isnan(final.rank), 999, final.rank), kind="stable")
        actual = bt.field_of(act_order, final).teams
        train = [r for r in rels if r.season != season]
        models = bt.fit_models(train, season)
        bundle = {"anchor": models["_anchor"], "sigma": {"anchor": 1.73}}
        cc = g[(g.season == season) & g.is_conf_champ & (g.home_conf == g.away_conf)]
        tg = {c: {"date": str(d.date()), "site": "neutral"} for c, d in zip(cc.home_conf, cc.start_date - pd.Timedelta(hours=8))}
        for rn in releases:
            if rn > len(sr) - 1:
                continue
            a = sr[rn - 1]
            res = simulate(season, now=a.cutoff, n_sims=n_sims, seed=season * 10 + rn, bundle=bundle, st=a.st,
                           anchor_rel=a, title_confs=tg, playoff=False, legacy_rules=True, final_no=len(sr))
            for r in res.teams.itertuples():
                rows.append({"season": season, "release": rn, "team": r.team, "p_field": r.p_field,
                             "made_field": int(r.idx in actual)})
            print(season, rn, "done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(REPORTS / "simulation_calibration.csv", index=False)
    bins = [0, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98, 1.0001]
    d["bin"] = pd.cut(d["p_field"], bins, right=False)
    t = d.groupby("bin", observed=True).agg(teams=("made_field", "size"), mean_predicted=("p_field", "mean"),
                                           actual_rate=("made_field", "mean")).reset_index()
    brier = float(((d.p_field - d.made_field) ** 2).mean())
    base = float(((d.made_field.mean() - d.made_field) ** 2).mean())
    L = ["# Calibration of simulated playoff odds\n",
         f"Seasons 2014–2025, standing at ranking release(s) {', '.join(map(str, releases))} of each season, {n_sims:,} simulations each, "
         "committee models trained without the season being simulated. A team counts as having made the field "
         "if the committee's actual final ranking put it in under that season's rules (top 4 through 2023; "
         "12 teams in 2024–25).\n",
         "| Simulated chance | Team-seasons | Mean simulated | Actually made it |", "|---|---|---|---|"]
    for r in t.itertuples():
        L.append(f"| {r.bin} | {r.teams} | {r.mean_predicted:.3f} | {r.actual_rate:.3f} |")
    L.append(f"\nBrier score {brier:.4f} (always guessing the base rate: {base:.4f}; lower is better). "
             f"Brier skill {1 - brier / base:.2f}.\n")
    for rn in releases:
        x = d[d.release == rn]
        L.append(f"- From release {rn}: Brier {((x.p_field - x.made_field) ** 2).mean():.4f}; teams given ≥50% that made it "
                 f"{int(x[x.p_field >= .5].made_field.sum())}/{int((x.p_field >= .5).sum())}; teams given <10% that made it "
                 f"{int(x[x.p_field < .1].made_field.sum())}/{int((x.p_field < .1).sum())}.")
    L.append("\nLimits: only 12 seasons, 10 of them with a four-team field, so the high-probability bins hold few "
             "teams; historical simulations use this model's own rating only (no FPI) and approximate each "
             "league's historical tiebreakers and title-game sites.\n")
    out = "\n".join(L)
    (REPORTS / "simulation_calibration.md").write_text(out)
    return out


if __name__ == "__main__":
    print(run())
