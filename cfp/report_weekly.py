"""Weekly Markdown report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cfp import data
from cfp.config import REPORTS
from cfp.predict import bracket_text


def pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return "<1%" if 0 < x < 0.005 else f"{x:.0%}"


def write_weekly(pred, sim, log: list[str] | None = None) -> Path:
    st = pred.rel.st
    names = data.team_names()
    rel = data.releases(); rel = rel[rel.season == st.season].sort_values("release_no")
    last = data.cfp_ranking(st.season, rel.cutoff.iloc[-1]) if len(rel) else pd.DataFrame()
    last_rank = {names[int(t)]: int(k) for t, k in zip(last.get("team_id", []), last.get("rank", []))}
    L = [f"# CFP committee predictor — week of {pred.release_date}\n",
         f"Prediction of ranking #{pred.release_no} ({'Selection Day' if pred.is_final else 'released ' + pred.release_date}). "
         + " ".join(pred.notes) + "\n"]
    for x in log or []:
        L.append(f"- {x}")
    L += ["", f"## Predicted committee Top 25\n",
          "| Pred | Team | Record | Last committee | AP | Chance of exactly this spot | Likely range (80%) | Top 4 | Top 12 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in pred.table.head(25).itertuples():
        prev = last_rank.get(r.team, "NR")
        mv = "" if prev == "NR" else (f" (▲{prev - r.rank})" if prev > r.rank else f" (▼{r.rank - prev})" if prev < r.rank else " (–)")
        L.append(f"| {r.rank} | {r.team} | {r.record} | {prev}{mv} | {'NR' if pd.isna(r.ap) else int(r.ap)} | {pct(r.p_exact)} | "
                 f"{r.low}–{r.high} | {pct(r.p_top4)} | {pct(r.p_top12)} |")
    nxt = pred.table.iloc[25:30]
    L.append("\nNext five: " + ", ".join(f"{r.team} ({pct(r.p_top25)} to be ranked)" for r in nxt.itertuples()) + ".\n")
    L.append("## Bracket if the season ended today\n")
    L.append("Uses the predicted ranking above with each league's current leader standing in as its champion.\n")
    L.append("```\n" + bracket_text(pred.field, st) + "\n```\n")
    L.append(f"## Simulated Selection Day ({sim.n_sims:,} seasons, seed {sim.seed})\n")
    L += [f"- {n}" for n in sim.notes]
    L += ["", "| Team | Make field | Bye | Host 1st round | Auto bid | Win league | Mean final rank (10–90%) | Most likely seed | Title | Wins out → in | One more loss → in |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sim.teams[sim.teams.p_field >= 0.01].itertuples():
        sd = sim.seeds.loc[r.team]
        ms = f"{int(sd.idxmax())} ({pct(sd.max())})" if sd.max() > 0 else ""
        L.append(f"| {r.team} | {pct(r.p_field)} | {pct(r.p_bye)} | {pct(r.p_host)} | {pct(r.p_auto)} | {pct(r.p_conf_champ)} | "
                 f"{r.mean_rank:.1f} ({r.rank_p10:.0f}–{r.rank_p90:.0f}) | {ms} | {r.p_champion:.1%} | "
                 f"{pct(r.field_if_win_out)} ({pct(r.p_win_out)} to win out) | {pct(r.field_if_one_more_loss)} |")
    L.append("\n'n/a' means fewer than 30 simulations met the condition.\n")
    L.append("### Seed probabilities\n")
    top = sim.teams[sim.teams.p_field >= 0.05]["team"]
    L += ["| Team | " + " | ".join(str(k) for k in range(1, 13)) + " |", "|---|" + "---|" * 12]
    for t in top:
        L.append(f"| {t} | " + " | ".join(pct(v) if v >= 0.005 else "" for v in sim.seeds.loc[t]) + " |")
    L.append("\n### Most likely conference title games\n")
    L += ["| League | Matchup (top seed first) | Chance | Site |", "|---|---|---|---|"]
    L += [f"| {r.conf} | {r.matchup} | {pct(r.p)} | {r.site} |" for r in sim.title_games.itertuples()]
    L.append("\n## Games that move the playoff picture most\n")
    L.append("Leverage = total change in all teams' chances of making the field between the two outcomes; "
             "expected impact scales that by how uncertain the game is.\n")
    L += ["| Game | Date | Home win prob | Leverage | Expected impact | Most affected |", "|---|---|---|---|---|---|"]
    if len(sim.games):
        wk = sim.games[sim.games.week == sim.games.week.min()].head(10)
        L += [f"| {r.game} | {r.date} | {pct(r.p_home_win)} | {r.leverage:.2f} | {r.expected_impact:.2f} | {r.most_affected} |" for r in wk.itertuples()]
        L.append("\nRest of the season:\n")
        L += ["| Game | Date | Home win prob | Leverage | Expected impact | Most affected |", "|---|---|---|---|---|---|"]
        rest = sim.games[sim.games.week != sim.games.week.min()].head(12)
        L += [f"| {r.game} | {r.date} | {pct(r.p_home_win)} | {r.leverage:.2f} | {r.expected_impact:.2f} | {r.most_affected} |" for r in rest.itertuples()]
    L.append("\n## Why each team is where it is\n")
    for nm, e in pred.explanations.items():
        up = "; ".join(f"{g} ({v:+.1f}{', ' + f if f else ''})" for g, v, f in e["up"]) or "nothing stands out"
        dn = "; ".join(f"{g} ({v:+.1f}{', ' + f if f else ''})" for g, v, f in e["down"]) or "nothing stands out"
        L.append(f"**{e['rank']}. {nm}** — Helping: {up}. Hurting: {dn}.\n")
    L.append("Numbers in brackets are log-odds relative to the average of the predicted top 25, from the "
             "interpretable (pairwise) half of the model.\n")
    L.append("### Neighbour by neighbour\n")
    L += [f"- {p}" for p in pred.pairs]
    L.append("\n## Model versus committee this season\n")
    f = REPORTS / f"{st.season}_prediction_scores.csv"
    if f.exists():
        s = pd.read_csv(f)
        L += ["| Release | Date | MAE | Within 1 | Within 2 | Top 4 | Top 12 |", "|---|---|---|---|---|---|---|"]
        L += [f"| {r.release} | {r.date} | {r.mae:.2f} | {r.within1:.0%} | {r.within2:.0%} | {r.top4}/4 | {r.top12}/12 |" for r in s.itertuples()]
    else:
        L.append("No pre-release prediction has been scored yet this season. A retrospective out-of-sample "
                 "prediction of the season's first ranking is in `reports/backtest_report.md`, section 8.\n")
    L.append("\n## Proxies that decided a tiebreaker in the simulation\n")
    if sim.proxy_log:
        L += ["| League: step → proxy | Times used per 1,000 simulations |", "|---|---|"]
        L += [f"| {k} | {v / sim.n_sims * 1000:.0f} |" for k, v in sorted(sim.proxy_log.items(), key=lambda kv: -kv[1])]
    L.append("\n## What cannot be known\n")
    L.append("- Thirteen people vote; the ranges above are calibrated on how far the committee has strayed from this "
             "model in the past, not on anything observable this week.\n"
             "- The committee's schedule-strength, record-strength and conference-strength numbers are private; "
             "the versions here are reconstructions.\n"
             "- Injuries and availability enter only through `data/availability.yaml`, with an assumed effect.\n"
             "- Which bowl each of the top three seeds picks is an assumption based on geography.\n")
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"weekly_{pred.release_date}.md"
    path.write_text("\n".join(L))
    return path
