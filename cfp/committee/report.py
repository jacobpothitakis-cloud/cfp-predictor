"""Backtest report -> reports/backtest_report.md (+ parquet of every prediction)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfp import data
from cfp.committee import backtest as bt
from cfp.committee import dataset, uncertainty
from cfp.committee.backtest import PRIMARY, SIM_MODEL
from cfp.committee.dataset import ERA_FEATURES
from cfp.committee.model import ERA_SCALE
from cfp.committee.train import train_production
from cfp.config import DATA, REPORTS

COLS = ["mae", "mae_top12", "spearman", "within1", "within2", "top4_exact", "top12_exact", "top12_overlap"]
HEAD = {"mae": "MAE (all 25)", "mae_top12": "MAE (top 12)", "spearman": "Spearman", "within1": "Within 1",
        "within2": "Within 2", "top4_exact": "Top-4 exact", "top12_exact": "Top-12 exact",
        "top12_overlap": "Top-12 overlap", "field_correct": "Field correct", "seeds_correct": "Seeds correct",
        "field_overlap": "Field overlap"}


def md(df: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    out = ["| " + " | ".join(str(c) for c in df.columns) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return "\n".join(out)


def selection_sunday(rels) -> tuple[pd.DataFrame, list[str]]:
    """How title-game winners, losers and idle teams moved between the last two rankings of each season."""
    rows, notes = [], []
    by: dict[int, list] = {}
    for r in rels:
        by.setdefault(r.season, []).append(r)
    for season, lst in sorted(by.items()):
        lst = sorted(lst, key=lambda r: r.release_no)
        if not lst[-1].is_final or len(lst) < 2:
            continue
        pen, fin = lst[-2], lst[-1]
        for t in np.where(~np.isnan(pen.rank))[0]:
            a, b = pen.rank[t], fin.rank[t]
            kind = "won title game" if fin.raw["champion"].iloc[t] > 0 else \
                "lost title game" if fin.raw["ccg_loss"].iloc[t] > 0 else "idle"
            rows.append({"season": season, "team": pen.st.names[t], "kind": kind, "before": a,
                         "after": b if not np.isnan(b) else 26.0})
        # did a title-game loss drop a team out of the field?
        o_pen = np.argsort(np.where(np.isnan(pen.rank), 99, pen.rank))
        o_fin = np.argsort(np.where(np.isnan(fin.rank), 99, fin.rank))
        cut = 4 if season <= 2023 else 12
        in_before, in_after = set(o_pen[:cut]), set(o_fin[:cut])
        for t in in_before - in_after:
            if fin.raw["ccg_loss"].iloc[t] > 0:
                notes.append(f"{season} {pen.st.names[t]}: No. {int(pen.rank[t])} before its title-game loss, "
                             f"No. {int(fin.rank[t]) if not np.isnan(fin.rank[t]) else 'NR'} after (out of the top {cut}).")
    d = pd.DataFrame(rows)
    d["move"] = d["before"] - d["after"]           # positive = moved up
    return d, notes


def run(C: float = 0.3) -> str:
    res = bt.loso(C=C, verbose=False)
    m, preds, pairs = res["metrics"], res["preds"], res["pairs"]
    REPORTS.mkdir(exist_ok=True)
    preds.to_parquet(REPORTS / "backtest_predictions.parquet")
    m.to_csv(REPORTS / "backtest_metrics.csv", index=False)
    rels = list(dataset.all_releases())
    L: list[str] = ["# Committee model backtest\n"]
    n_rel = m[m.model == PRIMARY].shape[0]
    L.append(f"Leave-one-season-out over 2014–2025: for each season the models are trained on every *other* "
             f"season's rankings and then predict each of that season's {n_rel} weekly committee rankings in total, "
             "using only games, polls and ratings that existed before each release. "
             "Ratings are the walk-forward team-strength model, so there is no look-ahead in the features.\n")
    L.append("Rank error is measured on the 25 teams the committee actually ranked; a team the model leaves "
             "outside its top 25 is scored at the rank the model gave it (26, 31, ...).\n")

    # 1. overall
    L.append("## 1. Overall, all weekly rankings\n")
    t = m.groupby("model")[COLS].mean().sort_values("mae").reset_index().rename(columns={"model": "Model", **HEAD})
    L.append(md(t))
    L.append("\n*Résumé only* uses no polls and no previous committee ranking. *+polls, last week* adds the AP and "
             "Coaches polls released two days before and last week's committee ranking (the AP poll stands in for "
             "the first release of a season). *Carry-over* models use an earlier committee ranking from the same "
             "season but no polls: these are what the season simulation uses, because future polls do not exist.\n")

    # 2. by season
    L.append("## 2. By season: mean absolute rank error\n")
    show = [PRIMARY, "AP poll", "Prev rank + results", "Strength of record", "Ensemble (résumé only)", SIM_MODEL]
    t = m[m.model.isin(show)].pivot_table(index="season", columns="model", values="mae")[show].reset_index()
    t.loc[len(t)] = ["All"] + [m[m.model == k]["mae"].mean() for k in show]
    L.append(md(t))
    worse = [int(s) for s in t["season"][:-1] if t.loc[t.season == s, PRIMARY].item() > t.loc[t.season == s, "AP poll"].item()]
    if worse:
        L.append(f"\n**The model loses to the AP poll in {', '.join(map(str, worse))}.** See section 9.\n")

    L.append("\nBy release number within the season (1 = first ranking, when there is no previous committee ranking):\n")
    t = m[m.model.isin([PRIMARY, "AP poll", "Prev rank + results"])].pivot_table(index="release", columns="model", values="mae").reset_index()
    L.append(md(t))

    # 3. final rankings
    L.append("\n## 3. Selection Day rankings only (12 seasons)\n")
    f = m[m.is_final]
    fc = ["mae", "within1", "top4_exact", "top12_exact", "field_correct", "field_overlap", "seeds_correct"]
    t = f.groupby("model")[fc].mean().sort_values("mae").reset_index().rename(columns={"model": "Model", **HEAD})
    L.append(md(t))
    L.append("\n*Field correct*: the playoff field implied by the predicted ranking equals the field implied by the "
             "committee's ranking under that season's rules (top 4 through 2023; five champions plus seven at-large "
             "in 2024–25). *Seeds correct*: share of seed lines filled by the right team.\n")
    t = f[f.model.isin([PRIMARY, SIM_MODEL, "AP poll"])].pivot_table(index="season", columns="model",
                                                                    values=["field_correct", "seeds_correct"])
    t.columns = [f"{HEAD[a]} — {b}" for a, b in t.columns]
    L.append(md(t.reset_index()))

    # 4. stickiness
    prod = train_production(save=False)
    w = prod["pair_full"].weights()
    wl = prod["pair_level"].weights()
    wa = prod["pair_anchor"].weights()
    lv, fu = m[m.model == "Ensemble (résumé only)"]["mae"].mean(), m[m.model == PRIMARY]["mae"].mean()
    L.append("\n## 4. Is the ranking a clean sheet each week? No.\n")
    L.append(f"Adding last week's rank and the current polls cuts the error from {lv:.2f} to {fu:.2f} places. "
             f"In the pairwise model the weight on last week's committee position is {w['prev']:.2f} log-odds per "
             f"25 places (i.e. per full length of the ranking), on the AP poll {w['ap']:.2f} and on the Coaches poll "
             f"{w['coaches']:.2f}; a loss this week by a previously ranked team is worth {w['prev_lost']:.2f} on top "
             "of what the loss does to the résumé features. At a season's first release, with no previous committee "
             f"ranking, the AP poll's weight rises by {w['ap_first']:.2f} and the Coaches poll's by {w['coaches_first']:.2f}. "
             "The committee says each ranking starts from scratch; "
             "the data say a team's previous position carries real weight after controlling for the résumé.\n")
    L.append("Carry-over decays with time. Weight on an earlier committee ranking, by how many releases ago it was: "
             f"1 release {wa['anc1']:.2f}, 2 releases {wa['anc2']:.2f}, 3+ releases {wa['anc3']:.2f}; each loss since "
             f"then {wa['new_losses']:.2f} (plus {wa['anc_lost']:.2f} × earlier position).\n")

    # 5. weights
    L.append("## 5. What the résumé-only pairwise model learned\n")
    L.append("Weights are log-odds of being ranked ahead per one unit of the feature (z-scored features: per "
             "standard deviation across FBS that week; counts: per game). Base = 2014–2023; the next two columns "
             "add the learned shift for the 12-team era (2024+) and for 2025+ (new metrics).\n")
    rows = []
    for fname in sorted(set(n.split("@")[0] for n in wl.index), key=lambda k: -abs(wl.get(k, 0))):
        base = wl.get(fname, 0.0)
        d12 = wl.get(f"{fname}@12team", np.nan) * ERA_SCALE
        d25 = wl.get(f"{fname}@2025+", np.nan) * ERA_SCALE
        rows.append({"Feature": fname, "Base": base, "12-team era": base + d12 if not np.isnan(d12) else np.nan,
                     "2025+": base + d12 + d25 if not np.isnan(d25) else np.nan})
    wt = pd.DataFrame(rows)
    L.append(md(wt.head(28).fillna("")))
    L.append("\n### How priorities shifted by era\n")
    era = wt[wt["Feature"].isin(ERA_FEATURES)].copy()
    era["Shift 2014–23 → 2025+"] = era["2025+"] - era["Base"]
    L.append(md(era.sort_values("Shift 2014–23 → 2025+", key=abs, ascending=False)))
    L.append("\nOnly two seasons (12 rankings) exist in the 12-team era and one-plus under the 2025 metrics, so "
             "these shifts are estimated from little data and are regularised toward zero.\n")
    L.append("### Which groups of criteria carry predictive weight (ablation)\n")
    L.append("Individual weights above are entangled because many features measure overlapping things (rating, "
             "margin and postgame win expectancy; win credit and best win). A cleaner test: remove one whole group "
             "from the résumé-only model, refit, and see how much the leave-one-season-out error rises.\n")
    ab = bt.ablation(C)
    L.append(md(ab.sort_values("MAE change", ascending=False), "{:.3f}"))
    L.append("\nA group whose removal costs nothing is either unused by the committee or already implied by the "
             "other groups; it is not evidence the committee ignores it.\n")
    pf = prod["pair_full"]
    L.append("### Head-to-head and common opponents (pod effect)\n")
    L.append(f"Head-to-head winner bonus: {prod['pair_level'].b_h2h_:.2f} log-odds (résumé-only model), "
             f"{pf.b_h2h_:.2f} (model with polls), applied with a closeness kernel of width "
             f"{prod['pair_level'].tau_:.0f} / {pf.tau_:.0f} log-odds, so it fades once two teams are far apart. "
             f"Common-opponent edge: {prod['pair_level'].b_common_:.2f} / {pf.b_common_:.2f} per net game.\n")

    # 6. selection sunday
    ss, notes = selection_sunday(rels)
    L.append("## 6. Selection Day behaviour, 2014–2025\n")
    g = ss.groupby("kind").agg(teams=("move", "size"), mean_move=("move", "mean"), moved_up=("move", lambda x: (x > 0).mean()),
                               moved_down=("move", lambda x: (x < 0).mean()), unchanged=("move", lambda x: (x == 0).mean())).reset_index()
    g.columns = ["Ranked team's last weekend", "Teams", "Mean places moved (+ = up)", "Share up", "Share down", "Share unchanged"]
    L.append(md(g))
    top = ss[ss.before <= 12]
    g = top.groupby("kind").agg(teams=("move", "size"), mean_move=("move", "mean"), moved_down=("move", lambda x: (x < 0).mean())).reset_index()
    g.columns = ["Top-12 teams only", "Teams", "Mean places moved", "Share down"]
    L.append("\n" + md(g))
    idle_passed = ss[(ss.kind == "idle") & (ss.move < 0)]
    L.append(f"\nIdle teams do get passed: {len(idle_passed)} of {int((ss.kind == 'idle').sum())} idle ranked teams "
             "finished lower than they started the weekend.\n")
    L.append("Title-game losses that knocked a team out of the playoff cut line (top 4 through 2023, top 12 after):\n")
    L += [f"- {n}" for n in notes] or ["- none"]

    # 7. calibration of rank distributions
    sig_p, sim_p, obs_p = uncertainty.calibrate_sigma(pairs[PRIMARY])
    sig_a, sim_a, obs_a = uncertainty.calibrate_sigma(pairs[SIM_MODEL])
    cov_p = uncertainty.coverage(pairs[PRIMARY], sig_p)
    cov_a = uncertainty.coverage(pairs[SIM_MODEL], sig_a)
    from cfp.committee.train import save_sigmas
    save_sigmas({"primary": sig_p, "anchor": sig_a})
    prod = train_production(sigmas={"primary": sig_p, "anchor": sig_a}, save=True)
    L.append("\n## 7. Uncertainty calibration\n")
    L.append("Each prediction ships with a rank distribution: Gaussian noise on team scores, re-ranked many times. "
             "The noise scale is set so a typical draw sits as far from the point prediction as the committee's "
             "actual ranking did in the backtest. Coverage is then checked on the same held-out predictions.\n")
    L.append(md(pd.DataFrame([
        {"Model": PRIMARY, "Noise σ (log-odds)": sig_p, "Observed MAE": obs_p, "50% interval covers": cov_p["cover50"],
         "80% interval covers": cov_p["cover80"], "95% interval covers": cov_p["cover95"]},
        {"Model": SIM_MODEL, "Noise σ (log-odds)": sig_a, "Observed MAE": obs_a, "50% interval covers": cov_a["cover50"],
         "80% interval covers": cov_a["cover80"], "95% interval covers": cov_a["cover95"]}])))
    L.append("\nIntervals are on integer ranks, so coverage above the nominal level is expected for teams whose "
             "rank is nearly certain. Calibration of simulated *playoff odds* is reported with the simulation.\n")

    # 8. 2026 forward test
    L.append("## 8. 2026 so far (true out-of-sample)\n")
    train = [r for r in rels if r.season < 2026]
    fm = bt.fit_models(train, 2026, C)
    r26 = [r for r in rels if r.season == 2026]
    log_rows = []
    for rel in sorted(r26, key=lambda r: r.release_no):
        order = fm[PRIMARY].order(rel)
        mt = bt.metrics(order, rel)
        ap = bt.metrics(bt.base_ap(rel), rel)
        L.append(f"**Release {rel.release_no} ({str(rel.cutoff)[:10]}),** predicted with a model trained only on "
                 f"2014–2025: MAE {mt['mae']:.2f} (AP poll {ap['mae']:.2f}), within one spot {mt['within1']:.0%}, "
                 f"within two {mt['within2']:.0%}, top-4 overlap {mt['top4_overlap'] * 4:.0f}/4, top-12 overlap "
                 f"{mt['top12_overlap'] * 12:.0f}/12, top-25 overlap {mt['top25_overlap'] * 25:.0f}/25.\n")
        n = len(order); pr = np.empty(n); pr[order] = np.arange(1, n + 1)
        rows = []
        for t in np.where(~np.isnan(rel.rank) | (pr <= 25))[0]:
            rows.append({"Team": rel.st.names[t], "Actual": rel.rank[t], "Predicted": int(pr[t]),
                         "AP": rel.st.ap[t], "Record": f"{int(rel.raw['wins'].iloc[t])}-{int(rel.raw['losses'].iloc[t])}",
                         "Error": (pr[t] - rel.rank[t]) if not np.isnan(rel.rank[t]) else np.nan})
            log_rows.append({"season": 2026, "release": rel.release_no, "date": str(rel.cutoff)[:10], **{k.lower(): v for k, v in rows[-1].items()}})
        d = pd.DataFrame(rows).sort_values("Actual")
        d["Actual"] = d["Actual"].map(lambda x: "NR" if np.isnan(x) else int(x))
        d["AP"] = d["AP"].map(lambda x: "NR" if np.isnan(x) else int(x))
        d["Error"] = d["Error"].map(lambda x: "" if np.isnan(x) else f"{int(x):+d}")
        L.append(md(d))
    pd.DataFrame(log_rows).to_csv(REPORTS / "2026_prediction_log.csv", index=False)

    # 9. biggest misses
    L.append("\n## 9. Biggest misses in the backtest\n")
    p = preds[(preds.model == PRIMARY) & preds.actual.notna()].copy()
    p["err"] = p["pred"] - p["actual"]
    big = p.reindex(p.err.abs().sort_values(ascending=False).index).head(15)
    big = big[["season", "release", "team", "actual", "pred", "err"]]
    big.columns = ["Season", "Release", "Team", "Actual", "Predicted", "Error"]
    L.append(md(big, "{:.0f}"))
    by_rank = p.assign(band=pd.cut(p.actual, [0, 4, 8, 12, 16, 25], labels=["1–4", "5–8", "9–12", "13–16", "17–25"])) \
        .groupby("band", observed=True)["err"].agg(lambda x: x.abs().mean()).reset_index()
    by_rank.columns = ["Actual rank band", "MAE"]
    L.append("\nError by position (the voting pools):\n\n" + md(by_rank))
    g6 = []
    for rel in rels:
        for t in np.where(~np.isnan(rel.rank) & rel.st.g6)[0]:
            g6.append((rel.season, rel.release_no, rel.st.names[t]))
    g6 = pd.DataFrame(g6, columns=["season", "release", "team"]).merge(p, on=["season", "release", "team"])
    if len(g6):
        L.append(f"\nGroup of Five/Six teams: mean signed error {g6.err.mean():+.2f} places, MAE {g6.err.abs().mean():.2f} "
                 f"({len(g6)} team-weeks). Power-conference teams: MAE "
                 f"{p[~p.set_index(['season', 'release', 'team']).index.isin(g6.set_index(['season', 'release', 'team']).index)].err.abs().mean():.2f}.\n")

    # 10. statements
    st_path = DATA / "committee_statements.csv"
    if st_path.exists():
        s = pd.read_csv(st_path)
        L.append("## 10. What the chair says versus what the data support\n")
        L.append(f"`data/committee_statements.csv` holds {len(s)} tagged sentences from {s['date'].nunique()} committee "
                 f"media calls ({int(s.season.min())}–{int(s.season.max())}; earlier transcripts were not obtainable). "
                 "Mentions per season:\n")
        t = s.drop_duplicates(["date", "factor", "quote"]).groupby(["factor", "season"]).size().unstack(fill_value=0)
        t["total"] = t.sum(1)
        L.append(md(t.sort_values("total", ascending=False).reset_index()))
        L.append("\nHow each stated factor shows up in the fitted résumé-only model (weight, base era):\n")
        mp = {"quality_wins": ["win_credit", "best_win", "best_win2", "w_ranked", "w_top10", "w500"],
              "loss_quality": ["losses", "loss_penalty", "worst_loss", "loss_unranked", "loss_sub500"],
              "strength_of_schedule": ["sos", "sos_top4"], "record_strength": ["wae", "sor_z", "record_strength"],
              "head_to_head": ["(pairwise) h2h"], "common_opponents": ["(pairwise) common"],
              "conference_championship": ["champion", "ccg_loss"],
              "game_control": ["game_control"], "margin": ["margin_cap", "margin_loss_cap"],
              "eye_test": ["rating"], "road_wins": ["road_w_ranked", "w_road"],
              "recency": ["last3_value", "trend", "recent_losses", "days_since_loss"],
              "conference_strength": ["conf_strength", "conf_sos_rel"], "group_of_five": ["is_g6", "g6_wae"],
              "availability": ["qb_out", "qb_back"]}
        rows = []
        for fac, feats in mp.items():
            vals = []
            for ft in feats:
                if ft == "(pairwise) h2h":
                    vals.append(f"h2h {prod['pair_level'].b_h2h_:+.2f}")
                elif ft == "(pairwise) common":
                    vals.append(f"common {prod['pair_level'].b_common_:+.2f}")
                else:
                    vals.append(f"{ft} {wl.get(ft, 0):+.2f}")
            rows.append({"Stated factor": fac, "Mentions": int(t.loc[fac, "total"]) if fac in t.index else 0,
                         "Model features and weights": "; ".join(vals)})
        L.append(md(pd.DataFrame(rows)))
    notes = REPORTS / "diagnosis_notes.md"
    if notes.exists():
        L.append("\n" + notes.read_text())
    out = "\n".join(L) + "\n"
    (REPORTS / "backtest_report.md").write_text(out)
    return out


if __name__ == "__main__":
    print(run())
