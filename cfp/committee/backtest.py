"""Leave-one-season-out backtest of the committee model against simple baselines."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cfp.committee import dataset
from cfp.committee.dataset import Release
from cfp.committee.field import select_field
from cfp.committee.model import Ensemble, LambdaCommittee, PairwiseCommittee

ND = "Notre Dame"


# ------------------------------------------------------------------ baselines
def _fill(primary: np.ndarray, rel: Release) -> np.ndarray:
    """Order by `primary` (lower = better, nan = unranked), unranked teams after by the résumé proxy."""
    key = np.where(np.isnan(primary), 100 + rel.raw["proxy_rank"].to_numpy(), primary)
    return np.argsort(key, kind="stable")


def base_ap(rel: Release) -> np.ndarray:
    return _fill(rel.st.ap, rel)


def base_prev(rel: Release) -> np.ndarray:
    """Last week's committee ranking adjusted only for results: a loss costs six places.
    For the first release of a season there is no previous committee ranking, so the AP poll stands in."""
    prev = rel.st.prev_cfp if rel.st.has_prev_cfp else rel.st.ap
    return _fill(prev + 6.0 * rel.raw["lost_week"].to_numpy(), rel)


def base_sor(rel: Release) -> np.ndarray:
    return np.argsort(-rel.raw["wae"].to_numpy(), kind="stable")


BASELINES = {"AP poll": base_ap, "Prev rank + results": base_prev, "Strength of record": base_sor}


# ------------------------------------------------------------------ metrics
def champions_of(rel: Release) -> dict[str, int]:
    ch = rel.raw["champion"].to_numpy() > 0
    return {rel.st.conf[i]: int(i) for i in np.where(ch)[0]}


def field_of(order: np.ndarray, rel: Release):
    nd = rel.st.names.index(ND) if ND in rel.st.names else None
    return select_field(list(map(int, order)), list(rel.st.conf), champions_of(rel), rel.season, nd)


def metrics(order: np.ndarray, rel: Release) -> dict[str, float]:
    n = len(order)
    pred = np.empty(n); pred[order] = np.arange(1, n + 1)
    ranked = np.where(~np.isnan(rel.rank))[0]
    a, p = rel.rank[ranked], pred[ranked]
    err = np.abs(p - a)
    act_order = ranked[np.argsort(a)]
    out = {"mae": float(err.mean()), "mae_top12": float(err[a <= 12].mean()),
           "spearman": float(spearmanr(a, p)[0]),
           "within1": float((err <= 1).mean()), "within2": float((err <= 2).mean()), "exact": float((err == 0).mean()),
           "top4_exact": float(set(order[:4]) == set(act_order[:4])),
           "top4_overlap": len(set(order[:4]) & set(act_order[:4])) / 4,
           "top12_exact": float(set(order[:12]) == set(act_order[:12])),
           "top12_overlap": len(set(order[:12]) & set(act_order[:12])) / 12,
           "top25_overlap": len(set(order[:25]) & set(act_order)) / 25}
    if rel.is_final:
        # actual field from the actual ranking (unranked teams ordered after, by our prediction)
        rest = [t for t in order if t not in set(act_order)]
        fa = field_of(np.concatenate([act_order, rest]).astype(int), rel)
        fp = field_of(order, rel)
        out["field_correct"] = float(fa.teams == fp.teams)
        out["field_overlap"] = len(fa.teams & fp.teams) / len(fa.teams)
        out["seeds_correct"] = float(np.mean([a == b for a, b in zip(fa.seeds, fp.seeds)]))
    return out


# ------------------------------------------------------------------ LOSO
PRIMARY = "Ensemble (+polls, last week)"
SIM_MODEL = "Ensemble (carry-over from first ranking)"


def fit_models(train: list[Release], target_season: int, C: float = 0.3) -> dict[str, object]:
    pl = PairwiseCommittee("level", C=C, target_season=target_season).fit(train)
    pf = PairwiseCommittee("full", C=C, target_season=target_season).fit(train)
    ll = LambdaCommittee("level", target_season=target_season).fit(train)
    lf = LambdaCommittee("full", target_season=target_season).fit(train)
    atrain = dataset.anchor_training_set(train)
    pa = PairwiseCommittee("anchor", C=C, target_season=target_season).fit(atrain)
    la = LambdaCommittee("anchor", target_season=target_season).fit(atrain)
    return {"Pairwise logistic (résumé only)": pl, "Pairwise logistic (+polls, last week)": pf,
            "LambdaRank (résumé only)": ll, "LambdaRank (+polls, last week)": lf,
            "Ensemble (résumé only)": Ensemble(pl, ll), PRIMARY: Ensemble(pf, lf),
            "_anchor": Ensemble(pa, la), "_anchor_pair": pa}


def _record(rows, preds, name, order, rel, score=None):
    rows.append({"model": name, "season": rel.season, "release": rel.release_no, "is_final": rel.is_final,
                 **metrics(order, rel)})
    n = len(order); pr = np.empty(n); pr[order] = np.arange(1, n + 1)
    for t in np.where(~np.isnan(rel.rank) | (pr <= 30))[0]:
        preds.append({"model": name, "season": rel.season, "release": rel.release_no, "is_final": rel.is_final,
                      "team": rel.st.names[t], "pred": int(pr[t]), "actual": rel.rank[t],
                      "score": np.nan if score is None else score[t]})


def loso(seasons=range(2014, 2026), C: float = 0.3, verbose: bool = True) -> dict[str, object]:
    """Returns {"metrics": df, "preds": df, "pairs": {model name: [(fitted model, held-out release)]}}."""
    rels = list(dataset.all_releases())
    rows, preds = [], []
    pairs: dict[str, list] = {PRIMARY: [], SIM_MODEL: []}
    for held in seasons:
        train = [r for r in rels if r.season != held]
        test = sorted([r for r in rels if r.season == held], key=lambda r: r.release_no)
        models = fit_models(train, held, C)
        for i, rel in enumerate(test):
            for k, m in models.items():
                if not k.startswith("_"):
                    _record(rows, preds, k, m.order(rel), rel, m.score(rel))
            for k, f in BASELINES.items():
                _record(rows, preds, k, f(rel), rel)
            pairs[PRIMARY].append((models[PRIMARY], rel))
            if i >= 1:   # carry-over models: anchored on last week's ranking, and on the season's first ranking
                r1 = dataset.with_anchor(rel, test[i - 1], 1)
                _record(rows, preds, "Ensemble (carry-over from last week, no polls)", models["_anchor"].order(r1), r1)
                r0 = dataset.with_anchor(rel, test[0], i)
                _record(rows, preds, SIM_MODEL, models["_anchor"].order(r0), r0)
                _record(rows, preds, "First ranking adjusted for new losses",
                        _fill(test[0].rank + 6.0 * (rel.raw["losses"] - test[0].raw["losses"]).to_numpy(), rel), rel)
                pairs[SIM_MODEL].append((models["_anchor"], r0))
        if verbose:
            d = pd.DataFrame([r for r in rows if r["season"] == held]).groupby("model")["mae"].mean()
            print(held, f"primary={d[PRIMARY]:.2f} AP={d['AP poll']:.2f} prev={d['Prev rank + results']:.2f} "
                        f"resume={d['Ensemble (résumé only)']:.2f} sim={d[SIM_MODEL]:.2f}", flush=True)
    return {"metrics": pd.DataFrame(rows), "preds": pd.DataFrame(preds), "pairs": pairs}


ABLATION_GROUPS = {
    "Losses and record": ["losses", "losses_sq", "wins", "win_pct", "no_loss"],
    "Team strength and dominance (rating, margin, game control)": ["rating", "margin_cap", "margin_loss_cap",
                                                                   "game_control", "postgame_wp"],
    "Quality of wins": ["win_credit", "best_win", "best_win2", "best_win3", "w_ranked", "w_top10", "w_ranked_g",
                        "w_top10_g", "w500", "w_power", "road_w_ranked", "w_road"],
    "Quality of losses": ["loss_penalty", "worst_loss", "loss_unranked", "loss_sub500", "l_ranked", "l_top10",
                          "l_ranked_g", "l500", "l_power", "fcs_losses", "l_home"],
    "Schedule strength, strength of record, record strength": ["sos", "sos_top4", "wae", "sor_z", "record_strength"],
    "Margin and game control only": ["margin_cap", "margin_loss_cap", "game_control", "postgame_wp"],
    "Recency and this week's result": ["last3_value", "last3_wins", "trend", "days_since_loss", "recent_losses",
                                       "played_week", "won_week", "lost_week", "week_value", "week_opp",
                                       "week_margin", "idle_final"],
    "Conference title and conference strength": ["champion", "ccg_loss", "ccg_played", "conf_strength", "conf_sos_rel"],
    "Power / Group of Six / independent status": ["is_power", "is_g6", "is_indep", "g6_wae"],
    "Brand (preseason AP rank)": ["preseason"],
}


def ablation(C: float = 0.3) -> pd.DataFrame:
    """Leave-one-season-out error of the résumé-only pairwise model with one feature group removed."""
    rels = list(dataset.all_releases())
    configs = {"(nothing removed)": ((), True), "Head-to-head and common opponents": ((), False)}
    configs.update({k: (tuple(v), True) for k, v in ABLATION_GROUPS.items()})
    rows = []
    for name, (drop, pw) in configs.items():
        ms = []
        for held in range(2014, 2026):
            train = [r for r in rels if r.season != held]
            mdl = PairwiseCommittee("level", C=C, target_season=held, exclude=drop, use_pairwise=pw).fit(train)
            ms += [metrics(mdl.order(r), r) for r in rels if r.season == held]
        d = pd.DataFrame(ms)
        rows.append({"Removed": name, "MAE": d["mae"].mean(), "MAE top 12": d["mae_top12"].mean(),
                     "Top-12 overlap": d["top12_overlap"].mean(), "Within 1": d["within1"].mean()})
    out = pd.DataFrame(rows)
    base = out.iloc[0]
    out["MAE change"] = out["MAE"] - base["MAE"]
    return out


if __name__ == "__main__":
    res = loso()
    cols = ["mae", "mae_top12", "spearman", "within1", "within2", "top4_exact", "top12_exact", "top12_overlap",
            "field_correct", "seeds_correct"]
    m = res["metrics"]
    print(m.groupby("model")[cols].mean().round(3).sort_values("mae").to_string())
    print(m[m.is_final].groupby("model")[cols].mean().round(3).sort_values("mae").to_string())
