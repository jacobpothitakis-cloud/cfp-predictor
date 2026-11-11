"""Walk-forward test of the weekly ranking model, run the way it is used in production.

For every season 2014-2025 and every ranking in it, the model is trained on all other seasons plus that
season's EARLIER rankings, then predicts the ranking. (The leave-one-season-out backtest never sees the
season's own earlier rankings, so it cannot judge anything that learns within a season.)

    from cfp.committee import walkforward as wf
    wf.summary(wf.run({}))                       # current settings
    wf.summary(wf.run({"season_boost": 10}))     # a variant
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd

from cfp.committee import backtest as bt
from cfp.committee import dataset, model
from cfp.committee.model import Ensemble, LambdaCommittee, PairwiseCommittee

_RELS: list | None = None
DEFAULTS = {"season_boost": None, "recency": None, "era_scale": None, "C": 0.3, "w_pair": bt.PRIMARY_W_PAIR, "exclude": (),
            "lambda": True, "variant": "full", "lam_params": None, "extra": ()}


def _season(args) -> list[dict]:
    S, cfg = args
    c = {**DEFAULTS, **cfg}
    if c["season_boost"] is not None:
        model.SEASON_BOOST = float(c["season_boost"])
    if c["recency"] is not None:
        model.RECENCY = float(c["recency"])
    if c["era_scale"] is not None:
        model.ERA_SCALE = float(c["era_scale"])
    rels = _RELS
    other = [r for r in rels if r.season != S]
    cur = sorted([r for r in rels if r.season == S], key=lambda r: r.release_no)
    out = []
    for k, rel in enumerate(cur):
        train = other + cur[:k]
        pf = PairwiseCommittee(c["variant"], C=c["C"], target_season=S, exclude=tuple(c["exclude"]), extra=tuple(c["extra"])).fit(train)
        if c["lambda"] and c["w_pair"] < 1:
            lf = LambdaCommittee(c["variant"], target_season=S, exclude=tuple(c["exclude"]), extra=tuple(c["extra"]), params=c["lam_params"]).fit(train)
            order = Ensemble(pf, lf, w_pair=c["w_pair"]).order(rel)
        else:
            order = pf.order(rel)
        mt = bt.metrics(order, rel)
        n = len(order); pred = np.empty(n); pred[order] = np.arange(1, n + 1)
        rk = np.where(~np.isnan(rel.rank))[0]
        a, p = rel.rank[rk], pred[rk]
        e = np.abs(p - a)
        out.append({"season": S, "release": rel.release_no, "final": rel.is_final, **mt,
                    "mae_1_8": float(e[a <= 8].mean()), "mae_9_16": float(e[(a > 8) & (a <= 16)].mean()),
                    "mae_17_25": float(e[a > 16].mean()), "bias_sq": float(np.mean(p - a))})
    return out


def run(cfg: dict | None = None, seasons=range(2014, 2026), rels: list | None = None, workers: int = 8) -> pd.DataFrame:
    global _RELS
    _RELS = list(rels) if rels is not None else list(dataset.all_releases())
    jobs = [(S, cfg or {}) for S in seasons]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("fork")) as ex:
        rows = [r for part in ex.map(_season, jobs) for r in part]
    return pd.DataFrame(rows)


def summary(d: pd.DataFrame) -> dict:
    f = d[d.final]
    return {"mae": round(d.mae.mean(), 4), "within1": round(d.within1.mean(), 3), "top4": round(d.top4_overlap.mean(), 3),
            "top12": round(d.top12_overlap.mean(), 3), "top25": round(d.top25_overlap.mean(), 3),
            "mae_1_8": round(d.mae_1_8.mean(), 3), "mae_9_16": round(d.mae_9_16.mean(), 3),
            "mae_17_25": round(d.mae_17_25.mean(), 3),
            "final_mae": round(f.mae.mean(), 3), "field": int(f.field_correct.sum()), "n": len(d)}


def compare(base: pd.DataFrame, new: pd.DataFrame) -> str:
    b, n = base.groupby("season").mae.mean(), new.groupby("season").mae.mean()
    return f"mae {base.mae.mean():.4f} -> {new.mae.mean():.4f}; better in {(n < b - 1e-9).sum()} of {len(b)} seasons, worse in {(n > b + 1e-9).sum()}"
