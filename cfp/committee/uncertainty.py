"""Ranking distributions.

The point prediction is a score per team in log-odds units. Uncertainty about what thirteen people
will do is modelled as independent Gaussian noise on those scores; re-ranking each noisy draw gives
P(rank = k) for every team. The noise scale is calibrated on leave-one-season-out predictions so that
the typical distance between a draw and the point prediction matches the typical distance between the
committee's actual ranking and the point prediction.
"""
from __future__ import annotations

import numpy as np

from cfp.committee.dataset import Release


def sample_orders(model, rel: Release, sigma: float, n: int = 2000, seed: int = 0,
                  score: np.ndarray | None = None) -> np.ndarray:
    """(n, m) candidate positions (index into rel.cand) best-first for each noisy draw."""
    rng = np.random.default_rng(seed)
    s = (model.score(rel) if score is None else score)[rel.cand]
    pair = model.pair if hasattr(model, "pair") else model
    draws = s[None, :] + sigma * rng.standard_normal((n, len(s)))
    if hasattr(pair, "borda"):
        b = pair.borda(draws, rel.pw["h2h"][None], rel.pw["common"][None], rel.pw["n_common"][None])
    else:
        b = draws
    return np.argsort(-b, axis=1, kind="stable")


def rank_distribution(model, rel: Release, sigma: float, n: int = 2000, seed: int = 0,
                      max_rank: int = 30, score: np.ndarray | None = None) -> np.ndarray:
    """P[team_candidate, k] = probability the candidate is ranked k+1 (last column = lower than max_rank)."""
    orders = sample_orders(model, rel, sigma, n, seed, score)
    m = orders.shape[1]
    pos = np.empty_like(orders)
    np.put_along_axis(pos, orders, np.arange(m)[None, :].repeat(n, 0), axis=1)
    P = np.zeros((m, max_rank + 1))
    for k in range(max_rank):
        P[:, k] = (pos == k).mean(0)
    P[:, max_rank] = (pos >= max_rank).mean(0)
    return P


def calibrate_sigma(pairs: list[tuple[object, Release]], grid=(0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 1.8, 2.3),
                    n: int = 300) -> tuple[float, dict[float, float], float]:
    """pairs: (fitted model, held-out release). Returns (sigma, {sigma: simulated MAE}, observed MAE)."""
    obs, sim = [], {g: [] for g in grid}
    for model, rel in pairs:
        order = model.order(rel)
        npos = len(order)
        pred = np.empty(npos); pred[order] = np.arange(1, npos + 1)
        ranked = np.where(~np.isnan(rel.rank))[0]
        obs.append(np.abs(pred[ranked] - rel.rank[ranked]).mean())
        cpos = {t: i for i, t in enumerate(rel.cand)}
        rc = np.array([cpos[t] for t in order[:25]])           # the predicted top 25, as candidate positions
        for g in grid:
            o = sample_orders(model, rel, g, n, seed=1)
            pos = np.empty_like(o)
            np.put_along_axis(pos, o, np.arange(o.shape[1])[None, :].repeat(n, 0), axis=1)
            sim[g].append(np.abs(pos[:, rc] - np.arange(25)[None, :]).mean())
    obs_mae = float(np.mean(obs))
    sim_mae = {g: float(np.mean(v)) for g, v in sim.items()}
    gs = np.array(sorted(sim_mae)); ms = np.array([sim_mae[g] for g in gs])
    sigma = float(np.interp(obs_mae, ms, gs))
    return sigma, sim_mae, obs_mae


def coverage(pairs: list[tuple[object, Release]], sigma: float, n: int = 400) -> dict[str, float]:
    """Calibration check: how often the actual rank falls inside the central 50% / 80% / 95% interval,
    and the mean probability assigned to the actual rank."""
    hits = {0.5: [], 0.8: [], 0.95: []}
    for model, rel in pairs:
        o = sample_orders(model, rel, sigma, n, seed=2)
        pos = np.empty_like(o)
        np.put_along_axis(pos, o, np.arange(o.shape[1])[None, :].repeat(n, 0), axis=1)
        for ci, t in enumerate(rel.cand):
            a = rel.rank[t]
            if np.isnan(a):
                continue
            ranks = pos[:, ci] + 1
            for lvl in hits:
                lo, hi = np.quantile(ranks, [(1 - lvl) / 2, 1 - (1 - lvl) / 2])
                hits[lvl].append(lo <= a <= hi)
    return {f"cover{int(k * 100)}": float(np.mean(v)) for k, v in hits.items()}
