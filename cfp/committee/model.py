"""Layer 3: the committee model.

Stage A  linear score per team from a regularised pairwise logistic regression on feature differences
         (interpretable; weights are log-odds that team i is ranked ahead of team j).
Stage B  head-to-head and common-opponent terms that only matter when two teams are close, mirroring
         the six-team pools the committee votes in.
Ranking  Borda count over the pairwise probabilities (expected number of candidates a team beats).
Alt      LightGBM LambdaRank on the same team features; the two can be averaged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from cfp.committee.dataset import ANCHOR_FEATURES, ERA_FEATURES, LEVEL_FEATURES, POLL_FEATURES, Release

ERA_SCALE = 0.5      # era-deviation columns are shrunk, i.e. penalised harder than the base weights
RECENCY = 0.90       # per-season sample-weight decay
SEASON_BOOST = 6.0   # extra sample weight on rankings already released in the season being predicted ("full" model only)
# Walk-forward test, 2014-2025, predicting each ranking from the second on with the season's earlier rankings in
# the training set: mean error 1.111 at weight 1, 1.075 at 3, 1.059 at 6, 1.060 at 10, 1.057 at 20; better than
# weight 1 in 10 of 12 seasons at 6. Each committee has its own leanings and its first rankings reveal them.


def _weight(target_season: int, season: int, variant: str = "") -> float:
    boost = SEASON_BOOST if (season == target_season and variant == "full") else 1.0
    return RECENCY ** abs(target_season - season) * boost


def era_flags(season: int) -> tuple[float, float]:
    """(12-team era, post-2025 metric changes)."""
    return float(season >= 2024), float(season >= 2025)


def design(X: pd.DataFrame, season: int, features: list[str], use_era: bool) -> tuple[np.ndarray, list[str]]:
    cols = [X[f].to_numpy(float) for f in features]
    names = list(features)
    if use_era:
        e12, e25 = era_flags(season)
        for f in ERA_FEATURES:
            if f in X.columns and f in features:
                cols.append(X[f].to_numpy(float) * e12 * ERA_SCALE); names.append(f"{f}@12team")
                cols.append(X[f].to_numpy(float) * e25 * ERA_SCALE); names.append(f"{f}@2025+")
    return np.column_stack(cols), names


def design_tensor(T: dict[str, np.ndarray], season: int, features: list[str], use_era: bool,
                  idx: np.ndarray | None = None) -> np.ndarray:
    """(S, m, p) design from a feature tensor, same column order as `design`."""
    take = (lambda x: x) if idx is None else (lambda x: x[:, idx])
    cols = [take(T[f]) for f in features]
    if use_era:
        e12, e25 = era_flags(season)
        for f in ERA_FEATURES:
            if f in T and f in features:
                cols.append(take(T[f]) * e12 * ERA_SCALE)
                cols.append(take(T[f]) * e25 * ERA_SCALE)
    return np.stack(cols, axis=-1)


def _pairs(rel: Release) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Candidate pairs (i, j) with at least one ranked team. y = 1 if i is ahead of j."""
    c = rel.cand
    r = np.where(np.isnan(rel.rank[c]), 99.0, rel.rank[c])
    ii, jj = np.triu_indices(len(c), 1)
    keep = (r[ii] < 99) | (r[jj] < 99)
    ii, jj = ii[keep], jj[keep]
    return ii, jj, (r[ii] < r[jj]).astype(float)


@dataclass
class PairwiseCommittee:
    variant: str = "full"            # "level" (résumé only) or "full" (adds polls and last week's rank)
    C: float = 0.3
    use_era: bool = True
    target_season: int = 2026
    exclude: tuple[str, ...] = ()    # features left out (ablation tests)
    use_pairwise: bool = True        # stage B on/off (ablation tests)
    coef_: np.ndarray | None = None
    names_: list[str] = field(default_factory=list)
    b_h2h_: float = 0.0
    b_common_: float = 0.0
    tau_: float = 2.0

    @property
    def features(self) -> list[str]:
        extra = {"full": POLL_FEATURES, "anchor": ANCHOR_FEATURES}.get(self.variant, [])
        return [f for f in LEVEL_FEATURES + extra if f not in self.exclude]

    # ------------------------------------------------------------------ fit
    def fit(self, releases: list[Release]) -> "PairwiseCommittee":
        D, Y, Wt = [], [], []
        for rel in releases:
            M, self.names_ = design(rel.X, rel.season, self.features, self.use_era)
            ii, jj, y = _pairs(rel)
            d = M[rel.cand[ii]] - M[rel.cand[jj]]
            w = np.full(len(y), _weight(self.target_season, rel.season, self.variant))
            D += [d, -d]; Y += [y, 1 - y]; Wt += [w, w]
        D, Y, Wt = np.vstack(D), np.concatenate(Y), np.concatenate(Wt)
        lr = LogisticRegression(C=self.C, fit_intercept=False, max_iter=2000)
        lr.fit(D, Y, sample_weight=Wt)
        self.coef_ = lr.coef_[0]
        if self.use_pairwise:
            self._fit_pairwise(releases)
        return self

    def _fit_pairwise(self, releases: list[Release]) -> None:
        ds, hh, cc, yy, ww = [], [], [], [], []
        for rel in releases:
            s = self.score(rel)[rel.cand]
            ii, jj, y = _pairs(rel)
            ds.append(s[ii] - s[jj]); hh.append(rel.pw["h2h"][ii, jj])
            cc.append(rel.pw["common"][ii, jj] * np.minimum(rel.pw["n_common"][ii, jj], 3) / 3)
            yy.append(y); ww.append(np.full(len(y), _weight(self.target_season, rel.season, self.variant)))
        ds, hh, cc, yy, ww = map(np.concatenate, (ds, hh, cc, yy, ww))
        best = (np.inf, 0.0, 0.0, 2.0)
        for tau in (1.0, 2.0, 4.0):
            k = np.exp(-(ds / tau) ** 2)

            def nll(b: np.ndarray) -> float:
                z = ds + k * (b[0] * hh + b[1] * cc)
                p = np.clip(expit(z), 1e-9, 1 - 1e-9)
                return float(-(ww * (yy * np.log(p) + (1 - yy) * np.log(1 - p))).sum() + 0.5 * (b ** 2).sum())

            res = minimize(nll, np.zeros(2), method="L-BFGS-B")
            if res.fun < best[0]:
                best = (res.fun, float(res.x[0]), float(res.x[1]), tau)
        _, self.b_h2h_, self.b_common_, self.tau_ = best

    # ------------------------------------------------------------------ predict
    def score(self, rel: Release) -> np.ndarray:
        M, _ = design(rel.X, rel.season, self.features, self.use_era)
        return M @ self.coef_

    def score_tensor(self, T: dict[str, np.ndarray], season: int, idx: np.ndarray | None = None) -> np.ndarray:
        return design_tensor(T, season, self.features, self.use_era, idx) @ self.coef_

    def pair_logit(self, s: np.ndarray, h2h: np.ndarray, common: np.ndarray, n_common: np.ndarray) -> np.ndarray:
        """Log-odds that row team is ahead of column team. s: (..., m); pairwise arrays (..., m, m)."""
        d = s[..., :, None] - s[..., None, :]
        k = np.exp(-(d / self.tau_) ** 2)
        return d + k * (self.b_h2h_ * h2h + self.b_common_ * common * np.minimum(n_common, 3) / 3)

    def borda(self, s: np.ndarray, h2h: np.ndarray, common: np.ndarray, n_common: np.ndarray) -> np.ndarray:
        """Expected number of other candidates each team is ranked ahead of."""
        p = expit(self.pair_logit(s, h2h, common, n_common))
        m = s.shape[-1]
        return p.sum(-1) - p[..., np.arange(m), np.arange(m)]

    def order(self, rel: Release, extra_score: np.ndarray | None = None) -> np.ndarray:
        """Team indices, best first, over all FBS teams (candidates by Borda, the rest by raw score)."""
        s = self.score(rel) if extra_score is None else extra_score
        b = self.borda(s[rel.cand], rel.pw["h2h"], rel.pw["common"], rel.pw["n_common"])
        top = rel.cand[np.argsort(-b, kind="stable")]
        rest = np.setdiff1d(np.arange(len(s)), rel.cand)
        return np.concatenate([top, rest[np.argsort(-s[rest], kind="stable")]])

    # ------------------------------------------------------------------ explain
    def weights(self) -> pd.Series:
        return pd.Series(self.coef_, index=self.names_)

    def contributions(self, rel: Release) -> pd.DataFrame:
        """Per-team, per-feature contribution to the score (era columns folded into their base feature)."""
        M, names = design(rel.X, rel.season, self.features, self.use_era)
        C = pd.DataFrame(M * self.coef_[None, :], index=rel.X.index, columns=names)
        base = [n.split("@")[0] for n in names]
        return C.T.groupby(base).sum().T


@dataclass
class LambdaCommittee:
    variant: str = "full"
    target_season: int = 2026
    params: dict | None = None
    model_: lgb.Booster | None = None

    @property
    def features(self) -> list[str]:
        extra = {"full": POLL_FEATURES, "anchor": ANCHOR_FEATURES}.get(self.variant, [])
        return LEVEL_FEATURES + extra + ["era12", "era25"]

    def _M(self, rel: Release) -> np.ndarray:
        e12, e25 = era_flags(rel.season)
        X = rel.X.assign(era12=e12, era25=e25)
        return X[self.features].to_numpy(float)

    def fit(self, releases: list[Release]) -> "LambdaCommittee":
        Xs, ys, groups, ws = [], [], [], []
        for rel in releases:
            M = self._M(rel)[rel.cand]
            r = rel.rank[rel.cand]
            Xs.append(M); ys.append(np.where(np.isnan(r), 0, 26 - r).astype(int)); groups.append(len(r))
            ws.append(np.full(len(r), _weight(self.target_season, rel.season, self.variant)))
        p = {"objective": "lambdarank", "label_gain": [g ** 1.5 for g in range(26)], "lambdarank_truncation_level": 40,
             "learning_rate": 0.04, "num_leaves": 12, "min_data_in_leaf": 25, "feature_fraction": 0.7,
             "lambda_l2": 5.0, "verbose": -1, "seed": 7, "num_threads": 4}
        p.update(self.params or {})
        ds = lgb.Dataset(np.vstack(Xs), label=np.concatenate(ys), group=groups, weight=np.concatenate(ws))
        self.model_ = lgb.train(p, ds, num_boost_round=350)
        return self

    def score(self, rel: Release) -> np.ndarray:
        return self.model_.predict(self._M(rel))

    def score_tensor(self, T: dict[str, np.ndarray], season: int, idx: np.ndarray | None = None) -> np.ndarray:
        e12, e25 = era_flags(season)
        cols = []
        for f in self.features:
            if f in ("era12", "era25"):
                continue
            cols.append(T[f] if idx is None else T[f][:, idx])
        M = np.stack(cols, axis=-1)
        S, m, _ = M.shape
        M = np.concatenate([M, np.full((S, m, 1), e12), np.full((S, m, 1), e25)], -1)
        return self.model_.predict(M.reshape(S * m, -1)).reshape(S, m)

    def order(self, rel: Release) -> np.ndarray:
        s = self.score(rel)
        return np.argsort(-s, kind="stable")


def zscore_on(s: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return (s - s[idx].mean()) / (s[idx].std() + 1e-9)


@dataclass
class Ensemble:
    pair: PairwiseCommittee
    lam: LambdaCommittee
    w_pair: float = 0.5

    def score(self, rel: Release) -> np.ndarray:
        """Blend in the pairwise model's log-odds units so stage B and the noise model still apply."""
        sp = self.pair.score(rel)
        sl = self.lam.score(rel)
        c = rel.cand
        sl = (sl - sl[c].mean()) / (sl[c].std() + 1e-9) * sp[c].std() + sp[c].mean()
        return self.w_pair * sp + (1 - self.w_pair) * sl

    def score_tensor(self, T: dict[str, np.ndarray], season: int, cand: np.ndarray) -> np.ndarray:
        """(S, m) blended scores for the candidate teams, standardised per simulation like `score`."""
        sp = self.pair.score_tensor(T, season, cand)
        sl = self.lam.score_tensor(T, season, cand)
        sl = (sl - sl.mean(1, keepdims=True)) / (sl.std(1, keepdims=True) + 1e-9) * sp.std(1, keepdims=True) \
            + sp.mean(1, keepdims=True)
        return self.w_pair * sp + (1 - self.w_pair) * sl

    def order(self, rel: Release) -> np.ndarray:
        return self.pair.order(rel, extra_score=self.score(rel))
