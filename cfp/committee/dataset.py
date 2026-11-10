"""Training / prediction tables for the committee model: one frame per ranking release."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import pandas as pd

from cfp import data, ratings
from cfp.features import engine
from cfp.features.state import SeasonState, build_state

# continuous features are z-scored across all FBS teams within a release (the committee compares
# teams to each other in the same week); counts and flags are used raw.
Z_FEATURES = ["rating", "wae", "sor_z", "sos", "sos_top4", "win_credit", "loss_penalty", "record_strength",
              "margin_cap", "game_control", "postgame_wp", "last3_value", "trend", "conf_strength",
              "conf_sos_rel", "best_win", "best_win2", "best_win3", "worst_loss", "week_value", "week_opp"]
RAW_FEATURES = ["losses", "wins", "win_pct", "w500", "l500", "loss_sub500", "w_power", "l_power", "fcs_losses",
                "w_ranked", "l_ranked", "w_top10", "l_top10", "w_ranked_g", "l_ranked_g", "w_top10_g",
                "road_w_ranked", "loss_unranked", "no_loss", "w_road", "l_home", "margin_loss_cap",
                "last3_wins", "days_since_loss", "recent_losses", "played_week", "won_week", "lost_week",
                "week_margin", "champion", "ccg_loss", "ccg_played", "is_power", "is_g6", "is_indep",
                "idle_final"]
SCALE = {"days_since_loss": 1 / 30.0, "week_margin": 1 / 10.0, "margin_loss_cap": 1 / 10.0}
LEVEL_FEATURES = Z_FEATURES + RAW_FEATURES + ["preseason", "qb_out", "qb_back", "losses_sq", "g6_wae"]
# prev = last week's committee position (0 for everyone at a season's first release, when the AP/Coaches
# polls have to carry the "prior opinion": ap_first / coaches_first let them carry more weight then).
POLL_FEATURES = ["ap", "coaches", "prev", "prev_lost", "ap_first", "coaches_first", "ap_lost_first"]
# "anchor": an earlier committee ranking from the same season, `lag` releases back. This is what the
# simulation uses, because future AP polls and future committee rankings do not exist yet.
ANCHOR_FEATURES = ["anc1", "anc2", "anc3", "anc_lost", "new_losses", "anc_missing"]
ERA_FEATURES = ["losses", "wae", "sos", "sos_top4", "record_strength", "rating", "best_win", "worst_loss",
                "champion", "is_g6", "margin_cap", "w500", "conf_strength", "conf_sos_rel", "loss_unranked"]


def poll_score(rank: np.ndarray) -> np.ndarray:
    """Rank 1..25 -> 1.0..0.04; unranked -> 0."""
    return np.where(np.isnan(rank), 0.0, (26 - rank) / 25.0)


@dataclass
class Release:
    season: int
    cutoff: pd.Timestamp
    release_no: int
    is_final: bool
    st: SeasonState
    X: pd.DataFrame              # one row per FBS team, model features
    raw: pd.DataFrame            # unscaled features (for explanations / résumé sheets)
    rank: np.ndarray             # actual committee rank (nan = unranked / unknown)
    cand: np.ndarray             # candidate team indices
    pw: dict[str, np.ndarray]    # pairwise arrays among candidates (S axis dropped)

    @property
    def key(self) -> str:
        return f"{self.season}-{self.release_no}"


def feature_tensor(st: SeasonState, f: dict[str, np.ndarray], *, is_final: bool,
                   availability: dict[str, dict] | None = None,
                   anchor: dict | None = None) -> dict[str, np.ndarray]:
    """Model features as {name: (S, n)} arrays. The single source of truth for both the historical
    frames (S=1) and the simulation (S=thousands).

    anchor: {"rank": (n,) earlier committee ranking, "lag": int, "losses_then": (n,)} or None.
    """
    S, n = f["wins"].shape
    X: dict[str, np.ndarray] = {}
    for k in Z_FEATURES:
        x = f[k].astype(float)
        X[k] = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)
    for k in RAW_FEATURES:
        X[k] = f[k].astype(float) * SCALE.get(k, 1.0)

    def row(v: np.ndarray) -> np.ndarray:
        return np.broadcast_to(np.asarray(v, float)[None, :], (S, n))

    X["preseason"] = row(poll_score(st.preseason_ap))
    X["losses_sq"] = np.minimum(f["losses"].astype(float), 4) ** 2 / 4
    X["g6_wae"] = X["is_g6"] * X["wae"]
    av = availability or {}
    X["qb_out"] = row([float(av.get(nm, {}).get("qb_out", 0)) for nm in st.names])
    X["qb_back"] = row([float(av.get(nm, {}).get("qb_returning", 0)) for nm in st.names])
    X["ap"] = row(poll_score(st.ap))
    X["coaches"] = row(poll_score(st.coaches))
    first = 0.0 if st.has_prev_cfp else 1.0
    X["prev"] = row(poll_score(st.prev_cfp) if st.has_prev_cfp else np.zeros(n))
    X["prev_lost"] = X["prev"] * X["lost_week"]
    X["ap_first"] = X["ap"] * first
    X["coaches_first"] = X["coaches"] * first
    X["ap_lost_first"] = X["ap"] * X["lost_week"] * first
    for k in ANCHOR_FEATURES:
        X[k] = np.zeros((S, n))
    if anchor is None:
        X["anc_missing"] = np.ones((S, n))
    else:
        a = row(poll_score(anchor["rank"]))
        lag = int(anchor["lag"])
        dl = np.clip(f["losses"].astype(float) - np.asarray(anchor["losses_then"], float)[None, :], 0, 3)
        X["anc1"], X["anc2"], X["anc3"] = a * (lag == 1), a * (lag == 2), a * (lag >= 3)
        X["anc_lost"] = a * dl
        X["new_losses"] = dl
    return X


def frame_from_features(st: SeasonState, f: dict[str, np.ndarray], *, is_final: bool, s: int = 0,
                        availability: dict[str, dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame({k: v[s] for k, v in f.items()}, index=st.names)
    T = feature_tensor(st, f, is_final=is_final, availability=availability)
    X = pd.DataFrame({k: v[s] for k, v in T.items()}, index=st.names)
    return X, raw


def anchor_columns(X: pd.DataFrame, anchor_rank: np.ndarray | None, lag: int,
                   losses_now: np.ndarray, losses_then: np.ndarray | None) -> pd.DataFrame:
    """Fill the anchor features from a committee ranking `lag` releases earlier."""
    X = X.copy()
    if anchor_rank is None:
        for k in ANCHOR_FEATURES:
            X[k] = 0.0
        X["anc_missing"] = 1.0
        return X
    a = poll_score(anchor_rank)
    dl = np.clip(losses_now - losses_then, 0, 3)
    X["anc1"], X["anc2"], X["anc3"] = a * (lag == 1), a * (lag == 2), a * (lag >= 3)
    X["anc_lost"] = a * dl
    X["new_losses"] = dl
    X["anc_missing"] = 0.0
    return X


def with_anchor(rel: "Release", anchor: "Release | None", lag: int = 0) -> "Release":
    if anchor is None:
        return replace(rel, X=anchor_columns(rel.X, None, 0, rel.raw["losses"].to_numpy(), None))
    return replace(rel, X=anchor_columns(rel.X, anchor.rank, lag, rel.raw["losses"].to_numpy(),
                                         anchor.raw["losses"].to_numpy()))


def anchor_training_set(rels: list["Release"]) -> list["Release"]:
    """Every (release, earlier release) combination within a season, plus first releases with no anchor."""
    out: list[Release] = []
    by: dict[int, list[Release]] = {}
    for r in rels:
        by.setdefault(r.season, []).append(r)
    for season, lst in by.items():
        lst = sorted(lst, key=lambda r: r.release_no)
        for i, r in enumerate(lst):
            if i == 0:
                out.append(with_anchor(r, None))
            for j in range(i):
                out.append(with_anchor(r, lst[j], i - j))
    return out


def candidates(raw: pd.DataFrame, rank: np.ndarray, k: int = 50) -> np.ndarray:
    top = np.argsort(raw["proxy_rank"].to_numpy())[:k]
    ranked = np.where(~np.isnan(rank))[0]
    return np.unique(np.concatenate([top, ranked]))


def build_release(season: int, cutoff: pd.Timestamp, release_no: int, is_final: bool,
                  with_labels: bool = True, availability: dict | None = None) -> Release:
    R = ratings.load()
    st = build_state(season, cutoff)
    f = engine.compute(st, st.played, st.margin[None], st.rating[None], hfa=R.p.hfa, sigma=R.p.sigma,
                       final_week=is_final)
    X, raw = frame_from_features(st, f, is_final=is_final, availability=availability)
    rank = np.full(st.n, np.nan)
    if with_labels:
        act = data.cfp_ranking(season, cutoff)
        for t, r in zip(act["team_id"], act["rank"]):
            rank[st.idx[int(t)]] = r
    cand = candidates(raw, rank)
    pw = {k: (v[0] if v.ndim == 3 else v) for k, v in
          engine.pairwise(st, st.played, st.margin[None], cand).items()}
    return Release(season, st.cutoff, release_no, is_final, st, X, raw, rank, cand, pw)


@lru_cache(maxsize=1)
def all_releases() -> tuple[Release, ...]:
    rel = data.releases()
    return tuple(build_release(int(r.season), r.cutoff, int(r.release_no), bool(r.is_final))
                 for r in rel.itertuples())
