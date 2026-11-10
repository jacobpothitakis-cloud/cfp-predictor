"""Layer 2: résumé features.

`compute` works on arrays with a leading simulation axis S, so the exact same code produces the
historical feature tables (S=1) and the features inside every Monte Carlo season (S=thousands).

Opponent quality is the as-of rating of the opponent, adjusted for where the game was played.
"An average top-10 team" is the mean rating of the ten best-rated FBS teams at that moment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from cfp.features.state import GameArrays, SeasonState, game_control_from_margin

MARGIN_CAP = 24.0   # the committee's own data caps margin at 24 in its schedule-strength metric
NO_WIN = -25.0      # value of "best win" for a team with no wins
RECENT_DAYS = 22.0


@dataclass
class Pad:
    """Per-team padded view of the game list: row = team, column = that team's k-th game."""
    g: np.ndarray       # (n, K) game index (0 where padded)
    sign: np.ndarray    # (n, K) +1 team was home, -1 away, 0 padding
    opp: np.ndarray     # (n, K) opponent index (n+1 = dummy for padding)
    valid: np.ndarray   # (n, K)
    loc: np.ndarray     # (n, K) +1 home, 0 neutral, -1 road


def build_pad(ga: GameArrays, n: int) -> Pad:
    lists: list[list[tuple[int, int, int]]] = [[] for _ in range(n)]
    order = np.argsort(ga.day, kind="stable")
    for g in order:
        h, a = int(ga.gh[g]), int(ga.ga[g])
        if h < n:
            lists[h].append((g, 1, a))
        if a < n:
            lists[a].append((g, -1, h))
    K = max(1, max(len(x) for x in lists))
    G = np.zeros((n, K), int); S = np.zeros((n, K), int); O = np.full((n, K), n + 1, int)
    for t, lst in enumerate(lists):
        for k, (g, s, o) in enumerate(lst):
            G[t, k], S[t, k], O[t, k] = g, s, o
    valid = S != 0
    loc = np.where(ga.neutral[G], 0, S) * valid
    return Pad(G, S, O, valid, loc)


def _rank_desc(x: np.ndarray) -> np.ndarray:
    """Rank along the last axis, 1 = largest."""
    order = np.argsort(-x, axis=-1, kind="stable")
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.arange(1, x.shape[-1] + 1)[None, :].repeat(x.shape[0], 0), axis=-1)
    return r


def compute(st: SeasonState, ga: GameArrays, margin: np.ndarray, rating: np.ndarray, *, hfa: float,
            sigma: float, now_day: float | None = None, final_week: bool = False,
            pad: Pad | None = None, extra: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Team features.

    margin : (S, G) home margin of every game in `ga`
    rating : (S, n+1) team ratings to use for opponent quality
    extra  : one additional game per team whose opponent differs by simulation (the conference title
             game): {"opp": (S,n) int, "valid": (S,n) bool, "loc": (S,n) int, "m": (S,n) team margin,
             "day": float}
    Returns a dict of (S, n) arrays.
    """
    n = st.n
    S = margin.shape[0]
    pad = pad or build_pad(ga, n)
    now_day = st.cutoff_day if now_day is None else now_day
    f32 = np.float32
    V = pad.valid[None]
    m = (margin[:, pad.g] * pad.sign[None]).astype(f32)
    opp = pad.opp[None]
    loc = pad.loc[None].astype(f32)
    is_ccg = (ga.ccg[pad.g] & pad.valid)[None]
    confg = (ga.conf_game[pad.g] & pad.valid)[None] & ~is_ccg
    gday = ga.day[pad.g][None]
    rk_game = np.where(pad.valid, np.where(pad.sign > 0, ga.rank_a[pad.g], ga.rank_h[pad.g]), np.nan)[None]
    gc_static = np.where(pad.sign > 0, ga.gc_h[pad.g], 1 - ga.gc_h[pad.g])[None]
    pg_static = np.where(pad.sign > 0, ga.pg_wp_h[pad.g], 1 - ga.pg_wp_h[pad.g])[None]
    if extra is not None:
        K = pad.valid.shape[1]

        def cat(base: np.ndarray, col: np.ndarray) -> np.ndarray:
            return np.concatenate([np.broadcast_to(base, (S, n, K)), col[:, :, None].astype(base.dtype)], 2)

        ev = extra["valid"]
        V = cat(V, ev); m = cat(m, np.where(ev, extra["m"], 0.0)); opp = cat(opp, np.where(ev, extra["opp"], n + 1))
        loc = cat(loc, np.where(ev, extra["loc"], 0)); is_ccg = cat(is_ccg, ev); confg = cat(confg, np.zeros_like(ev))
        gday = cat(gday, np.full((S, n), float(extra["day"])))
        nan = np.full((S, n), np.nan)
        rk_game = cat(rk_game, extra.get("rank", nan)); gc_static = cat(gc_static, nan); pg_static = cat(pg_static, nan)
    sidx = np.arange(S)[:, None, None]
    win = (m > 0) & V
    loss = (m < 0) & V
    r_ext = np.concatenate([rating, np.zeros((S, 1))], 1).astype(f32)   # dummy column for padding
    q = r_ext[sidx, opp]                                  # opponent rating (S, n, K)
    top10 = np.sort(rating[:, :n], axis=1)[:, -10:].mean(1).astype(f32)[:, None, None]
    v = q - hfa * loc                                     # opponent quality adjusted for venue
    p_top = np.clip(ndtr((top10 - v) / sigma), 0.02, 0.98)   # chance an average top-10 team wins this game
    own = rating[:, :n].astype(f32)

    def bc(x: np.ndarray) -> np.ndarray:
        return np.broadcast_to(x, (S, n)).astype(f32)

    W = bc(win.sum(2)); L = bc(loss.sum(2)); GP = np.maximum(W + L, 1)
    out: dict[str, np.ndarray] = {"wins": W, "losses": L, "games": W + L, "win_pct": W / GP, "rating": own}

    # --- strength of record / schedule / "record strength"
    exp_top = (p_top * V).sum(2)
    var_top = (p_top * (1 - p_top) * V).sum(2)
    out["wae"] = W - exp_top                              # wins above an average top-10 team's expectation
    out["sor_z"] = out["wae"] / np.sqrt(var_top + 0.25)
    diff = (1 - p_top) * V
    out["sos"] = diff.sum(2) / GP
    out["sos_top4"] = np.sort(diff, axis=2)[:, :, -4:].mean(2)   # 2025 enhancement: weight strong opponents
    win_credit = -np.log(p_top) * win                     # big for beating strong teams, ~0 for weak ones
    loss_pen = -np.log(1 - p_top) * loss                  # small for losing to strong teams, big for weak
    out["win_credit"] = win_credit.sum(2)
    out["loss_penalty"] = loss_pen.sum(2)
    out["record_strength"] = out["win_credit"] - out["loss_penalty"]
    gs = win_credit - loss_pen                            # per-game résumé value

    # --- opponent records
    Wf = np.concatenate([W, np.zeros((S, 2), f32)], 1); Lf = np.concatenate([L, np.ones((S, 2), f32)], 1)
    opp_500 = (Wf >= Lf)[sidx, opp] & (opp < n)
    out["w500"] = bc((win & opp_500).sum(2)); out["l500"] = bc((loss & opp_500).sum(2))
    out["g500"] = out["w500"] + out["l500"]
    out["loss_sub500"] = bc((loss & ~opp_500).sum(2))
    pw = np.concatenate([st.power, [False, False]])[opp]
    out["w_power"] = bc((win & pw).sum(2)); out["l_power"] = bc((loss & pw).sum(2))
    fcs = (opp == n) & V
    out["fcs_games"] = bc(fcs.sum(2))
    out["fcs_losses"] = bc((loss & fcs).sum(2))

    # --- ranked opponents: as ranked at kickoff (polls), and as rated now (proxy for "currently ranked")
    out["w_ranked_g"] = bc((win & (rk_game <= 25)).sum(2))
    out["l_ranked_g"] = bc((loss & (rk_game <= 25)).sum(2))
    out["w_top10_g"] = bc((win & (rk_game <= 10)).sum(2))
    # "currently ranked" proxy: top 25 / top 10 by a blend of résumé (wae) and rating, no model involved
    comp = _z(out["wae"]) + _z(own)
    crank = np.concatenate([_rank_desc(comp), np.full((S, 2), 999)], 1)
    rk_now = crank[sidx, opp]
    r25, r10 = rk_now <= 25, rk_now <= 10
    out["w_ranked"] = bc((win & r25).sum(2)); out["l_ranked"] = bc((loss & r25).sum(2))
    out["w_top10"] = bc((win & r10).sum(2)); out["l_top10"] = bc((loss & r10).sum(2))
    out["road_w_ranked"] = bc((win & r25 & (loc < 0)).sum(2))
    out["loss_unranked"] = bc((loss & ~r25).sum(2))
    out["proxy_rank"] = crank[:, :n].astype(f32)

    # --- best wins / worst loss (venue-adjusted opponent quality, relative to an average top-10 team)
    rel = v - top10
    wv = np.sort(np.where(win, rel, NO_WIN), axis=2)
    out["best_win"] = wv[:, :, -1]; out["best_win2"] = wv[:, :, -2] if wv.shape[2] > 1 else wv[:, :, -1]
    out["best_win3"] = wv[:, :, -3] if wv.shape[2] > 2 else wv[:, :, -1]
    out["worst_loss"] = bc(np.where(loss, rel, 10.0).min(2))
    out["no_loss"] = (L == 0).astype(f32)

    # --- venue splits
    for nm, cond in (("home", loc > 0), ("road", loc < 0), ("neutral", (loc == 0) & V)):
        out[f"w_{nm}"] = bc((win & cond).sum(2)); out[f"l_{nm}"] = bc((loss & cond).sum(2))

    # --- margin / dominance (kept apart from wins)
    out["margin_cap"] = (np.clip(m, -MARGIN_CAP, MARGIN_CAP) * V).sum(2) / GP
    out["margin_loss_cap"] = (np.clip(m, -MARGIN_CAP, 0) * loss).sum(2) / np.maximum(L, 1)
    gc = np.where(np.isnan(gc_static), game_control_from_margin(m), gc_static)
    out["game_control"] = (gc * V).sum(2) / GP
    pg = np.where(np.isnan(pg_static), ndtr(m / 12.0), pg_static)
    out["postgame_wp"] = (pg * V).sum(2) / GP

    # --- recency (games are in chronological order along the last axis)
    from_end = np.cumsum(V[:, :, ::-1], axis=2)[:, :, ::-1]          # 1 = most recent game
    from_start = np.cumsum(V, axis=2)
    cnt = V.sum(2, keepdims=True)
    last3 = V & (from_end <= 3)
    out["last3_wins"] = bc((win & last3).sum(2))
    out["last3_value"] = (gs * last3).sum(2) / np.maximum(last3.sum(2), 1)
    first_half = V & (from_start <= cnt // 2)
    second_half = V & ~first_half
    out["trend"] = (gs * second_half).sum(2) / np.maximum(second_half.sum(2), 1) - \
        (gs * first_half).sum(2) / np.maximum(first_half.sum(2), 1)
    last_loss_day = np.where(loss, gday, -1e9).max(2)
    out["days_since_loss"] = np.clip(now_day - last_loss_day, 0, 120).astype(f32)
    out["recent_losses"] = bc((loss & (gday > now_day - RECENT_DAYS)).sum(2))

    # --- this week (since the previous ranking): the movement inputs
    wk = (gday > now_day - 7.5) & V
    out["played_week"] = bc(wk.any(2))
    out["won_week"] = bc((win & wk).any(2))
    out["lost_week"] = bc((loss & wk).any(2))
    out["week_value"] = (gs * wk).sum(2)
    out["week_margin"] = (np.clip(m, -MARGIN_CAP, MARGIN_CAP) * wk).sum(2)
    out["week_opp"] = np.where(wk.any(2), (rel * wk).sum(2) / np.maximum(wk.sum(2), 1), 0.0)

    # --- conference
    cw = bc((win & confg).sum(2)); cl = bc((loss & confg).sum(2))
    out["conf_wins"], out["conf_losses"] = cw, cl
    out["conf_pct"] = cw / np.maximum(cw + cl, 1)
    out["ccg_played"] = bc(is_ccg.any(2))
    out["champion"] = bc((win & is_ccg).any(2))
    out["ccg_loss"] = bc((loss & is_ccg).any(2))
    confs = sorted(set(st.conf))
    cidx = np.array([confs.index(c) for c in st.conf])
    onehot = np.eye(len(confs), dtype=f32)[cidx]                    # (n, C)
    size = onehot.sum(0)
    conf_mean = (own @ onehot) / size                               # (S, C)
    out["conf_strength"] = np.where(st.indep[None], 0.0, conf_mean[:, cidx])
    ncg = np.maximum(confg.sum(2), 1)
    conf_opp = bc((q * confg).sum(2) / ncg)                         # mean rating of league opponents faced
    peers = (conf_opp @ onehot) / size
    out["conf_sos_rel"] = np.where(st.indep[None] | (confg.sum(2) == 0), 0.0, conf_opp - peers[:, cidx])

    # --- flags
    for nm, arr in (("is_power", st.power), ("is_g6", st.g6), ("is_indep", st.indep)):
        out[nm] = bc(arr.astype(f32)[None])
    out["idle_final"] = (1 - out["played_week"]) * float(final_week)
    return out


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)


def pairwise(st: SeasonState, ga: GameArrays, margin: np.ndarray, cand: np.ndarray,
             pad: Pad | None = None, extra: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Head-to-head and common-opponent comparisons among candidate teams `cand` (indices).

    h2h[s, i, j]    : +1 if i beat j more than j beat i, -1 the reverse, 0 otherwise / not played
    common[s, i, j] : (i's net wins - j's net wins) against opponents both have played, per common opponent
    n_common[i, j]  : number of common opponents
    """
    n = st.n
    pad = pad or build_pad(ga, n)
    S = margin.shape[0]
    mc = len(cand)
    net = np.zeros((S, mc, n + 2), np.float32)
    played = np.zeros((mc, n + 2), bool)
    rows = np.arange(mc)
    for k in range(pad.valid.shape[1]):
        g, sg, op, ok = pad.g[cand, k], pad.sign[cand, k], pad.opp[cand, k], pad.valid[cand, k]
        res = np.sign(margin[:, g] * sg[None]) * ok[None]
        net[:, rows, op] += res
        played[rows, op] |= ok
    if extra is not None:
        # the title game: counts toward head-to-head only (who is a "common opponent" stays schedule-based)
        ev = extra["valid"][:, cand]
        eo = np.where(ev, extra["opp"][:, cand], n + 1)
        res = np.sign(extra["m"][:, cand]) * ev
        np.add.at(net, (np.arange(S)[:, None], rows[None, :], eo), res)
    net[:, :, n:] = 0; played[:, n:] = False                      # FCS / padding are not common opponents
    h2h = np.clip(net[:, :, cand], -1, 1)
    comm = (played[:, None, :] & played[None, :, :]).astype(np.float32)   # (mc, mc, n+2)
    for a in range(mc):                                           # i and j themselves are not "common"
        comm[a, :, cand[a]] = 0
        comm[:, a, cand[a]] = 0
    ncomm = comm.sum(2)
    co_i = np.stack([net[:, i, :] @ comm[i].T for i in range(mc)], 1)          # (S, i, j)
    co_j = np.stack([net[:, j, :] @ comm[:, j].T for j in range(mc)], 2)       # (S, i, j)
    common = (co_i - co_j) / np.maximum(ncomm, 1)[None]
    return {"h2h": h2h, "common": common, "n_common": ncomm}
