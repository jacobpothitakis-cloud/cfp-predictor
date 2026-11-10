"""Layer 5: Monte Carlo of the rest of the season, the committee's final ranking, the field and the playoff.

One simulation:
  1. draw each team's true strength around its current rating (shared across its remaining games)
  2. simulate every remaining regular-season game (score margin)
  3. update the ratings the committee model sees with those results (first-order ridge update)
  4. resolve every conference race with the league's tiebreakers and simulate its title game
  5. recompute every résumé feature for the simulated season and apply the committee model (carry-over
     variant, anchored on the latest real committee ranking) with calibrated noise
  6. apply the selection and seeding rules, build the bracket, and play the playoff
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from cfp import data, ratings
from cfp.committee import dataset
from cfp.committee.field import select_field
from cfp.committee.train import load_production
from cfp.conferences import races
from cfp.config import DATA
from cfp.features import engine
from cfp.features.state import GameArrays, SeasonState, build_state

ND = "Notre Dame"
RATING_SD = 4.0       # sd of a team's true strength around its rating, shared across its remaining games
FINAL_RELEASES = {2026: 6}


@dataclass
class Forced:
    winner: str
    loser: str | None = None          # None for title-game conditions
    margin: int | None = None
    title_game: bool = False          # "X wins/loses its conference title game"
    wins: bool = True


@dataclass
class SimResult:
    st: SeasonState
    teams: pd.DataFrame               # one row per candidate team
    seeds: pd.DataFrame               # P(seed = k) per team
    games: pd.DataFrame               # leverage of each remaining game
    swing: pd.DataFrame               # team x game swings
    title_games: pd.DataFrame         # most likely title-game matchups
    proxy_log: dict[str, int]
    n_sims: int
    seed: int
    notes: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def parse_result(text: str) -> Forced:
    """ "Ohio State over Michigan 27-24", "Ohio State over Michigan", "Georgia loses SEC title game",
    "Georgia wins title game"."""
    t = text.strip()
    m = re.match(r"^(.+?)\s+(wins|loses)\s+(?:the\s+)?(?:\w+\s+)?(?:title|championship)\s+game$", t, re.I)
    if m:
        return Forced(winner=m.group(1).strip(), title_game=True, wins=m.group(2).lower() == "wins")
    m = re.match(r"^(.+?)\s+(?:over|beats|def\.?|defeats)\s+(.+?)(?:\s+(?:by\s+)?(\d+)(?:\s*-\s*(\d+))?)?$", t, re.I)
    if not m:
        raise ValueError(f"cannot parse result: {text!r}")
    margin = None
    if m.group(3) and m.group(4):
        margin = abs(int(m.group(3)) - int(m.group(4)))
    elif m.group(3):
        margin = int(m.group(3))
    if margin == 0:
        raise ValueError("a forced result needs a winner; ties do not exist")
    return Forced(winner=m.group(1).strip(), loser=m.group(2).strip(), margin=margin)


def team_index(st: SeasonState, name: str) -> int:
    low = {n.lower(): i for i, n in enumerate(st.names)}
    key = name.lower().strip()
    alias = {"usc": "usc", "southern california": "usc", "southern cal": "usc", "miami (fl)": "miami"}
    key = alias.get(key, key)
    if key in low:
        return low[key]
    hits = [i for n, i in low.items() if key in n]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"unknown or ambiguous team: {name!r}")


def blended_rating(st: SeasonState, fpi_weight: float = 0.5) -> tuple[np.ndarray, str]:
    """Own rating blended with ESPN FPI as fetched now (mapped onto the own-rating scale). FPI snapshots
    by week do not exist historically, so the blend weight is an assumption, not a fitted value."""
    r = st.rating.copy()
    try:
        f = data.table("espn_fpi")
    except Exception:
        return r, "own rating only (no FPI table)"
    f = f[f["season"] == st.season]
    col = next((c for c in f.columns if c.endswith("fpi_fpi") or c == "fpi_fpi"), None)
    if col is None or f.empty or fpi_weight <= 0:
        return r, "own rating only (FPI not available)"
    v = np.full(st.n, np.nan)
    for t, x in zip(f["team_id"], f[col]):
        if int(t) in st.idx:
            v[st.idx[int(t)]] = x
    ok = ~np.isnan(v)
    if ok.sum() < 100:
        return r, "own rating only (FPI incomplete)"
    b, a = np.polyfit(v[ok], r[:st.n][ok], 1)
    r[:st.n][ok] = (1 - fpi_weight) * r[:st.n][ok] + fpi_weight * (a + b * v[ok])
    return r, f"{1 - fpi_weight:.0%} own rating + {fpi_weight:.0%} ESPN FPI (fetched {str(f['fetched'].iloc[0])[:10]}; corr {np.corrcoef(v[ok], st.rating[:st.n][ok])[0, 1]:.2f})"


def _round_margin(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    m = np.rint(x)
    z = m == 0
    m[z] = np.where(x[z] >= 0, 1, -1)
    return m


def simulate(season: int = 2026, now: pd.Timestamp | None = None, n_sims: int = 10000, seed: int = 2026,
             forced: list[Forced] | None = None, bundle: dict | None = None, chunk: int = 2000,
             fpi_weight: float = 0.5, n_cand: int = 60, st: SeasonState | None = None,
             anchor_rel=None, title_confs: dict | None = None, playoff: bool = True,
             noise: bool = True, legacy_rules: bool = False, final_no: int | None = None) -> SimResult:
    rng = np.random.default_rng(seed)
    R = ratings.load()
    hfa, sigma = R.p.hfa, R.p.sigma
    sig_g = float(np.sqrt(max(sigma ** 2 - RATING_SD ** 2, 1.0)))
    now = now or pd.Timestamp.now(tz="UTC")
    st = st or build_state(season, now)
    bundle = bundle or load_production()
    model, sig_c = bundle["anchor"], bundle["sigma"]["anchor"] if noise else 0.0
    notes: list[str] = []
    n = st.n
    r0, rnote = blended_rating(st, fpi_weight) if season == data.games()["season"].max() else (st.rating.copy(), "own rating only")
    notes.append(f"Game simulation ratings: {rnote}.")

    # --- anchor: the latest real committee ranking of this season
    rels = [r for r in dataset.all_releases() if r.season == season and r.cutoff <= st.cutoff] if anchor_rel is None else [anchor_rel]
    final_no = final_no or FINAL_RELEASES.get(season, max([r.release_no for r in dataset.all_releases() if r.season == season], default=6))
    anchor = None
    if rels:
        a = max(rels, key=lambda r: r.release_no)
        anchor = {"rank": a.rank, "lag": max(final_no - a.release_no, 1), "losses_then": a.raw["losses"].to_numpy()}
        notes.append(f"Committee model anchored on the {str(a.cutoff)[:10]} ranking ({anchor['lag']} releases before Selection Day).")
    else:
        notes.append("No committee ranking yet this season: résumé-only prediction of the final ranking.")

    # --- games
    fut = st.future.take(~st.future.ccg)
    F = len(fut)
    GA = GameArrays.concat(st.played, fut)
    pad = engine.build_pad(GA, n)
    G = len(st.played)
    tg = title_confs if title_confs is not None else (data.overrides().get("title_games", {}).get(season, {}))
    if not tg:
        notes.append("No title-game schedule configured for this season; conference champions are not simulated.")
    end_reg_day = float(fut.day.max()) + 1 if F else st.cutoff_day
    final_day = max(end_reg_day, st.cutoff_day) + 8.0
    if tg:
        d = pd.Timestamp(max(v["date"] for v in tg.values()), tz="UTC") + pd.Timedelta(days=1, hours=8)
        final_day = (d.value - pd.Timestamp(f"{season}-08-01", tz="UTC").value) / 86400e9
    ccg_day = final_day - 1.0
    # Army-Navy and anything else after Selection Day does not count
    counted = fut.day < final_day

    # forced results
    f_margin = np.full(F, np.nan); f_sign = np.zeros(F)
    f_ccg: dict[int, bool] = {}
    for fr in forced or []:
        if fr.title_game:
            f_ccg[team_index(st, fr.winner)] = fr.wins
            continue
        w, l = team_index(st, fr.winner), team_index(st, fr.loser)
        hit = np.where(((fut.gh == w) & (fut.ga == l)) | ((fut.gh == l) & (fut.ga == w)))[0]
        if len(hit) == 0:
            raise ValueError(f"{fr.winner} v {fr.loser} is not a remaining scheduled game")
        g = hit[0]
        sgn = 1.0 if fut.gh[g] == w else -1.0
        f_sign[g] = sgn
        if fr.margin is not None:
            f_margin[g] = sgn * fr.margin

    # candidates: by the current anchor-model score, plus the best Group of Six teams
    f_now = engine.compute(st, st.played, st.margin[None], st.rating[None], hfa=hfa, sigma=sigma)
    T_now = dataset.feature_tensor(st, f_now, is_final=False, anchor=anchor)
    s_now = model.pair.score_tensor(T_now, season)[0]
    cand = np.argsort(-s_now)[:n_cand]
    g6 = np.where(st.g6)[0]
    cand = np.unique(np.concatenate([cand, g6[np.argsort(-s_now[g6])][:10]]))
    m_c = len(cand)
    nd = st.names.index(ND) if ND in st.names else None
    conf_list = list(st.conf)
    games_per_team = np.bincount(np.concatenate([GA.gh, GA.ga]), minlength=n + 1)[:n + 1].astype(float)

    out_rank = np.zeros((n_sims, m_c), np.int16)
    out_seed = np.zeros((n_sims, n), np.int8)
    out_auto = np.zeros((n_sims, n), bool)
    out_champ = np.zeros((n_sims, n), bool)
    out_ccg = np.zeros((n_sims, n), bool)
    out_round = np.zeros((n_sims, n), np.int8)        # 1 first round .. 4 final, 5 champion
    out_newloss = np.zeros((n_sims, n), np.int8)
    out_losses = np.zeros((n_sims, n), np.int8)
    out_homewin = np.zeros((n_sims, F), bool)
    ccg_pairs: dict[str, dict[tuple[int, int], int]] = {c: {} for c in tg}
    proxy_log: dict[str, int] = {}

    for lo in range(0, n_sims, chunk):
        S = min(chunk, n_sims - lo)
        sl = slice(lo, lo + S)
        r_true = r0[None, :] + RATING_SD * rng.standard_normal((S, n + 1))
        mu = r_true[:, fut.gh] - r_true[:, fut.ga] + hfa * (~fut.neutral)[None, :]
        mf = _round_margin(mu + sig_g * rng.standard_normal((S, F)), rng)
        want = f_sign[None, :] != 0
        flip = want & (np.sign(mf) != f_sign[None, :])
        mf = np.where(flip, -mf, mf)                               # forced winner: reflect the margin
        mf = np.where(np.isnan(f_margin)[None, :], mf, np.nan_to_num(f_margin)[None, :])
        margins = np.concatenate([np.broadcast_to(st.margin[None, :], (S, G)), mf], 1)
        out_homewin[sl] = mf > 0
        # ratings the committee model sees: first-order update of the current rating with simulated results
        resid = np.clip(mf, -R.p.cap, R.p.cap) - (r0[fut.gh] - r0[fut.ga] + hfa * (~fut.neutral))[None, :]
        resid = resid * counted[None, :]
        delta = np.zeros((S, n + 1))
        np.add.at(delta, (np.arange(S)[:, None], fut.gh[None, :]), resid)
        np.add.at(delta, (np.arange(S)[:, None], fut.ga[None, :]), -resid)
        r_end = r0[None, :] + delta / (games_per_team[None, :] + R.p.ridge)
        r_end[:, n] = r0[n]
        m_use = np.where(np.concatenate([np.ones(G, bool), counted])[None, :], margins, 0.0)   # 0 = not played yet
        # conference races
        extra = None
        champs: list[dict[str, int]] = [dict() for _ in range(S)]
        if tg:
            f0 = engine.compute(st, GA, m_use, r_end, hfa=hfa, sigma=sigma, now_day=end_reg_day, pad=pad)
            res = races.resolve(st, GA, m_use, f0, rng, has_ccg=set(tg), proxy_log=proxy_log, legacy=legacy_rules)
            e_opp = np.full((S, n), n + 1); e_valid = np.zeros((S, n), bool)
            e_loc = np.zeros((S, n)); e_m = np.zeros((S, n))
            rows = np.arange(S)
            for c, (s1, s2) in res.items():
                campus = tg[c].get("site", "neutral") == "campus"
                mu_c = r_true[rows, s1] - r_true[rows, s2] + (hfa if campus else 0.0)
                mc = _round_margin(mu_c + sig_g * rng.standard_normal(S), rng)
                for t, wins in f_ccg.items():
                    for side, sgn in ((s1, 1.0), (s2, -1.0)):
                        hit = side == t
                        bad = hit & ((np.sign(mc) * sgn > 0) != wins)
                        mc = np.where(bad, -mc, mc)
                e_opp[rows, s1], e_opp[rows, s2] = s2, s1
                e_valid[rows, s1] = True; e_valid[rows, s2] = True
                e_loc[rows, s1], e_loc[rows, s2] = (1 if campus else 0), (-1 if campus else 0)
                e_m[rows, s1], e_m[rows, s2] = mc, -mc
                win = np.where(mc > 0, s1, s2)
                out_champ[lo + rows, win] = True
                out_ccg[lo + rows, s1] = True; out_ccg[lo + rows, s2] = True
                for i in range(S):
                    champs[i][c] = int(win[i])
                    key = (int(s1[i]), int(s2[i]))
                    ccg_pairs[c][key] = ccg_pairs[c].get(key, 0) + 1
            extra = {"opp": e_opp, "valid": e_valid, "loc": e_loc, "m": e_m, "day": ccg_day}
        feats = engine.compute(st, GA, m_use, r_end, hfa=hfa, sigma=sigma, now_day=final_day, final_week=True,
                               pad=pad, extra=extra)
        out_newloss[sl] = np.clip(feats["losses"] - f_now["losses"], 0, 9).astype(np.int8)
        out_losses[sl] = feats["losses"].astype(np.int8)
        T = dataset.feature_tensor(st, feats, is_final=True, anchor=anchor)
        sc = model.score_tensor(T, season, cand)
        pw = engine.pairwise(st, GA, m_use, cand, pad=pad, extra=extra)
        sc_noisy = sc + sig_c * rng.standard_normal(sc.shape)
        b = model.pair.borda(sc_noisy, pw["h2h"], pw["common"], pw["n_common"][None])
        order_c = np.argsort(-b, axis=1, kind="stable")
        pos = np.empty_like(order_c)
        np.put_along_axis(pos, order_c, np.arange(m_c)[None, :].repeat(S, 0), axis=1)
        out_rank[sl] = pos + 1
        raw_all = model.pair.score_tensor(T, season)
        in_c = np.zeros(n, bool); in_c[cand] = True
        rest = np.where(~in_c)[0]
        for i in range(S):
            order = list(cand[order_c[i]]) + list(rest[np.argsort(-raw_all[i, rest])])
            fld = select_field(order, conf_list, champs[i], season, nd)
            for k, t in enumerate(fld.seeds):
                out_seed[lo + i, t] = k + 1
            for t in fld.auto:
                out_auto[lo + i, t] = True
            if playoff and len(fld.seeds) == 12:
                _play(fld.seeds, r_true[i], hfa, sig_g, rng, out_round[lo + i])

    res = _summarise(st, cand, fut, counted, out_rank, out_seed, out_auto, out_champ, out_ccg, out_round,
                     out_newloss, out_homewin, ccg_pairs, proxy_log, n_sims, seed, notes, tg)
    res.raw["losses"] = out_losses
    res.raw["auto"] = out_auto
    res.raw |= {"homewin": out_homewin, "ccg": out_ccg, "counted": counted}
    # how many losses the field carries: context for reading individual teams' odds
    inf = out_seed > 0
    three = (inf & (out_losses >= 3) & ~out_auto).sum(1)
    res.notes.append(f"At-large teams with three or more losses in the simulated field: {three.mean():.1f} on average; "
                     f"at least one in {np.mean(three >= 1):.0%} of simulations. Field teams average "
                     f"{out_losses[inf].mean():.2f} losses.")
    return res


def _play(seeds: list[int], r: np.ndarray, hfa: float, sig: float, rng: np.random.Generator, rounds: np.ndarray) -> None:
    """Play the 12-team bracket once. rounds[t] = furthest round reached (1 first round, 2 quarterfinal,
    3 semifinal, 4 final, 5 champion). First round at the higher seed's campus, the rest neutral."""
    z = rng.standard_normal(11)
    k = 0

    def game(a: int, b: int, home: float) -> int:
        nonlocal k
        w = a if (r[a] - r[b] + home + sig * z[k]) > 0 else b
        k += 1
        return w

    for t in seeds[4:]:
        rounds[t] = 1
    for t in seeds[:4]:
        rounds[t] = 2
    w = {h: game(seeds[h - 1], seeds[v - 1], hfa) for h, v in ((5, 12), (6, 11), (7, 10), (8, 9))}
    for t in w.values():
        rounds[t] = 2
    q = [game(seeds[0], w[8], 0), game(seeds[3], w[5], 0), game(seeds[1], w[7], 0), game(seeds[2], w[6], 0)]
    for t in q:
        rounds[t] = 3
    s = [game(q[0], q[1], 0), game(q[2], q[3], 0)]
    for t in s:
        rounds[t] = 4
    rounds[game(s[0], s[1], 0)] = 5


def _summarise(st, cand, fut, counted, rank, seed, auto, champ, ccg, rounds, newloss, homewin, ccg_pairs,
               proxy_log, n_sims, seed_val, notes, tg) -> SimResult:
    n = st.n
    infield = seed > 0
    rows = []
    # remaining games per team (regular season, before Selection Day)
    rem = np.bincount(np.concatenate([fut.gh[counted], fut.ga[counted]]), minlength=n + 1)[:n]
    for ci, t in enumerate(cand):
        rk = rank[:, ci]
        inf = infield[:, t]
        nl = newloss[:, t]
        wo, ol = nl == 0, nl == 1
        rows.append({
            "team": st.names[t], "conf": st.conf[t], "idx": int(t), "remaining_games": int(rem[t]),
            "p_field": inf.mean(), "p_bye": (seed[:, t] <= 4)[inf].sum() / n_sims,
            "p_host": ((seed[:, t] >= 5) & (seed[:, t] <= 8)).mean(), "p_auto": auto[:, t].mean(),
            "p_title_game": ccg[:, t].mean(), "p_conf_champ": champ[:, t].mean(),
            "mean_rank": rk.mean(), "median_rank": float(np.median(rk)),
            "rank_p10": float(np.quantile(rk, 0.1)), "rank_p90": float(np.quantile(rk, 0.9)),
            "p_top4": (rk <= 4).mean(), "p_top12": (rk <= 12).mean(), "p_top25": (rk <= 25).mean(),
            "p_quarterfinal": (rounds[:, t] >= 2).mean(), "p_semifinal": (rounds[:, t] >= 3).mean(),
            "p_final": (rounds[:, t] >= 4).mean(), "p_champion": (rounds[:, t] >= 5).mean(),
            "p_win_out": wo.mean(), "field_if_win_out": inf[wo].mean() if wo.sum() >= 30 else np.nan,
            "n_win_out": int(wo.sum()),
            "field_if_one_more_loss": inf[ol].mean() if ol.sum() >= 30 else np.nan, "n_one_more_loss": int(ol.sum()),
            "field_if_two_plus": inf[nl >= 2].mean() if (nl >= 2).sum() >= 30 else np.nan,
        })
    teams = pd.DataFrame(rows).sort_values(["p_field", "mean_rank"], ascending=[False, True]).reset_index(drop=True)
    sd = pd.DataFrame({k: [(seed[:, t] == k).mean() for t in cand] for k in range(1, 13)}, index=[st.names[t] for t in cand])
    sd = sd.loc[teams["team"]]

    # leverage: how much each remaining game moves each team's chance of making the field
    watch = teams[(teams.p_field > 0.01) & (teams.p_field < 0.99)]["idx"].to_numpy()
    allc = teams[teams.p_field > 0.005]["idx"].to_numpy()
    grow, srow = [], []
    for g in np.where(counted)[0]:
        hw = homewin[:, g]
        nh = hw.sum()
        if nh < 50 or (n_sims - nh) < 50:
            continue
        ph = infield[hw][:, allc].mean(0); pa = infield[~hw][:, allc].mean(0)
        d = ph - pa
        h, a = int(fut.gh[g]), int(fut.ga[g])
        hn = st.names[h] if h < n else "FCS opponent"; an = st.names[a] if a < n else "FCS opponent"
        day = pd.Timestamp(f"{st.season}-08-01") + pd.Timedelta(days=float(fut.day[g]) - 0.34)
        top = np.argsort(-np.abs(d))[:3]
        p_h = float(hw.mean())
        grow.append({"game": f"{an} at {hn}" if not fut.neutral[g] else f"{an} v {hn}", "date": str(day.date()),
                     "week": int(fut.week[g]), "p_home_win": p_h, "leverage": float(np.abs(d).sum()),
                     "expected_impact": float(np.abs(d).sum() * 4 * p_h * (1 - p_h)),
                     "most_affected": "; ".join(f"{st.names[allc[i]]} {d[i]:+.0%} if {hn} wins" for i in top if abs(d[i]) > 0.01)})
        for i in np.where(np.abs(d) > 0.02)[0]:
            if allc[i] in watch:
                srow.append({"team": st.names[allc[i]], "game": grow[-1]["game"], "date": grow[-1]["date"],
                             "own_game": allc[i] in (h, a), "field_if_home_wins": ph[i], "field_if_away_wins": pa[i],
                             "swing": d[i]})
    games = pd.DataFrame(grow).sort_values("expected_impact", ascending=False).reset_index(drop=True) if grow else pd.DataFrame()
    swing = pd.DataFrame(srow)
    if len(swing):
        swing = swing.reindex(swing["swing"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    trow = []
    for c, d in ccg_pairs.items():
        for (a, b), cnt in sorted(d.items(), key=lambda kv: -kv[1])[:3]:
            trow.append({"conf": c, "matchup": f"{st.names[a]} v {st.names[b]}", "p": cnt / n_sims,
                         "site": tg[c].get("venue", f"at {st.names[a]}" if tg[c].get("site") == "campus" else "")})
    return SimResult(st=st, teams=teams, seeds=sd, games=games, swing=swing, title_games=pd.DataFrame(trow),
                     proxy_log=proxy_log, n_sims=n_sims, seed=seed_val, notes=notes,
                     raw={"cand": cand, "rank": rank, "seed": seed, "champ": champ, "rounds": rounds})
