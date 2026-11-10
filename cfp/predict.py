"""Next-ranking prediction with uncertainty and plain-English explanations, plus résumé sheets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from cfp import data, ratings
from cfp.committee import dataset, uncertainty
from cfp.committee.backtest import champions_of
from cfp.committee.dataset import Release
from cfp.committee.field import Field, select_field
from cfp.committee.train import load_production
from cfp.config import DATA
from cfp.conferences import races
from cfp.features import engine

SELECTION_DAY = {2026: "2026-12-06"}
RELEASE_DATES = {2026: ["2026-11-03", "2026-11-10", "2026-11-17", "2026-11-24", "2026-12-01", "2026-12-06"]}

GROUPS = {
    "last week's committee position": ["prev", "prev_lost"],
    "AP and Coaches polls": ["ap", "coaches", "ap_first", "coaches_first", "ap_lost_first"],
    "losses and record": ["losses", "losses_sq", "wins", "win_pct", "no_loss"],
    "team strength and dominance": ["rating", "margin_cap", "margin_loss_cap", "game_control", "postgame_wp"],
    "quality of wins": ["win_credit", "best_win", "best_win2", "best_win3", "w_ranked", "w_top10", "w_ranked_g",
                        "w_top10_g", "w500", "w_power", "road_w_ranked", "w_road"],
    "quality of losses": ["loss_penalty", "worst_loss", "loss_unranked", "loss_sub500", "l_ranked", "l_top10",
                          "l_ranked_g", "l500", "l_power", "fcs_losses", "l_home"],
    "schedule strength and strength of record": ["sos", "sos_top4", "wae", "sor_z", "record_strength"],
    "recent form and this week's result": ["last3_value", "last3_wins", "trend", "days_since_loss", "recent_losses",
                                           "played_week", "won_week", "lost_week", "week_value", "week_opp",
                                           "week_margin", "idle_final"],
    "conference strength and title game": ["champion", "ccg_loss", "ccg_played", "conf_strength", "conf_sos_rel"],
    "power / Group of Six status": ["is_power", "is_g6", "is_indep", "g6_wae"],
    "preseason reputation": ["preseason"],
}


def availability() -> tuple[dict, dict]:
    p = DATA / "availability.yaml"
    if not p.exists():
        return {}, {}
    d = yaml.safe_load(p.read_text()) or {}
    return d.get("teams") or {}, d.get("assumed_effect") or {}


@dataclass
class Prediction:
    rel: Release
    table: pd.DataFrame            # predicted ranking with uncertainty
    dist: pd.DataFrame             # P(rank = k) per team
    contrib: pd.DataFrame          # grouped contributions per team (relative to the predicted top-25 average)
    order: np.ndarray
    release_no: int
    release_date: str
    is_final: bool
    explanations: dict[str, dict]
    pairs: list[str]
    field: Field | None
    leaders: dict[str, int]
    notes: list[str]


def next_release(season: int, now: pd.Timestamp) -> tuple[int, str, bool]:
    rel = data.releases()
    done = int((rel["season"] == season).sum())
    dates = RELEASE_DATES.get(season, [])
    date = dates[done] if done < len(dates) else str(now.date())
    return done + 1, date, date == SELECTION_DAY.get(season)


def game_log(st, R: ratings.Ratings | None = None) -> dict[int, list[dict]]:
    """Per team: its played games with opponent, venue, score and venue-adjusted opponent quality."""
    R = R or ratings.load()
    ga, n = st.played, st.n
    w = np.zeros(n + 1); l = np.zeros(n + 1)
    hw = st.margin > 0
    np.add.at(w, ga.gh, hw); np.add.at(l, ga.gh, ~hw); np.add.at(w, ga.ga, ~hw); np.add.at(l, ga.ga, hw)
    order = np.argsort(-st.rating[:n]); rrank = np.empty(n, int); rrank[order] = np.arange(1, n + 1)
    out: dict[int, list[dict]] = {i: [] for i in range(n)}
    for g in np.argsort(ga.day):
        for t, o, sgn, pts, opts, rk in ((ga.gh[g], ga.ga[g], 1, st.home_pts[g], st.away_pts[g], ga.rank_a[g]),
                                         (ga.ga[g], ga.gh[g], -1, st.away_pts[g], st.home_pts[g], ga.rank_h[g])):
            if t >= n:
                continue
            loc = 0 if ga.neutral[g] else sgn
            out[int(t)].append({
                "opp": st.names[o] if o < n else "FCS opponent", "opp_idx": int(o), "loc": {1: "home", -1: "road", 0: "neutral"}[loc],
                "won": bool(sgn * st.margin[g] > 0), "score": f"{int(pts)}-{int(opts)}",
                "opp_record": f"{int(w[o])}-{int(l[o])}" if o < n else "", "opp_rating_rank": int(rrank[o]) if o < n else None,
                "opp_rank_then": None if np.isnan(rk) else int(rk),
                "value": float(st.rating[o] - R.p.hfa * loc), "title_game": bool(ga.ccg[g]), "day": float(ga.day[g])})
    return out


def _facts(st, raw: pd.Series, log: list[dict], name: str) -> dict[str, str]:
    wins = sorted([g for g in log if g["won"]], key=lambda g: -g["value"])
    losses = sorted([g for g in log if not g["won"]], key=lambda g: g["value"])

    def gs(g: dict) -> str:
        at = {"home": "v", "road": "at", "neutral": "v (neutral)"}[g["loc"]]
        rk = f"No. {g['opp_rank_then']} " if g["opp_rank_then"] else ""
        return f"{at} {rk}{g['opp']} ({g['opp_record']}) {g['score']}"

    f = {
        "losses and record": f"{int(raw.wins)}-{int(raw.losses)}",
        "quality of wins": ("best wins: " + "; ".join(gs(g) for g in wins[:2]) if wins else "no wins") +
                           f"; {int(raw.w500)}-{int(raw.l500)} v .500+ teams, {int(raw.w_ranked)}-{int(raw.l_ranked)} v the current top 25 by résumé",
        "quality of losses": ("losses: " + "; ".join(gs(g) for g in losses[:3])) if losses else "no losses",
        "team strength and dominance": f"power rating {raw.rating:+.1f} points v an average FBS team, average capped margin {raw.margin_cap:+.1f}",
        "schedule strength and strength of record": f"{raw.wae:+.2f} wins v what an average top-10 team would have against this schedule",
        "recent form and this week's result": ("lost this week" if raw.lost_week else "won this week" if raw.won_week else "idle this week") +
                                              f"; {int(raw.last3_wins)} wins in the last three",
        "conference strength and title game": f"league average rating {raw.conf_strength:+.1f}",
        "last week's committee position": "",
        "AP and Coaches polls": "",
        "power / Group of Six status": "Group of Six" if raw.is_g6 else "independent" if raw.is_indep else "power conference",
        "preseason reputation": "",
    }
    return f


def predict_next(season: int = 2026, now: pd.Timestamp | None = None, bundle: dict | None = None,
                 n_draws: int = 4000, seed: int = 0) -> Prediction:
    now = now or pd.Timestamp.now(tz="UTC")
    bundle = bundle or load_production()
    model, sigma = bundle["primary"], bundle["sigma"]["primary"]
    release_no, date, is_final = next_release(season, now)
    av, eff = availability()
    rel = dataset.build_release(season, now, release_no, is_final, with_labels=False, availability=av)
    st = rel.st
    notes = []
    score = model.score(rel)
    adj = eff.get("qb_out", 0.0) * rel.X["qb_out"].to_numpy() + eff.get("qb_returning", 0.0) * rel.X["qb_back"].to_numpy()
    if np.any(adj != 0):
        notes.append("Availability flags applied (assumed effect, not learned): " +
                     ", ".join(f"{st.names[i]} {adj[i]:+.1f}" for i in np.where(adj != 0)[0]))
    score = score + adj
    order = model.pair.order(rel, extra_score=score)
    P = uncertainty.rank_distribution(model, rel, sigma, n=n_draws, seed=seed, max_rank=30, score=score)
    cpos = {int(t): i for i, t in enumerate(rel.cand)}
    ap_date = data.rankings()
    ap_date = ap_date[(ap_date.season == season) & (ap_date.poll == "ap") & (ap_date.poll_date < st.cutoff)]["poll_date"].max()
    notes.append(f"Uses games through {str(st.cutoff)[:16]} UTC, the AP/Coaches polls dated {str(ap_date)[:10]}, "
                 f"and the committee's previous ranking{'' if st.has_prev_cfp else ' (none yet)'}.")
    rows = []
    for k, t in enumerate(order[:30]):
        p = P[cpos[int(t)]] if int(t) in cpos else None
        ranks = np.arange(1, 32)
        cdf = np.cumsum(p) if p is not None else None
        lo = int(ranks[np.searchsorted(cdf, 0.1)]) if p is not None else None
        hi = int(ranks[min(np.searchsorted(cdf, 0.9), 30)]) if p is not None else None
        raw = rel.raw.iloc[t]
        rows.append({"rank": k + 1, "team": st.names[t], "record": f"{int(raw.wins)}-{int(raw.losses)}", "conf": st.conf[t],
                     "prev": None if np.isnan(st.prev_cfp[t]) else int(st.prev_cfp[t]),
                     "ap": None if np.isnan(st.ap[t]) else int(st.ap[t]),
                     "p_exact": float(p[k]) if p is not None and k < 30 else np.nan, "low": lo, "high": hi,
                     "p_top4": float(p[:4].sum()) if p is not None else 0.0,
                     "p_top12": float(p[:12].sum()) if p is not None else 0.0,
                     "p_top25": float(p[:25].sum()) if p is not None else 0.0, "idx": int(t)})
    table = pd.DataFrame(rows)
    dist = pd.DataFrame(P[[cpos[int(t)] for t in order[:30]]], index=[st.names[t] for t in order[:30]],
                        columns=[str(i) for i in range(1, 31)] + ["31+"])

    # explanations from the interpretable (pairwise) half of the ensemble
    C = model.pair.contributions(rel)
    G = pd.DataFrame({g: C[[c for c in cols if c in C.columns]].sum(1) for g, cols in GROUPS.items()})
    top25 = [st.names[t] for t in order[:25]]
    Grel = G - G.loc[top25].mean()
    logs = game_log(st)
    expl: dict[str, dict] = {}
    for k, t in enumerate(order[:25]):
        nm = st.names[t]
        facts = _facts(st, rel.raw.iloc[t], logs[int(t)], nm)
        s = Grel.loc[nm].sort_values()
        up = [(g, float(v), facts.get(g, "")) for g, v in s[::-1].items() if v > 0.15][:3]
        down = [(g, float(v), facts.get(g, "")) for g, v in s.items() if v < -0.15][:3]
        expl[nm] = {"rank": k + 1, "up": up, "down": down, "facts": facts}
    pairs = []
    for k in range(24):
        a, b = order[k], order[k + 1]
        na, nb = st.names[a], st.names[b]
        d = (G.loc[na] - G.loc[nb]).sort_values(ascending=False)
        why = [g for g, v in d.items() if v > 0.1][:2]
        but = [g for g, v in d[::-1].items() if v < -0.1][:1]
        txt = f"{k + 1}. {na} over {k + 2}. {nb}: " + (" and ".join(why) if why else "a marginal overall edge")
        if but:
            txt += f"; {nb} is better on {but[0]}"
        if int(a) in cpos and int(b) in cpos:
            h = rel.pw["h2h"][cpos[int(a)], cpos[int(b)]]
            if h > 0:
                txt += f". {na} won the head-to-head"
            elif h < 0:
                txt += f". {nb} won the head-to-head but is behind anyway"
            pa = P[cpos[int(a)]]; pb = P[cpos[int(b)]]
        gap = float(score[a] - score[b])
        txt += f" (gap {gap:.2f}: {'close call' if gap < 0.6 else 'clear' if gap > 1.8 else 'moderate'})."
        pairs.append(txt)

    # "if the season ended today": current league leaders stand in as champions
    R = ratings.load()
    f = engine.compute(st, st.played, st.margin[None], st.rating[None], hfa=R.p.hfa, sigma=R.p.sigma)
    lead = races.resolve(st, st.played, st.margin[None], f, np.random.default_rng(0))
    leaders = {c: int(s1[0]) for c, (s1, s2) in lead.items()}
    champs = champions_of(rel) or leaders
    nd = st.names.index("Notre Dame") if "Notre Dame" in st.names else None
    fld = select_field(list(map(int, order)), list(st.conf), {**leaders, **champions_of(rel)}, season, nd)
    return Prediction(rel, table, dist, Grel, order, release_no, date, is_final, expl, pairs, fld, leaders, notes)


# bowl paths: the top three seeds choose in order. Their choices are not knowable in advance; this is an
# ASSUMPTION based on geography by conference, documented in the README.
QF_PREF = {"sec": ["Peach", "Cotton", "Rose", "Fiesta"], "acc": ["Peach", "Cotton", "Rose", "Fiesta"],
           "big10": ["Rose", "Peach", "Cotton", "Fiesta"], "big12": ["Cotton", "Fiesta", "Rose", "Peach"],
           "pac12": ["Fiesta", "Rose", "Cotton", "Peach"], "mwest": ["Fiesta", "Rose", "Cotton", "Peach"]}
QF_DATE = {"Fiesta": "Dec 30", "Cotton": "Jan 1", "Peach": "Jan 1", "Rose": "Jan 1"}
WEST = {"Utah", "BYU", "UCLA", "USC", "Oregon", "Washington", "Arizona", "Arizona State", "Colorado"}


def bowl_paths(seeds: list[int], st) -> dict[int, str]:
    """Seed -> quarterfinal bowl. Seeds 1-3 pick by assumed geographic preference; seed 4 gets what is left."""
    left = ["Peach", "Rose", "Cotton", "Fiesta"]
    out = {}
    for s in (1, 2, 3):
        t = seeds[s - 1]
        pref = QF_PREF.get(st.conf[t], ["Cotton", "Peach", "Rose", "Fiesta"])
        if st.names[t] in WEST:
            pref = ["Rose", "Fiesta", "Cotton", "Peach"] if st.conf[t] == "big10" else ["Fiesta", "Rose", "Cotton", "Peach"]
        pick = next(b for b in pref if b in left)
        out[s] = pick; left.remove(pick)
    out[4] = left[0]
    return out


def bracket_text(fld: Field, st, title: str = "") -> str:
    s = fld.seeds
    if len(s) != 12:
        return ""
    nm = lambda k: f"({k}) {st.names[s[k - 1]]}"   # noqa: E731
    bowls = bowl_paths(s, st)
    L = [title] if title else []
    L.append("First round, Dec 18-19, on campus:")
    for h, v in ((5, 12), (6, 11), (7, 10), (8, 9)):
        L.append(f"  {nm(v)} at {nm(h)}")
    L.append("Quarterfinals (bowl choice by seeds 1-3 is an assumption):")
    for top, h, v in ((1, 8, 9), (2, 7, 10), (3, 6, 11), (4, 5, 12)):
        L.append(f"  {bowls[top]} Bowl, {QF_DATE[bowls[top]]}: {nm(top)} v winner of {v}/{h}")
    L.append("Semifinals: Orange Bowl Jan 14, Sugar Bowl Jan 15 (1/8/9 side v 4/5/12 side; 2/7/10 side v 3/6/11 side). "
             "The No. 1 seed chooses which semifinal its side plays in.")
    L.append("Championship: Jan 25, 2027, Allegiant Stadium, Las Vegas.")
    if fld.auto:
        L.append("Automatic bids: " + "; ".join(f"{st.names[t]} ({why})" for t, why in fld.auto.items()))
    if fld.bumped:
        L.append("Ranked in the top 12 but out: " + ", ".join(st.names[t] for t in fld.bumped))
    return "\n".join(L)


def team_sheet(name: str, season: int = 2026, now: pd.Timestamp | None = None, pred: Prediction | None = None,
               sim=None) -> str:
    """A résumé sheet like the committee would see."""
    from cfp.simulate.season import team_index
    pred = pred or predict_next(season, now)
    rel, st = pred.rel, pred.rel.st
    t = team_index(st, name)
    raw = rel.raw.iloc[t]
    n = st.n
    rk = lambda col, asc=False: int(rel.raw[col].rank(ascending=asc, method="min").iloc[t])   # noqa: E731
    pr = int(np.where(pred.order == t)[0][0]) + 1
    L = [f"# {st.names[t]} — résumé as of {str(st.cutoff)[:10]}", ""]
    L.append(f"Record {int(raw.wins)}-{int(raw.losses)} ({st.conf[t]} {int(raw.conf_wins)}-{int(raw.conf_losses)}) | "
             f"Predicted committee rank {pr} | Last committee rank {'NR' if np.isnan(st.prev_cfp[t]) else int(st.prev_cfp[t])} | "
             f"AP {'NR' if np.isnan(st.ap[t]) else int(st.ap[t])} | Coaches {'NR' if np.isnan(st.coaches[t]) else int(st.coaches[t])} | "
             f"Preseason AP {'NR' if np.isnan(st.preseason_ap[t]) else int(st.preseason_ap[t])}")
    L += ["", "| Measure | Value | FBS rank |", "|---|---|---|",
          f"| Power rating (points v average) | {raw.rating:+.1f} | {rk('rating')} |",
          f"| Strength of record (wins v avg top-10 team's expectation) | {raw.wae:+.2f} | {rk('wae')} |",
          f"| Schedule strength played | {raw.sos:.3f} | {rk('sos')} |",
          f"| Schedule strength, four toughest games (2025 emphasis) | {raw.sos_top4:.3f} | {rk('sos_top4')} |",
          f"| Record strength (reconstruction) | {raw.record_strength:+.2f} | {rk('record_strength')} |",
          f"| League schedule v league peers (2026 emphasis) | {raw.conf_sos_rel:+.2f} | {rk('conf_sos_rel')} |",
          f"| Conference strength (mean rating) | {raw.conf_strength:+.1f} | |",
          f"| Average capped margin (24) | {raw.margin_cap:+.1f} | {rk('margin_cap')} |",
          f"| Game control (quarter-score proxy) | {raw.game_control:.2f} | {rk('game_control')} |", "",
          f"- v teams ranked at kickoff: {int(raw.w_ranked_g)}-{int(raw.l_ranked_g)} (top 10: {int(raw.w_top10_g)} wins)",
          f"- v current top 25 by résumé proxy: {int(raw.w_ranked)}-{int(raw.l_ranked)}; v top 10: {int(raw.w_top10)}-{int(raw.l_top10)}",
          f"- v .500-or-better teams: {int(raw.w500)}-{int(raw.l500)}; v power-conference teams: {int(raw.w_power)}-{int(raw.l_power)}",
          f"- Home {int(raw.w_home)}-{int(raw.l_home)}, road {int(raw.w_road)}-{int(raw.l_road)}, neutral {int(raw.w_neutral)}-{int(raw.l_neutral)}; "
          f"road wins over current top-25: {int(raw.road_w_ranked)}",
          f"- Losses to teams outside the current top 25: {int(raw.loss_unranked)}; to sub-.500 teams: {int(raw.loss_sub500)}; FCS games: {int(raw.fcs_games)}",
          "", "## Games", "", "| Result | Opponent | Site | Score | Opp record | Opp rating rank | Opp rank then |", "|---|---|---|---|---|---|---|"]
    for g in game_log(st)[t]:
        L.append(f"| {'W' if g['won'] else 'L'} | {g['opp']}{' (title game)' if g['title_game'] else ''} | {g['loc']} | {g['score']} | "
                 f"{g['opp_record']} | {g['opp_rating_rank'] or ''} | {g['opp_rank_then'] or ''} |")
    R = ratings.load()
    fut = st.future
    rows = [g for g in range(len(fut)) if t in (fut.gh[g], fut.ga[g])]
    if rows:
        from scipy.stats import norm
        L += ["", "## Remaining", "", "| Date | Opponent | Site | Win probability |", "|---|---|---|---|"]
        for g in rows:
            home = fut.gh[g] == t
            o = fut.ga[g] if home else fut.gh[g]
            mu = (st.rating[t] - st.rating[o]) + (0 if fut.neutral[g] else (R.p.hfa if home else -R.p.hfa))
            day = pd.Timestamp(f"{st.season}-08-01") + pd.Timedelta(days=float(fut.day[g]) - 0.34)
            L.append(f"| {day.date()} | {st.names[o] if o < n else 'FCS opponent'} | {'neutral' if fut.neutral[g] else 'home' if home else 'road'} | {norm.cdf(mu / R.p.sigma):.0%} |")
    if st.names[t] in pred.explanations:
        e = pred.explanations[st.names[t]]
        L += ["", "## Why the model has it here", ""]
        L += [f"- Up: {g} ({v:+.2f}){': ' + f if f else ''}" for g, v, f in e["up"]]
        L += [f"- Down: {g} ({v:+.2f}){': ' + f if f else ''}" for g, v, f in e["down"]]
    if sim is not None:
        row = sim.teams[sim.teams.team == st.names[t]]
        if len(row):
            r = row.iloc[0]
            L += ["", "## Simulated outlook", "",
                  f"Make the field {r.p_field:.0%} | bye {r.p_bye:.0%} | host a first-round game {r.p_host:.0%} | win conference {r.p_conf_champ:.0%} | national title {r.p_champion:.1%}",
                  f"Wins out ({r.p_win_out:.0%} of simulations): in {r.field_if_win_out:.0%}. One more loss: in {r.field_if_one_more_loss:.0%}."]
            sw = sim.swing[sim.swing.team == st.names[t]].head(6)
            if len(sw):
                L += ["", "Games that swing its odds most:"]
                L += [f"- {x.game} ({x.date}): {x.field_if_home_wins:.0%} if the home team wins, {x.field_if_away_wins:.0%} if not" for x in sw.itertuples()]
    return "\n".join(L)
