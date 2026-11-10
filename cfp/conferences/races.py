"""Conference standings and title-game participants from a SeasonState, for one or many simulated seasons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfp import data, ratings
from cfp.conferences import tiebreak as tb
from cfp.conferences.tiebreak import ConfContext
from cfp.features import engine
from cfp.features.state import GameArrays, SeasonState, build_state

NEUTRAL_SITE = {"sec", "big10", "big12", "acc", "midam"}


def conf_members(st: SeasonState) -> dict[str, np.ndarray]:
    return {c: np.where(st.conf == c)[0] for c in sorted(set(st.conf)) if c != "ind"}


def win_tensors(st: SeasonState, ga: GameArrays, margin: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per conference: (team indices, W (S,k,k) league wins, capped league margin per game (S,k))."""
    S = margin.shape[0]
    out = {}
    for c, T in conf_members(st).items():
        loc = {int(t): i for i, t in enumerate(T)}
        k = len(T)
        W = np.zeros((S, k, k), np.float32); M = np.zeros((S, k), np.float32); N = np.zeros(k)
        sel = np.where(ga.conf_game & ~ga.ccg & np.isin(ga.gh, T) & np.isin(ga.ga, T))[0]
        for g in sel:
            i, j = loc[int(ga.gh[g])], loc[int(ga.ga[g])]
            m = margin[:, g]
            hw = m > 0
            W[:, i, j] += hw; W[:, j, i] += ~hw
            mc = np.clip(m, -24, 24)
            M[:, i] += mc; M[:, j] -= mc; N[i] += 1; N[j] += 1
        out[c] = (T, W, M / np.maximum(N, 1)[None])
    return out


def resolve(st: SeasonState, ga: GameArrays, margin: np.ndarray, feats: dict[str, np.ndarray],
            rng: np.random.Generator, has_ccg: set[str] | None = None, divisions: bool | None = None,
            proxy_log: dict[str, int] | None = None, legacy: bool = False) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Title-game seeds per conference: {code: (seed1 (S,), seed2 (S,))} as global team indices."""
    S = margin.shape[0]
    proxy_log = proxy_log if proxy_log is not None else {}
    fcs_w = feats["fcs_games"] - feats["fcs_losses"]
    ow = feats["wins"] - np.maximum(fcs_w - 1, 0)
    res = {}
    for c, (T, W, CM) in win_tensors(st, ga, margin).items():
        if has_ccg is not None and c not in has_ccg:
            continue
        # legacy=True: seasons before 2026, when leagues had other rules: generic ladder, divisions as listed
        rules = {"steps": tb.GENERIC} if legacy else tb.RULES.get(c, {"steps": tb.GENERIC})
        div = [st.division[t] for t in T]
        two = len({d for d in div if d}) == 2
        use_div = two if (legacy or divisions is None) else (divisions and two)
        cw, cl = W.sum(2), W.sum(1)
        pct = cw / np.maximum(cw + cl, 1)
        s1 = np.zeros(S, int); s2 = np.zeros(S, int)
        srt = np.sort(pct, axis=1)[:, ::-1]
        idx = np.argsort(-pct, axis=1, kind="stable")
        clear = (srt[:, 1] > srt[:, 2] + 1e-9) & ((srt[:, 0] > srt[:, 1] + 1e-9) | (c in NEUTRAL_SITE))
        if use_div or rules.get("acc_groups"):
            clear[:] = False
        s1[clear], s2[clear] = idx[clear, 0], idx[clear, 1]
        for s in np.where(~clear)[0]:
            ctx = ConfContext(code=c, wins=W[s], division=div if use_div else [""] * len(T),
                              rating=feats["rating"][s, T], sor=feats["wae"][s, T], overall_w=ow[s, T],
                              overall_pct=feats["win_pct"][s, T], conf_margin=CM[s], rng=rng, proxy_log=proxy_log)
            s1[s], s2[s] = tb.title_game(ctx, rules, has_divisions=use_div)
        res[c] = (T[s1], T[s2])
    return res


# ----------------------------------------------------------------------------- history / current
def actual_title_game(season: int, conf: str) -> list[str]:
    g = data.games()
    x = g[(g["season"] == season) & g["is_conf_champ"] & (g["home_conf"] == conf) & (g["away_conf"] == conf)]
    return [] if x.empty else [x["home_team"].iloc[0], x["away_team"].iloc[0]]


def historical_title_game(season: int, conf: str, generic_divisions: bool = True) -> list[str]:
    """Apply the tiebreaker engine to the completed regular season and return the two participants."""
    g = data.games()
    ccg = g[(g["season"] == season) & g["is_conf_champ"]]
    cutoff = ccg["start_date"].min() - pd.Timedelta(hours=1)
    st = build_state(season, cutoff)
    R = ratings.load()
    f = engine.compute(st, st.played, st.margin[None], st.rating[None], hfa=R.p.hfa, sigma=R.p.sigma)
    has_div = len({d for d, c in zip(st.division, st.conf) if c == conf and d}) == 2
    res = resolve(st, st.played, st.margin[None], f, np.random.default_rng(0), has_ccg={conf},
                  divisions=has_div if generic_divisions else None)
    a, b = res[conf]
    return [st.names[int(a[0])], st.names[int(b[0])]]


def standings_table(st: SeasonState, ga: GameArrays, margin: np.ndarray) -> pd.DataFrame:
    rows = []
    for c, (T, W, _) in win_tensors(st, ga, margin[None] if margin.ndim == 1 else margin).items():
        cw, cl = W[0].sum(1), W[0].sum(0)
        for i, t in enumerate(T):
            rows.append({"conf": c, "team": st.names[t], "division": st.division[t], "conf_w": int(cw[i]), "conf_l": int(cl[i])})
    return pd.DataFrame(rows).sort_values(["conf", "conf_w", "conf_l"], ascending=[True, False, True])


if __name__ == "__main__":
    rows = []
    for season in range(2014, 2026):
        g = data.games()
        confs = sorted(set(g[(g.season == season) & g.is_conf_champ & (g.home_conf == g.away_conf)]["home_conf"]))
        for c in confs:
            act, pred = actual_title_game(season, c), historical_title_game(season, c)
            rows.append({"season": season, "conf": c, "ok": set(act) == set(pred), "actual": " v ".join(sorted(act)), "engine": " v ".join(sorted(pred))})
    d = pd.DataFrame(rows)
    print(d.groupby("conf")["ok"].agg(["sum", "size"]).to_string())
    print(d[~d.ok].to_string())
