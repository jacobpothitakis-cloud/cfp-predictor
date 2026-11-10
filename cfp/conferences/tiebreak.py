"""Layer 4: conference races. Who plays in each conference championship game.

Each league is a list of tiebreaker steps applied in its published order. A step looks at the group of
tied teams and returns one number per team (higher is better) or None when it does not apply. The
ordering routine follows the common rule that once a step separates the group, the teams that came out
on top restart the procedure among themselves.

Steps that depend on something not publicly computable (SportSource Analytics ratings, the CFP ranking on a
future date, a draw) use a proxy; every time a proxy decides a tie it is counted in `ctx.proxy_log`.

Sources and verification status per league are in RULES below and in the README.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class ConfContext:
    """One conference in one (real or simulated) season. Indices are local 0..k-1."""
    code: str
    wins: np.ndarray                 # (k, k) wins[i, j] = conference wins of i over j (title game excluded)
    division: list[str]              # per team ("" if none)
    rating: np.ndarray               # proxy for a SportSource-style team rating
    sor: np.ndarray                  # proxy for a résumé ranking (strength of record)
    overall_w: np.ndarray            # overall wins, with at most one FCS win counted
    overall_pct: np.ndarray
    conf_margin: np.ndarray          # capped scoring margin per conference game
    rng: np.random.Generator
    proxy_log: dict[str, int] = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.rating)

    @property
    def cw(self) -> np.ndarray:
        return self.wins.sum(1)

    @property
    def cl(self) -> np.ndarray:
        return self.wins.sum(0)

    @property
    def pct(self) -> np.ndarray:
        g = self.cw + self.cl
        return np.where(g > 0, self.cw / np.maximum(g, 1), 0.0)

    def log(self, step: str) -> None:
        key = f"{self.code}:{step}"
        self.proxy_log[key] = self.proxy_log.get(key, 0) + 1


Step = Callable[[ConfContext, list[int]], np.ndarray | None]


def _played(ctx: ConfContext, i: int, j: int) -> bool:
    return (ctx.wins[i, j] + ctx.wins[j, i]) > 0


def h2h(ctx: ConfContext, grp: list[int]) -> np.ndarray | None:
    """Two teams: the head-to-head winner. Three or more: record among the tied teams if they all played
    each other; otherwise a team that beat every other tied team is first (and nothing else is decided)."""
    sub = ctx.wins[np.ix_(grp, grp)]
    if len(grp) == 2:
        return None if sub[0, 1] == sub[1, 0] else np.array([sub[0, 1], sub[1, 0]], float)
    full = all(_played(ctx, a, b) for x, a in enumerate(grp) for b in grp[x + 1:])
    if full:
        w, l = sub.sum(1), sub.sum(0)
        return w / np.maximum(w + l, 1)
    beat_all = np.array([all(ctx.wins[a, b] > ctx.wins[b, a] for b in grp if b != a) for a in grp])
    return beat_all.astype(float) if beat_all.any() else None


def h2h_acc(ctx: ConfContext, grp: list[int]) -> np.ndarray | None:
    """ACC 2026: as h2h, plus in a non-round-robin tie a team that lost to every other tied team is last."""
    v = h2h(ctx, grp)
    if len(grp) <= 2:
        return v
    full = all(_played(ctx, a, b) for x, a in enumerate(grp) for b in grp[x + 1:])
    if full:
        return v
    lost_all = np.array([all(ctx.wins[b, a] > ctx.wins[a, b] for b in grp if b != a) for a in grp])
    if v is None and not lost_all.any():
        return None
    return (v if v is not None else np.zeros(len(grp))) - lost_all


def common(ctx: ConfContext, grp: list[int]) -> np.ndarray | None:
    """Win percentage against conference opponents that every tied team has played."""
    others = [o for o in range(ctx.k) if o not in grp and all(_played(ctx, a, o) for a in grp)]
    if not others:
        return None
    w = ctx.wins[np.ix_(grp, others)].sum(1); l = ctx.wins[np.ix_(others, grp)].sum(0)
    return w / np.maximum(w + l, 1)


def descending(ctx: ConfContext, grp: list[int]) -> np.ndarray | None:
    """Record against the highest-placed common opponent(s), working down the standings. Opponents tied
    with each other in the standings are taken together as a group."""
    pct = ctx.pct
    others = [o for o in range(ctx.k) if o not in grp and all(_played(ctx, a, o) for a in grp)]
    for level in sorted({round(pct[o], 9) for o in others}, reverse=True):
        tier = [o for o in others if round(pct[o], 9) == level]
        w = ctx.wins[np.ix_(grp, tier)].sum(1); l = ctx.wins[np.ix_(tier, grp)].sum(0)
        v = w / np.maximum(w + l, 1)
        if not np.allclose(v, v[0]):
            return v
    return None


def opp_winpct(ctx: ConfContext, grp: list[int]) -> np.ndarray | None:
    """Cumulative conference winning percentage of each team's conference opponents."""
    cw, cl = ctx.cw, ctx.cl
    out = []
    for a in grp:
        opp = [o for o in range(ctx.k) if _played(ctx, a, o)]
        n = np.array([ctx.wins[a, o] + ctx.wins[o, a] for o in opp])   # weight repeat opponents
        out.append((cw[opp] * n).sum() / max(((cw[opp] + cl[opp]) * n).sum(), 1))
    return np.array(out)


def _proxy(name: str, attr: str) -> Step:
    def step(ctx: ConfContext, grp: list[int]) -> np.ndarray | None:
        v = getattr(ctx, attr)[grp].astype(float)
        if np.allclose(v, v[0]):
            return None
        ctx.log(name)
        return v
    step.__name__ = name
    return step


total_wins: Step = lambda ctx, grp: ctx.overall_w[grp].astype(float)          # noqa: E731
overall_pct: Step = lambda ctx, grp: ctx.overall_pct[grp].astype(float)       # noqa: E731
rating_proxy = _proxy("SportSource rating -> own team rating", "rating")
sor_proxy = _proxy("SportSource Team Success Ranking -> own strength of record", "sor")
margin_proxy = _proxy("SportSource capped relative scoring margin -> own capped conference margin", "conf_margin")


def composite_proxy(ctx: ConfContext, grp: list[int]) -> np.ndarray | None:
    """CFP ranking / composite of computer metrics (SP+, SOR, KPI, SportSource): proxy = z(rating)+z(SOR)."""
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)   # noqa: E731
    v = (z(ctx.rating) + z(ctx.sor))[grp]
    if np.allclose(v, v[0]):
        return None
    ctx.log("CFP ranking / computer composite -> own rating + strength of record")
    return v


def draw(ctx: ConfContext, grp: list[int]) -> np.ndarray:
    ctx.log("random draw")
    return ctx.rng.permutation(len(grp)).astype(float)


GENERIC = [h2h, common, descending, opp_winpct, rating_proxy, draw]
RULES: dict[str, dict] = {
    "sec": {"steps": [h2h, common, descending, opp_winpct, margin_proxy, draw], "verified": "official 2024 SEC procedure (PDF); no 2026 revision found",
            "source": "https://collegefootballnetwork.com/wp-content/uploads/2024/11/SEC-Tiebreaking-Procedures.pdf"},
    "big10": {"steps": [h2h, common, descending, opp_winpct, rating_proxy, draw], "verified": "Big Ten 2025 title-game release (worked examples, steps 1-4) and 2024 press coverage (steps 5-6); official PDF not retrievable",
              "source": "https://rrsn.com/?p=62717"},
    "big12": {"steps": [h2h, common, descending, opp_winpct, total_wins, rating_proxy, draw], "verified": "official Big 12 policy PDF (2024 policy, re-posted Nov 2025)",
              "source": "https://s3.amazonaws.com/big12sports.com/documents/2025/11/4/Big_12_Football_2024_Tiebreaker_Policy.pdf"},
    "acc": {"steps": [h2h_acc, sor_proxy, draw], "verified": "ACC release of July 15, 2026 as reproduced by FBSchedules; theacc.com page itself not retrievable", "acc_groups": True,
            "source": "https://fbschedules.com/acc-announces-new-football-tiebreaking-procedures/"},
    "mwest": {"steps": [h2h, composite_proxy, overall_pct, descending, draw], "verified": "pre-2026 Mountain West rules as quoted by SI; 2026 text not found",
              "source": "https://www.si.com/college/unlv/football/what-happens-if-the-unlv-rebels-end-up-in-a-tiebreaker-for-a-spot-in-the-mountain-west-championship-game"},
    "american": {"steps": [h2h, composite_proxy, overall_pct, draw], "verified": "UNVERIFIED: press summaries only (head-to-head, CFP ranking, overall record, coin toss)", "source": ""},
    "usa": {"steps": [h2h, composite_proxy, draw], "verified": "UNVERIFIED: press mention of a computer-ranking tiebreaker only", "source": ""},
    "midam": {"steps": GENERIC, "verified": "UNVERIFIED: league text not found; generic ladder used", "source": ""},
    "pac12": {"steps": GENERIC, "verified": "UNVERIFIED: league text not found; generic ladder used. Top two by conference record, top seed hosts (pac-12.com schedule release)", "source": ""},
    "belt": {"steps": GENERIC, "verified": "UNVERIFIED: league text not found; generic ladder used. East champion v West champion (FBSchedules)", "divisions": True, "source": ""},
}


def order(ctx: ConfContext, grp: list[int], steps: list[Step]) -> list[int]:
    """Order tied teams best-first."""
    if len(grp) <= 1:
        return list(grp)
    for step in steps:
        v = step(ctx, grp)
        if v is None or np.allclose(v, v[0]):
            continue
        best = [t for t, x in zip(grp, v) if np.isclose(x, v.max())]
        rest = [t for t in grp if t not in best]
        return order(ctx, best, steps) + order(ctx, rest, steps)
    return list(grp)


def standings(ctx: ConfContext, teams: list[int], steps: list[Step], acc_groups: bool = False) -> list[int]:
    """Full ordering of `teams` by conference winning percentage with ties broken by `steps`."""
    out: list[int] = []
    remaining = list(teams)
    pct, cw, cl = ctx.pct, ctx.cw, ctx.cl
    while remaining:
        top = max(pct[t] for t in remaining)
        grp = [t for t in remaining if np.isclose(pct[t], top)]
        if acc_groups:
            # ACC 2026: a team with a different number of league games joins the tie if it matches the
            # wins or the losses of a team in it
            games = {int(cw[t] + cl[t]) for t in grp}
            for t in remaining:
                if t not in grp and int(cw[t] + cl[t]) not in games and \
                        any(cw[t] == cw[g] or cl[t] == cl[g] for g in grp):
                    grp.append(t)
        ordered = order(ctx, grp, steps)
        if acc_groups and len(grp) > 1:
            ordered = ordered[:1]          # ACC: seat one team, then re-define the tied group and restart
        out += ordered
        remaining = [t for t in remaining if t not in ordered]
    return out


def title_game(ctx: ConfContext, rules: dict | None = None, has_divisions: bool | None = None) -> tuple[int, int]:
    """(seed 1, seed 2) as local indices. With divisions: the two division winners, better record first."""
    rules = rules or RULES.get(ctx.code, {"steps": GENERIC})
    steps = rules["steps"]
    divs = sorted({d for d in ctx.division if d})
    use_div = rules.get("divisions", False) if has_divisions is None else has_divisions
    if use_div and len(divs) == 2:
        champs = [standings(ctx, [t for t in range(ctx.k) if ctx.division[t] == d], steps)[0] for d in divs]
        return tuple(order(ctx, champs, [lambda c, g: c.pct[g], *steps]))          # type: ignore[return-value]
    s = standings(ctx, list(range(ctx.k)), steps, rules.get("acc_groups", False))
    return s[0], s[1]
