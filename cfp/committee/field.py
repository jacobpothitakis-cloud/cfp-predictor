"""Selection and seeding rules, by season. Sources: docs/committee_protocol.md.

2014-2023  four teams: the top four of the final ranking.
2024       twelve teams: five highest-ranked conference champions + seven at-large;
           seeds 1-4 reserved for the four highest-ranked champions.
2025       same field rule; seeding straight by ranking.
2026       champions of the ACC, Big Ten, Big 12, SEC + the highest-ranked Group of Six TEAM
           (not necessarily a champion) + seven at-large; Notre Dame guaranteed if in the top 12;
           seeding straight by ranking; an automatic qualifier outside the top 12 goes to the bottom.
"""
from __future__ import annotations

from dataclasses import dataclass

P4 = ("acc", "big10", "big12", "sec")
G6 = ("american", "usa", "midam", "mwest", "belt", "pac12")


@dataclass
class Field:
    seeds: list[int]                 # team index by seed (seed 1 first)
    auto: dict[int, str]             # team index -> reason for an automatic bid
    bumped: list[int]                # top-12 teams left out because of automatic bids
    byes: list[int]

    @property
    def teams(self) -> set[int]:
        return set(self.seeds)


def select_field(order: list[int], conf: list[str], champions: dict[str, int], season: int,
                 notre_dame: int | None = None) -> Field:
    """order: team indices best-first (must cover every team that could qualify).
    champions: conference code -> team index of its champion (missing = no champion)."""
    rank = {t: i + 1 for i, t in enumerate(order)}
    if season <= 2023:
        top = list(order[:4])
        return Field(top, {}, [], [])
    auto: dict[int, str] = {}
    if season <= 2025:
        champs = sorted((t for t in champions.values() if t in rank), key=lambda t: rank[t])[:5]
        for t in champs:
            auto[t] = "conference champion (one of the five highest-ranked)"
    else:
        for c in P4:
            if c in champions:
                auto[champions[c]] = f"{c} champion"
        g6 = [t for t in order if conf[t] in G6]
        if g6:
            auto[g6[0]] = "highest-ranked Group of Six team"
    field = list(auto)
    protected = set(auto)
    if season >= 2026 and notre_dame is not None and rank.get(notre_dame, 999) <= 12:
        protected.add(notre_dame)
        if notre_dame not in field:
            field.append(notre_dame)
    for t in order:
        if len(field) >= 12:
            break
        if t not in field:
            field.append(t)
    field = sorted(field, key=lambda t: rank[t])
    bumped = [t for t in order[:12] if t not in field]
    if season == 2024:
        champs = [t for t in field if t in auto][:4]
        rest = [t for t in field if t not in champs]
        seeds = champs + rest
    else:
        seeds = field
    return Field(seeds, auto, bumped, seeds[:4])


def bracket(seeds: list[int]) -> dict[str, list[tuple[int, int]]]:
    """First-round pairings as (host seed, visitor seed) and the quarterfinal each feeds."""
    return {"first_round": [(5, 12), (6, 11), (7, 10), (8, 9)],
            "quarterfinals": [(4, 5), (3, 6), (2, 7), (1, 8)],   # top seed v. winner of the host seed's game
            "semifinals": [(1, 4), (2, 3)]}                      # by quarterfinal branch
