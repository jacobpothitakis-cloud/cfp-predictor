"""Selection, seeding and bracket rules."""
from __future__ import annotations

from cfp.committee.field import select_field


def setup(n=40):
    conf = ["sec"] * 6 + ["big10"] * 6 + ["big12"] * 5 + ["acc"] * 5 + ["ind"] + ["mwest"] * 5 + ["american"] * 6 + ["belt"] * 6
    return list(range(n)), conf[:n]


def test_four_team_era_is_top_four():
    order, conf = setup()
    f = select_field(order, conf, {}, 2019)
    assert f.seeds == [0, 1, 2, 3]


def test_2026_straight_seeding_and_g6_team_not_champion():
    order, conf = setup()
    # champions: SEC=0, Big Ten=6, Big 12=12, ACC=17. Best G6 team is index 23 (mwest), ranked 24th; the
    # mwest champion is someone else (index 25). The bid goes to the highest-ranked G6 TEAM.
    champs = {"sec": 0, "big10": 6, "big12": 12, "acc": 17, "mwest": 25}
    f = select_field(order, conf, champs, 2026, notre_dame=22)
    assert f.seeds[:4] == [0, 1, 2, 3]                 # byes go to the top four regardless of titles
    assert 23 in f.teams and 25 not in f.teams
    assert f.auto[23] == "highest-ranked Group of Six team"
    # automatic qualifiers outside the top 12 (12, 17 and 23) sit at the bottom in rank order
    assert f.seeds[-3:] == [12, 17, 23]
    assert len(f.seeds) == 12 and len(set(f.seeds)) == 12
    # three automatic bids from outside the top 12 bump the three lowest at-large teams (ranked 10, 11, 12)
    assert f.bumped == [9, 10, 11]


def test_notre_dame_protected_in_top_12():
    order, conf = setup()
    conf[11] = "ind"                                   # Notre Dame ranked 12th
    champs = {"sec": 0, "big10": 6, "big12": 12, "acc": 17}
    f = select_field(order, conf, champs, 2026, notre_dame=11)
    assert 11 in f.teams
    assert f.bumped == [8, 9, 10]                      # the bumped teams are the lowest non-protected at-larges
    f2 = select_field(order, conf, champs, 2026, notre_dame=None)
    assert 11 not in f2.teams


def test_notre_dame_outside_top_12_is_out():
    order, conf = setup()
    f = select_field(order, conf, {"sec": 0, "big10": 6, "big12": 12, "acc": 17}, 2026, notre_dame=22)
    assert 22 not in f.teams


def test_2024_byes_go_to_four_highest_ranked_champions():
    order, conf = setup()
    champs = {"sec": 1, "big10": 6, "big12": 12, "acc": 17, "mwest": 23}
    f = select_field(order, conf, champs, 2024)
    assert f.seeds[:4] == [1, 6, 12, 17]
    assert 23 in f.teams and f.seeds[-1] == 23


def test_2025_five_highest_ranked_champions_straight_seeding():
    order, conf = setup()
    champs = {"sec": 1, "big10": 6, "big12": 12, "acc": 30, "mwest": 23, "american": 29}
    f = select_field(order, conf, champs, 2025)
    assert f.seeds[:4] == [0, 1, 2, 3]
    assert {1, 6, 12, 23, 29} <= f.teams and 30 not in f.teams   # the ACC champion ranked 31st misses out


def test_bracket_poster_draws_all_twelve_seeds():
    from cfp.app.poster import bracket_poster
    seeds = {k: {"name": f"Team {k}", "team_id": 1000 + k} for k in range(1, 13)}
    html = bracket_poster(seeds, {1: "Peach", 2: "Rose", 3: "Cotton", 4: "Fiesta"}, subtitle="test")
    assert html.count("<image") == 12 and "FIESTA BOWL" in html and "PROJECTED BRACKET" in html
