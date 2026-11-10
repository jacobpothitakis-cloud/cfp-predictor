"""Conference tiebreakers: hand-built cases for each step, plus real 2024-25 title-game matchups."""
from __future__ import annotations

import numpy as np
import pytest

from cfp.conferences import tiebreak as tb
from cfp.conferences.tiebreak import ConfContext


def ctx(k, results, code="sec", **kw):
    w = np.zeros((k, k))
    for a, b in results:
        w[a, b] += 1
    base = dict(code=code, wins=w, division=[""] * k, rating=np.zeros(k), sor=np.zeros(k),
                overall_w=np.zeros(k), overall_pct=np.zeros(k), conf_margin=np.zeros(k),
                rng=np.random.default_rng(0))
    base.update(kw)
    return ConfContext(**base)


def test_two_team_head_to_head():
    # 0 and 1 both 2-1; 0 beat 1.
    c = ctx(4, [(0, 1), (0, 2), (3, 0), (1, 2), (1, 3), (2, 3)])
    assert tb.order(c, [0, 1], tb.RULES["sec"]["steps"]) == [0, 1]


def test_three_team_round_robin_cycle_falls_to_common_opponents():
    # 0>1, 1>2, 2>0 (cycle). Against common opponents 3 and 4: team 1 is 2-0, team 0 is 1-1, team 2 is 0-2.
    # Team 1 is seeded on common opponents; the other two then RESTART at head-to-head, which 2 won.
    c = ctx(5, [(0, 1), (1, 2), (2, 0), (0, 3), (4, 0), (1, 3), (1, 4), (3, 2), (4, 2)])
    assert tb.order(c, [0, 1, 2], tb.RULES["sec"]["steps"]) == [1, 2, 0]


def test_team_that_beat_all_others_is_seeded_then_rest_restart():
    # 0 beat 1 and 2; 1 and 2 did not play each other. After 0 is seeded, 1 v 2 goes to common opponents (3).
    c = ctx(4, [(0, 1), (0, 2), (2, 3), (3, 1)])
    assert tb.order(c, [0, 1, 2], tb.RULES["big12"]["steps"]) == [0, 2, 1]


def test_descending_standings_groups_tied_opponents():
    # 0 and 1 did not play. Common opponents 2, 3 (both top of the standings, tied) and 4.
    # Each is 1-1 against the tied pair {2, 3} taken together, so that tier cannot split them even though 0 beat
    # the "better" one; they split on the next tier (team 4): 0 won, 1 lost.
    games = [(0, 2), (3, 0), (2, 1), (1, 3), (0, 4), (4, 1), (2, 4), (3, 4), (2, 5), (3, 5), (2, 3), (3, 2)]
    c = ctx(6, games)
    v = tb.descending(c, [0, 1])
    assert v is not None and v[0] > v[1]
    assert tb.common(c, [0, 1]) is not None


def test_opponent_win_pct_step():
    # 0 and 1 unbeaten vs different opponents; 0's opponents have the better league record.
    c = ctx(6, [(0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)])
    assert tb.h2h(c, [0, 1]) is None and tb.common(c, [0, 1]) is None
    assert tb.order(c, [0, 1], tb.RULES["sec"]["steps"]) == [0, 1]


def test_big12_total_wins_before_rating_and_proxy_is_logged():
    c = ctx(2, [], code="big12", overall_w=np.array([9, 10]), rating=np.array([5.0, 1.0]))
    assert tb.order(c, [0, 1], tb.RULES["big12"]["steps"]) == [1, 0]
    assert c.proxy_log == {}
    c2 = ctx(2, [], code="big10", rating=np.array([5.0, 1.0]))
    assert tb.order(c2, [0, 1], tb.RULES["big10"]["steps"]) == [0, 1]
    assert sum(c2.proxy_log.values()) == 1 and "rating" in next(iter(c2.proxy_log))


def test_random_draw_is_seeded_and_logged():
    a = ctx(3, [], rng=np.random.default_rng(5)); b = ctx(3, [], rng=np.random.default_rng(5))
    assert tb.order(a, [0, 1, 2], tb.GENERIC) == tb.order(b, [0, 1, 2], tb.GENERIC)
    assert any("random draw" in k for k in a.proxy_log)


def test_divisions_send_two_division_winners():
    # East: 0, 1.  West: 2, 3.  Team 1 has the second-best league record but is in 0's division.
    c = ctx(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], code="belt", division=["E", "E", "W", "W"])
    assert tb.title_game(c) == (0, 2)


def test_acc_alternate_game_counts_join_the_tie():
    # Team 0 is 2-0 (two league games), team 1 is 2-1 (three league games): same wins, so 1 joins the tie
    # for first, and 1 beat 0 head-to-head?  Here 0 beat 1, so 0 stays first.
    c = ctx(4, [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 2)], code="acc")
    s1, s2 = tb.title_game(c)
    assert (s1, s2) == (0, 1)


@pytest.mark.data
@pytest.mark.parametrize("season,conf", [(2024, "sec"), (2025, "sec"), (2024, "big10"), (2025, "big10"),
                                         (2024, "big12"), (2025, "big12")])
def test_real_title_game_matchups(season, conf):
    """The procedure must reproduce who actually played in the 2024 and 2025 title games of the three
    leagues whose current rules were in force then (SEC, Big Ten, Big 12)."""
    from cfp.config import DB_PATH
    if not DB_PATH.exists():
        pytest.skip("database not built")
    from cfp.conferences.races import actual_title_game, historical_title_game
    assert set(historical_title_game(season, conf)) == set(actual_title_game(season, conf))
