"""Unit tests for the résumé feature engine on a tiny hand-built season."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfp.features import engine
from cfp.features.state import GameArrays, SeasonState, game_control_home


def tiny_state() -> tuple[SeasonState, GameArrays, np.ndarray]:
    """Four FBS teams A,B,C,D (+FCS). Results: A beat B at home 30-10, A beat C away 21-20, B beat C at home
    28-0, D lost to FCS at home 10-17, A beat D on a neutral field in a title game 40-0, C beat D away 14-7."""
    n = 4
    gh = np.array([0, 2, 1, 3, 0, 3]); ga_ = np.array([1, 0, 2, 4, 3, 2])
    margin = np.array([20, -1, 28, -7, 40, -7], float)
    nan = np.full(6, np.nan)
    ga = GameArrays(gh=gh, ga=ga_, neutral=np.array([0, 0, 0, 0, 1, 0], bool),
                    conf_game=np.array([1, 1, 1, 0, 1, 1], bool), ccg=np.array([0, 0, 0, 0, 1, 0], bool),
                    day=np.array([30, 37, 44, 51, 100, 58], float), rank_h=nan.copy(),
                    rank_a=np.array([10, np.nan, np.nan, np.nan, np.nan, np.nan]), pg_wp_h=nan.copy(), gc_h=nan.copy(),
                    game_id=np.arange(6), week=np.arange(6))
    st = SeasonState(season=2026, cutoff=pd.Timestamp("2026-11-10", tz="UTC"), team_ids=np.arange(4),
                     names=list("ABCD"), conf=np.array(["x", "x", "x", "x"]), division=np.array([""] * 4),
                     power=np.array([1, 1, 1, 1], bool), indep=np.zeros(4, bool), g6=np.zeros(4, bool),
                     played=ga, margin=margin, home_pts=np.zeros(6), away_pts=np.zeros(6),
                     future=ga.take(np.zeros(6, bool)), rating=np.array([20.0, 10.0, 0.0, -10.0, -19.0]),
                     ap=np.full(4, np.nan), coaches=np.full(4, np.nan), prev_cfp=np.full(4, np.nan),
                     has_prev_cfp=False, preseason_ap=np.full(4, np.nan), cutoff_day=101.0)
    return st, ga, margin


@pytest.fixture(scope="module")
def feats():
    st, ga, margin = tiny_state()
    f = engine.compute(st, ga, margin[None], st.rating[None], hfa=2.5, sigma=14.0, final_week=True)
    return {k: v[0] for k, v in f.items()}


def test_records(feats):
    assert feats["wins"].tolist() == [3, 1, 1, 0]
    assert feats["losses"].tolist() == [0, 1, 2, 3]
    assert feats["win_pct"][0] == 1.0 and feats["no_loss"].tolist() == [1, 0, 0, 0]


def test_venue_splits(feats):
    assert feats["w_home"][0] == 1 and feats["w_road"][0] == 1 and feats["w_neutral"][0] == 1
    assert feats["l_home"][3] == 2      # D lost at home to the FCS team and to C


def test_fcs(feats):
    assert feats["fcs_games"].tolist() == [0, 0, 0, 1]
    assert feats["fcs_losses"].tolist() == [0, 0, 0, 1]


def test_conference_and_title_game(feats):
    assert feats["champion"].tolist() == [1, 0, 0, 0]
    assert feats["ccg_loss"].tolist() == [0, 0, 0, 1]
    assert feats["conf_wins"][0] == 2       # the title game is not counted in the league record
    assert feats["idle_final"].tolist() == [0, 1, 1, 0]


def test_record_vs_500_and_ranked_at_game_time(feats):
    # B finished 1-1 (.500): A's win over B is a .500+ win. C (1-2) and D (0-3) are not.
    assert feats["w500"][0] == 1 and feats["l500"][1] == 1
    assert feats["w_ranked_g"][0] == 1 and feats["w_top10_g"][0] == 1   # B was No. 10 at kickoff


def test_best_win_prefers_strong_opponent_on_the_road(feats):
    st, ga, margin = tiny_state()
    # A's wins: B (rating 10) at home, C (0) on the road, D (-10) neutral. Best is B even after the home discount.
    top10 = np.sort(st.rating[:4])[-10:].mean()
    assert feats["best_win"][0] == pytest.approx(10 - 2.5 - top10, abs=1e-4)
    assert feats["best_win2"][0] == pytest.approx(0 + 2.5 - top10, abs=1e-4)
    assert feats["best_win"][3] == engine.NO_WIN      # winless team


def test_loss_to_weak_team_costs_more_than_loss_to_strong_team(feats):
    # B's only loss is at A (best team); D's losses include the FCS team at home.
    assert feats["loss_penalty"][3] > feats["loss_penalty"][1] * 3
    assert feats["worst_loss"][3] < feats["worst_loss"][1]


def test_beating_strong_team_earns_more_credit():
    st, ga, margin = tiny_state()
    f = engine.compute(st, ga, margin[None], st.rating[None], hfa=2.5, sigma=14.0)
    # C's one win is at D (weak); B's one win is over C at home (weak-ish). A has three.
    assert f["win_credit"][0, 0] > f["win_credit"][0, 1] > 0


def test_margin_is_capped_at_24(feats):
    # A's margins: +20, +1, +40 -> capped mean (20 + 1 + 24) / 3
    assert feats["margin_cap"][0] == pytest.approx(15.0, abs=1e-5)


def test_simulation_axis_matches_single_run():
    st, ga, margin = tiny_state()
    one = engine.compute(st, ga, margin[None], st.rating[None], hfa=2.5, sigma=14.0)
    many = engine.compute(st, ga, np.repeat(margin[None], 5, 0), np.repeat(st.rating[None], 5, 0), hfa=2.5, sigma=14.0)
    for k in one:
        assert np.allclose(many[k][3], one[k][0], equal_nan=True), k


def test_head_to_head_and_common_opponents():
    st, ga, margin = tiny_state()
    pw = engine.pairwise(st, ga, margin[None], np.array([0, 1, 2, 3]))
    h = pw["h2h"][0]
    assert h[0, 1] == 1 and h[1, 0] == -1 and h[1, 2] == 1 and h[2, 3] == 1 and h[1, 3] == 0
    # A and B both played C: A won, B won -> no edge. B and D both played A (lost, lost) and C (B won, D lost).
    assert pw["n_common"][0, 1] == 1 and pw["common"][0, 0, 1] == 0
    assert pw["n_common"][1, 3] == 2 and pw["common"][0, 1, 3] == pytest.approx(1.0)   # (0 net vs -2 net) / 2


def test_game_control_proxy():
    assert game_control_home("[14, 14, 7, 0]", "[0, 0, 0, 7]", 28) > 0.9
    assert game_control_home("[0, 0, 0, 21]", "[7, 7, 3, 0]", 4) < 0.5     # trailed all game, won late
    assert np.isnan(game_control_home(None, None, 3))


# ----------------------------------------------------------------------------- inputs added in the Nov 2026 model review
def test_receiving_votes_only_for_teams_outside_the_top_25():
    import numpy as np
    from cfp.committee import dataset
    from cfp.features import external
    rel = dataset.all_releases()[-1]
    rv = external.receiving_votes(rel.st)
    assert rv.min() >= 0 and rv.max() <= 1
    ranked_both = ~np.isnan(rel.st.ap) & ~np.isnan(rel.st.coaches)
    assert (rv[ranked_both] == 0).all()          # a team ranked in both polls is not an "other receiving votes"
    assert (rv > 0).sum() >= 3


def test_cfbd_home_away_is_aligned_to_espn():
    """CFBD lists some neutral-site games with the teams the other way round; after alignment the two sources
    may differ by a point here and there but never on who won."""
    from cfp import data
    if not data.has_table("cfbd_games"):
        return
    c = data.table("score_conflicts")
    assert len(c) < 10
    assert (((c.espn_home > c.espn_away) == (c.cfbd_home > c.cfbd_away))).all()


def test_same_season_rankings_are_weighted_up():
    from cfp.committee import model
    assert model._weight(2026, 2026, "full") == model.SEASON_BOOST
    assert model._weight(2026, 2026, "anchor") == model.SEASON_BOOST
    assert model._weight(2026, 2026, "level") == 1.0
    assert model._weight(2026, 2025, "full") == model.RECENCY
