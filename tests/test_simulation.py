"""Simulation, what-if parsing, bracket play and an end-to-end run on a frozen historical week."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfp.config import DB_PATH
from cfp.simulate.season import Forced, _play, parse_result


def test_parse_results():
    f = parse_result("Ohio State over Michigan 27-24")
    assert (f.winner, f.loser, f.margin) == ("Ohio State", "Michigan", 3)
    f = parse_result("Ohio State beats Michigan by 3")
    assert f.margin == 3
    f = parse_result("Texas over Texas A&M")
    assert f.margin is None and f.loser == "Texas A&M"
    f = parse_result("Georgia loses the SEC title game")
    assert f.title_game and not f.wins and f.winner == "Georgia"
    f = parse_result("Miami wins title game")
    assert f.title_game and f.wins
    with pytest.raises(ValueError):
        parse_result("nonsense")
    with pytest.raises(ValueError):
        parse_result("A over B 21-21")


def test_bracket_play_structure():
    """With no randomness in strength (huge gaps) the chalk bracket must hold: seed 1 wins it all, and each
    team's furthest round follows the bracket lines (no reseeding)."""
    seeds = list(range(12))
    r = np.array([1000.0 - 100 * i for i in range(12)])
    rounds = np.zeros(12, np.int8)
    _play(seeds, r, 2.5, 13.0, np.random.default_rng(0), rounds)
    assert rounds[0] == 5                       # champion
    assert rounds[1] == 4                       # 2 seed loses the final
    assert rounds[2] == 3 and rounds[3] == 3    # 3 and 4 lose semifinals
    assert list(rounds[4:8]) == [2, 2, 2, 2]    # first-round hosts win, lose quarterfinals
    assert list(rounds[8:]) == [1, 1, 1, 1]


def test_no_reseeding_five_seed_path():
    """A dominant 12 seed must go through 5, then 4, then the 1/8/9 side, never meeting 2 or 3 before the final."""
    seeds = list(range(12))
    r = np.zeros(12); r[11] = 1000.0; r[1] = 500.0
    rounds = np.zeros(12, np.int8)
    _play(seeds, r, 2.5, 13.0, np.random.default_rng(1), rounds)
    assert rounds[11] == 5 and rounds[1] == 4   # the 2 seed still reaches the final from the other half


pytestmark_data = pytest.mark.data


@pytest.fixture(scope="module")
def frozen():
    if not DB_PATH.exists():
        pytest.skip("database not built")
    from cfp.committee import backtest as bt
    from cfp.committee import dataset
    rels = list(dataset.all_releases())
    sr = sorted([r for r in rels if r.season == 2023], key=lambda r: r.release_no)
    models = bt.fit_models([r for r in rels if r.season != 2023], 2023)
    return rels, sr, models


@pytest.mark.data
@pytest.mark.slow
def test_end_to_end_frozen_week_2023(frozen):
    """Frozen week: the 4th ranking of 2023 (Nov 21). Models trained without 2023 must reproduce a sane
    ranking from the as-of data: this guards the whole ingest -> ratings -> features -> model chain."""
    from cfp.committee import backtest as bt
    rels, sr, models = frozen
    rel = sr[3]
    assert str(rel.cutoff)[:10] == "2023-11-21"
    # as-of integrity: no game after the cutoff is in the state
    start = pd.Timestamp("2023-08-01", tz="UTC")
    assert rel.st.played.day.max() < (rel.cutoff.value - start.value) / 86400e9
    order = models[bt.PRIMARY].order(rel)
    m = bt.metrics(order, rel)
    assert m["mae"] < 2.0 and m["top4_overlap"] >= 0.75 and m["top12_overlap"] >= 10 / 12
    # deterministic
    assert list(order[:25]) == list(models[bt.PRIMARY].order(rel)[:25])


@pytest.mark.data
@pytest.mark.slow
def test_simulation_is_reproducible_and_consistent(frozen):
    from cfp import data
    from cfp.simulate.season import simulate
    rels, sr, models = frozen
    a = sr[0]
    g = data.games()
    cc = g[(g.season == 2023) & g.is_conf_champ & (g.home_conf == g.away_conf)]
    tg = {c: {"date": str((d - pd.Timedelta(hours=8)).date()), "site": "neutral"} for c, d in zip(cc.home_conf, cc.start_date)}
    kw = dict(now=a.cutoff, n_sims=300, bundle={"anchor": models["_anchor"], "sigma": {"anchor": 1.7}}, st=a.st,
              anchor_rel=a, title_confs=tg, playoff=False, legacy_rules=True, final_no=len(sr))
    r1 = simulate(2023, seed=11, **kw); r2 = simulate(2023, seed=11, **kw); r3 = simulate(2023, seed=12, **kw)
    pd.testing.assert_frame_equal(r1.teams, r2.teams)
    assert not r1.teams["p_field"].equals(r3.teams["p_field"])
    # four-team era: exactly four teams per simulated season
    assert np.all((r1.raw["seed"] > 0).sum(1) == 4)
    assert abs(r1.teams["p_field"].sum() - 4.0) < 0.05
    # one champion per league with a title game
    assert np.all(r1.raw["champ"].sum(1) == len(tg))
    # a forced result is respected in every simulation
    t = r1.teams
    fut = a.st.future
    gi = next(i for i in range(len(fut)) if fut.gh[i] < a.st.n and fut.ga[i] < a.st.n and not fut.ccg[i])
    h, aw = a.st.names[fut.gh[gi]], a.st.names[fut.ga[gi]]
    rf = simulate(2023, seed=11, forced=[Forced(winner=aw, loser=h, margin=10)], **kw)
    assert len(rf.games[rf.games.game.str.contains(f"{aw} at {h}", regex=False)]) == 0   # no variation left to measure


@pytest.mark.data
def test_2026_field_rules_in_simulation_outputs():
    """Whatever the last update simulated: 12 seeds per season, four P4 champions always in."""
    from cfp.pipeline import load_outputs
    o = load_outputs()
    if o is None:
        pytest.skip("no outputs yet; run `cfp update`")
    t = o["sim"]["teams"]
    assert abs(t["p_field"].sum() - 12.0) < 0.2
    assert abs(o["sim"]["seeds"].sum().sum() - 12.0) < 0.2
    assert ((t["p_bye"] + t["p_host"]) <= t["p_field"] + 1e-9).all()
