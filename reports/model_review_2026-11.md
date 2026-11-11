# Model review, November 2026

Everything that was tried to improve the model after its first live week, what was kept and what was thrown out.

## How ideas were judged

A new walk-forward test (`cfp/committee/walkforward.py`) runs the weekly model the way it is used: for each of
the 72 rankings in 2014–2025 it trains on every other season plus that season's earlier rankings, then predicts.
A change was kept only if it lowered the average miss and did so in clearly more seasons than it hurt.

Caveat: the settings below were chosen on the same twelve seasons they are scored on, so the gains are
somewhat optimistic. The live 2026 log is the only fully clean test.

## Result

| | Before | After |
|---|---|---|
| Average miss per team, all rankings | 1.21 places | 1.10 places |
| Within one spot | 72% | 74% |
| Ranks 1–8 | 0.56 | 0.52 |
| Ranks 9–16 | 1.10 | 0.98 |
| Ranks 17–25 | 1.89 | 1.72 |
| Top 12 teams identified | 95.8% | 96.1% |
| Top 25 teams identified | 94.9% | 95.6% |
| Exact Selection Day field | 8 of 12 | 8 of 12 |

Better in 11 of 12 seasons (2019 slightly worse). The simulation's carry-over model, tested separately,
went from 1.64 to 1.43 and improved in all 12 seasons.

## Kept

| Change | Evidence |
|---|---|
| Rankings already released this season count six times as much in training (weekly model and simulation model) | Weekly: 1.215 → 1.172, 10 of 12 seasons. Simulation model: 1.638 → 1.434, 12 of 12. |
| Ensemble mix moved to 70% pairwise logistic, 30% LambdaRank | 1.172 → 1.122, 10 of 12 seasons. |
| New input: poll points of teams just outside the AP and Coaches top 25 ("others receiving votes") | 1.120 → 1.098, 8 of 12 seasons; ranks 17–25 from 1.78 to 1.72. Data was already in the cached ESPN responses. |
| Betting lines in the simulation: where a line is posted, the game's expected margin is 70% line, 30% rating | 3,112 games in the week after each ranking, 2014–2026: log loss 0.522 → 0.506, better than ratings alone in all 13 seasons. |
| Two-source score check, now that CollegeFootballData is connected | 11,057 completed games compared; two differ, each by one point, neither changes a winner (2018 Florida Atlantic–Air Force, Georgia State–Texas State). |

## Found and fixed

- **Home and away reversed in CollegeFootballData for 15 neutral-site games.** Merged as-is it would have credited
  the wrong winner (for example LSU over Florida State, 2022). Now aligned by team id before use, with a test.
- **A documentation claim that was not true**: the protocol notes said the model had a recusal feature. It never did.
- **A media call about playoff expansion filed as a chair's ranking call.** Excluded.

## Tested and rejected

| Idea | Result |
|---|---|
| Chair's conference gets a boost | No. Conferences did no better in years their own AD chaired: −0.08 places against +0.13 otherwise (p = 0.54). As a model input: 6 seasons better, 6 worse. |
| More sitting ADs from a conference on the committee helps that conference | Weak and not significant (p = 0.33 résumé model, 0.11 full model). As an input: 6 better, 3 worse, 0.009 places. Not adopted. Affiliations were found for 47 of 61 members, so this test is incomplete. |
| A team whose own AD is on the committee is ranked higher | Against the résumé-only model such teams sit 0.5 places higher than expected (31 team-seasons, p = 0.085), but the AP and Coaches polls rate them higher too, and against the full model the gap is 0.09 (p = 0.56). As an input: 5 better, 7 worse. Not a committee bias the model can use. |
| CollegeFootballData's play-by-play postgame win probability in place of the margin-based stand-in | Worse: 1.172 → 1.187, 9 of 12 seasons worse. Reverted. |
| Market-implied team rating (from past closing spreads) as a committee input | 1.128 → 1.120, 6 better, 5 worse. Noise. |
| Elo, net points-per-play, net success rate, record against the spread | All within ±0.006, none consistent. |
| Wins over teams in the committee's own previous top 25 / top 12 | 5 better, 6 worse. |
| Stronger or weaker regularisation, longer or shorter memory across seasons, era-term scaling | No setting beat the current one in more seasons than it lost. |
| Pulling predictions toward the résumé-only ranking; an idle-week penalty | Both made history worse (see `diagnosis_2026-11-10.md`). |
| SP+, SRS and FPI ratings as inputs | Not testable: the sources serve end-of-season values, which would leak the future into past weeks. |

## Committee membership: what the data says

Source: the CFP's official all-time membership page (61 members, chairs by season), saved as
`data/committee_roster.csv`. Schools for sitting athletic directors came from press and CFP announcements.

The chair's conference shows no advantage over twelve seasons. The 2026 pattern (SEC teams about a place
higher than the model expected, with an SEC athletic director as chair) is therefore better explained by the
reasons the chair gave, schedule strength and quality wins, than by who he is. The model does not use the
roster, and the same-season weighting picks up whatever this year's committee is doing differently without
needing to know why.

## What still limits it

- The committee's private record-strength and schedule-strength figures.
- Film and injuries. The injury input remains manual.
- Two seasons of the 12-team format, and a 24-team format coming in 2027.
- The 2026 live record is one ranking.
