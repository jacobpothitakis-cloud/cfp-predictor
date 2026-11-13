# CFP committee ranking predictor

Predicts what the College Football Playoff selection committee will do: its next weekly Top 25, its
Selection Day ranking, the 12-team field and the bracket. The target is the committee's ranking, not a
power rating; ratings and polls are inputs.

It does three things:

1. **Next ranking.** After a week's games, predict the next committee Top 25 with a probability for every rank
   and a plain-English reason for every team.
2. **Season simulation.** Simulate every remaining game, conference race and title game 10,000 times, apply
   the committee model and the 2026 selection rules to each, and report field / bye / host / seed / title odds.
3. **What-if.** Force any results and see the ranking, field and odds.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
./scripts/fix_lightgbm_macos.sh     # macOS without Homebrew only: points LightGBM at scikit-learn's libomp
cp .env.example .env                # then put your CollegeFootballData key after CFBD_API_KEY=
```

The key is read from the environment and `.env` is git-ignored. Without a key everything still runs on
ESPN's public feeds; see "What is missing without the CFBD key".

## Weekly workflow

```bash
.venv/bin/cfp update            # Sunday or Monday: fetch, rebuild, retrain, predict, simulate, write the weekly report
.venv/bin/cfp predict           # the next ranking with ranges and explanations (also stores it for later scoring)
.venv/bin/cfp simulate          # 10,000 simulated seasons
.venv/bin/cfp whatif -r "Ohio State over Michigan 27-24" -r "Georgia loses SEC title game"
.venv/bin/cfp team "Notre Dame" # résumé sheet
.venv/bin/cfp backtest          # regenerate reports/backtest_report.md (about 5 minutes)
.venv/bin/cfp-dashboard         # local Streamlit dashboard
.venv/bin/python -m pytest      # 52 tests
```

`cfp update` stops with a message if any fetch fails; it never fills a gap from memory. Every response is
cached under `data/cache/` (completed seasons permanently, current-season pages re-fetched by `update`).
After the committee releases a ranking, the next `cfp update` scores the stored prediction against it and
appends to `reports/2026_prediction_log.md`.

Outputs: `reports/weekly_<date>.md`, `reports/backtest_report.md`, `reports/ratings_validation.md`,
`reports/simulation_calibration.md`, `reports/2026_prediction_log.md`, `reports/predictions/`.

## How each layer works

| Layer | Code | What it does |
|---|---|---|
| Ingest | `cfp/ingest/` | ESPN scoreboards (every game involving an FBS team, 2014–2026, with quarter scores), weekly CFP / AP / Coaches rankings, conference membership by season, FPI table; CFBD client for games, lines, SP+, SRS, Elo, advanced stats; chair transcripts from ASAP Sports. All into `data/cfp.duckdb`. |
| Team strength | `cfp/ratings/` | Walk-forward ridge rating on capped margins with a preseason prior from last season. Every game is predicted from the snapshot before kickoff. Win probability = Φ(rating gap / 14). |
| Résumé features | `cfp/features/` | About 80 features per team per week, strictly as-of: record, strength of record, schedule strength, reconstructed "record strength", records v ranked / .500+ / power teams, best wins, worst loss, venue splits, capped margin, game control, recency, this week's result, conference title and strength, head-to-head and common opponents. The same array code runs for one real week or thousands of simulated seasons. |
| Committee model | `cfp/committee/` | Pairwise logistic regression on feature differences (interpretable), a head-to-head / common-opponent term that only bites when two teams are close (the six-team voting pools), and a LightGBM LambdaRank model; the shipped ranking averages the two. Three variants: résumé only; plus polls and last week's rank (used for the next ranking); and "carry-over" from an earlier committee ranking with no polls (used in simulation, where future polls do not exist). Era terms let weights shift for the 12-team era and the 2025 metric changes. |
| Uncertainty | `cfp/committee/uncertainty.py` | Gaussian noise on team scores, re-ranked thousands of times; noise scale calibrated on leave-one-season-out errors. |
| Conference races | `cfp/conferences/` | Each league's tiebreaker ladder as a list of steps; title-game participants for real or simulated standings. |
| Simulation | `cfp/simulate/` | Draw team strengths, simulate remaining games, update ratings, resolve conference races and title games, recompute every feature, apply the committee model with noise, apply the 2026 selection and seeding rules, play the bracket. Seeded and reproducible. |
| Interface | `cfp/cli.py`, `cfp/app/dashboard.py`, `cfp/report_weekly.py` | CLI, dashboard, weekly report. |

Rules and their sources are in `docs/committee_protocol.md`; the saved text of every cited page is in
`docs/sources/`.

## How good is it

Leave-one-season-out, 2014–2025, 72 weekly rankings (full detail in `reports/backtest_report.md`):

| | Mean abs. rank error | Within 1 | Top-12 overlap | Final field exactly right |
|---|---|---|---|---|
| Model (polls + last week) | 1.21 | 72% | 95% | 7 of 12 seasons |
| AP poll | 1.78 | 60% | 95% | 4 of 12 |
| Last week's ranking adjusted for losses | 1.85 | 61% | 94% | 4 of 12 |
| Simulation model (carry-over from first ranking) | 1.84 | 57% | 94% | 9 of 12 |
| Pure strength of record | 4.39 | 30% | 84% | 6 of 12 |

This test never shows the model a season's own earlier rankings. Run the way it is actually used, with those
included (`reports/walkforward_metrics.csv`), the weekly model misses by 1.10 places and names the exact final
field in 8 of 12 seasons. (The table reflects the November 2026 model review; before it the first row read
1.30, 69%, 95%, 8 of 12.)

Simulated playoff odds, checked against 2014–2025 from each season's first and third rankings: Brier skill
0.58 over the base rate (`reports/simulation_calibration.md`). Team strength: 74.9% of winners walk-forward,
well calibrated (`reports/ratings_validation.md`).

Live 2026 results are scored automatically by each `cfp update` (`reports/2026_prediction_log.md`, and the
dashboard's Report card tab). First scored ranking, Nov 10, 2026: mean miss 1.84 places, 4 of 25 exact, 48% within
one spot, top four 2 of 4, top twelve 11 of 12; the stated 80% range held the actual rank for 76% of teams.
That is weaker than the backtest average. Biggest misses: Indiana (predicted 4, actual 9 after an idle week),
Oklahoma State (outside the model's top 30, actual 25), Ole Miss (15 v 19), Alabama (6 v 3).

After that ranking the weekly model was changed to weight the current season's earlier rankings six times as
heavily when retraining (walk-forward 2014–2025: 1.11 to 1.06 places, better in 10 of 12 seasons). The full
write-up, with the chair's stated reasons and what was tried and rejected, is `reports/diagnosis_2026-11-10.md`.

## Known limitations

- The committee is thirteen people. The model's typical miss is one to two places, and about two places
  for ranks 17–25. It has missed the exact Selection Day field in 4 of 12 seasons.
- Only two seasons (2024, 2025) exist with a 12-team field and one-plus under the 2025 metrics. Anything
  specific to this era, including how three-loss teams and Group of Six teams are treated, rests on little data.
- On the Nov 3, 2026 ranking the model sat too close to the AP poll: two-loss SEC teams too low, teams with
  soft schedules too high. The 2026 "Strength of Conference Enhancement" is not separately learnable yet.
- Simulated ratings are updated with a one-step approximation of the full rating refit.
- A what-if "X loses its title game" applies only in simulations where X reaches the game.

## CollegeFootballData (connected Nov 2026)

With `CFBD_API_KEY` in `.env`, `cfp update` also pulls CFBD games, betting lines, Elo, SP+, SRS and play-by-play
stats (about 14 calls a week against a 1,000-a-month allowance; every response is cached). What it is used for:

- **Score check.** CFBD is the primary score source and ESPN the cross-check. CFBD lists some neutral-site games
  with home and away reversed, so games are aligned by team id first. After alignment 2 of 11,057 games differ,
  each by one point; they are in the `score_conflicts` table.
- **Betting lines in the simulation.** Where a line is posted for a remaining game, its expected margin is 70% line
  and 30% rating.
- **Not used as committee-model inputs.** Market rating, Elo, points per play, success rate, record against the
  spread and CFBD's postgame win probability were each tested and none improved the ranking model
  (`reports/model_review_2026-11.md`). SP+, SRS and FPI are end-of-season values and cannot be used for past weeks.

Without the key everything still runs on ESPN data alone, minus the two uses above.

Each `cfp update` also archives ESPN's FPI / strength-of-record table to `data/snapshots/espn_fpi_weekly.csv`,
because ESPN serves only current values. Commit that file with the weekly update. It is not a model input yet.

## Model review, November 2026

After the first live week every idea for improving the model was tested on a production-style walk-forward
(`cfp/committee/walkforward.py`, results in `reports/walkforward_metrics.csv`). Average miss went from 1.21 to
1.10 places, better in 11 of 12 seasons; the simulation's carry-over model from 1.64 to 1.43. Kept: heavier weight
on the current season's earlier rankings, a 70/30 ensemble mix, the polls' "others receiving votes", and betting
lines in the simulation. Rejected, with numbers: committee-membership effects (chair's conference, members'
conferences, a team's own AD), all the CFBD strength and efficiency inputs, and several tuning changes.
Full write-up: `reports/model_review_2026-11.md`. The committee roster by season is `data/committee_roster.csv`.

## Every assumption and proxy

**Data**
- All FCS opponents are one pooled pseudo-team in the ratings.
- "Opponent ranked at kickoff" is ESPN's displayed rank (AP before the first committee ranking, CFP after).
- "Currently ranked" opponents are the top 25 by a model-free blend of strength of record and rating, to
  avoid circularity; it is not the committee's own list.
- ESPN gives no date on 2014 rankings; those dates are derived from the schedule.
- FPI and SP+ exist only as season values, not weekly snapshots, so they are not training features. The
  simulation blends this model's rating 50/50 with the current ESPN FPI; that weight is an assumption.
- Chair transcripts cover Nov 2020 onward only. `data/committee_statements.csv` is tagged by keyword
  matching, not by hand.
- Pac-12 "flex week" pairings in `data/schedule_overrides.yaml` are the tentative ones published in February.
- Title-game dates and sites come from FBSchedules, not from each league's site.

**Committee metrics (private; reconstructed)**
- Strength of record: wins minus the wins an average top-10 team would expect against the same schedule.
- Record strength (2025): sum over wins of −log P(top-10 team wins) minus sum over losses of
  −log P(top-10 team loses), which rewards beating strong teams and punishes losing to weak ones.
- Enhanced schedule strength (2025): mean difficulty of the four toughest games.
- Strength of Conference Enhancement (2026): league-opponent rating relative to conference peers.
- Game control: average win probability at the end of each quarter from quarter scores, not play-by-play.
- Postgame win expectancy: a function of margin until CFBD's version is ingested.
- Margin cap of 24 points, from the executive director's Nov 25, 2025 statement.

**Committee behaviour**
- QB availability has no learned weight (no historical data). `data/availability.yaml` applies an assumed
  score shift and is empty by default.
- Recusals are recorded (`docs/committee_protocol.md`) but not modelled; no ballot data exists.
- Notre Dame ranked in the top 12 is treated as protected from being bumped by automatic qualifiers.
- Bowl choices by the top three seeds are assigned by a geographic preference table.
- Playoff games after the first round are simulated as neutral-site.

**Conference tiebreakers** (`cfp/conferences/tiebreak.py`; usage counts appear in each weekly report)

| League | Status |
|---|---|
| SEC | Official 2024 procedure; no 2026 revision found. "Capped relative scoring margin" → own capped league margin. |
| Big 12 | Official policy PDF. SportSource rating → own rating. |
| Big Ten | Steps 1–4 from the league's own 2025 worked examples; steps 5–6 from press. SportSource rating → own rating. |
| ACC | New 2026 policy as reproduced by FBSchedules. SportSource "Team Success Ranking" → own strength of record. This proxy decides ACC ties often because the rule goes straight to it after head-to-head. |
| Mountain West | Pre-2026 rules as quoted by SI. CFP rank / computer composite → own rating + strength of record. |
| American, Conference USA | Unverified; press summaries only. Same composite proxy. |
| MAC, Pac-12, Sun Belt | Unverified; generic ladder (head-to-head, common opponents, down the standings, opponents' record, rating, draw). Sun Belt uses its two divisions. |
| All | Random draws are simulated with the seeded generator. |

Checked against history: the engine reproduces 104 of the 110 actual title-game pairings from 2014–2025. The
six misses are a COVID withdrawal (Pac-12 2020), a postseason-ineligible team (Sun Belt 2022, 2023), a
computer-composite tie (Mountain West 2025), a division tie (Mountain West 2016) and the 2025 ACC, where
the old rule sent Duke and the new 2026 rule would have sent Miami.
