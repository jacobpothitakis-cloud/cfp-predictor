# PLAN — CFP committee ranking predictor

Target: the CFP selection committee's weekly Top 25 and the final 12-team field/bracket.
Not a power rating. Ratings (own model, Elo, FPI, SP+, SRS) are inputs only.

## Ground rules (from the brief)

- No fabricated data. Every number comes from a fetched source; a failed fetch stops that step and is reported.
- Every rule is verified against collegefootballplayoff.com first and written to `docs/committee_protocol.md` with a URL.
- Every API response is cached on disk under `data/cache/` and never re-fetched unless forced.
- Keys live in `.env` (git-ignored) and are read from the environment.
- Tests run at the end of each phase before the next starts.

## Layout

```
cfp/
  config.py            paths, env, constants
  ingest/              http cache, cfbd.py, espn.py, transcripts.py, load.py (DuckDB)
  ratings/             own opponent-adjusted rating, blend, win-prob + score model
  features/            résumé features per team-week, pairwise features
  committee/           pairwise logistic, LambdaRank, movement model, explain, backtest
  conferences/         per-conference tiebreakers, title-game participants
  simulate/            Monte Carlo season + playoff, leverage, what-if
  app/                 Streamlit dashboard
  cli.py               `cfp update|predict|simulate|whatif|backtest|team`
data/                  cache/, cfp.duckdb, availability.yaml, committee_statements.csv
docs/                  committee_protocol.md, sources/ (saved text of each cited page)
reports/               backtest report, weekly reports, 2026 prediction log
tests/
```

## Phases

1. **Research** → `docs/committee_protocol.md`. Resolve: G6 bid (champion vs team), seeding, top-3 bowl
   choice, voting procedure, roster and recusals, 2025 and 2026 metric changes, chair transcripts.
2. **Ingest** → DuckDB. CFBD primary (games 2014–2026 incl. FCS opponents, line scores, lines, rankings,
   polls, ratings, advanced stats, team/conference by season, 2026 schedule). ESPN secondary for same-day
   scores and score reconciliation. Integrity tests.
3. **Team strength**: walk-forward opponent-adjusted rating (ridge on capped margin with a preseason prior),
   blended with market/Elo where it helps; home field; rating diff → win prob and score distribution.
   Validated on held-out seasons and against closing lines.
4. **Résumé features** per team per week, strictly as-of that week. Unit tests.
5. **Committee model**: pairwise logistic (interpretable) and LightGBM LambdaRank; level and movement;
   era effects; pod-style neighbour features. Leave-one-season-out backtest against baselines.
   **Stop and show the backtest report.**
6. **Conference races + simulation**: published 2026 tiebreakers per conference, ≥10,000 seeded runs,
   selection and seeding rules, bracket, playoff sim, leverage.
7. **Interface**: CLI, Streamlit dashboard, weekly Markdown report.
8. **Run the current week**: predict the Nov 10, 2026 ranking and projected field with explanations.

## Known constraints going in

- CFBD free tier has a monthly call cap; ingest is planned at roughly a dozen calls per season.
- CFBD serves SP+ and FPI as season values, not week-by-week snapshots. Historical weekly snapshots of
  those two ratings may not be obtainable; if so the training features use ratings that are computable
  as-of each week (own rating, Elo, SRS-style) and the gap is stated in the README.
- The committee's data portal (SportSource Analytics) is private. "Record strength", the adjusted schedule
  strength and the conference-strength number are reconstructions, labelled as such.
