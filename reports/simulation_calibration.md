# Calibration of simulated playoff odds

Seasons 2014–2025, standing at ranking release(s) 1, 3 of each season, 1,000 simulations each, committee models trained without the season being simulated. A team counts as having made the field if the committee's actual final ranking put it in under that season's rules (top 4 through 2023; 12 teams in 2024–25).

| Simulated chance | Team-seasons | Mean simulated | Actually made it |
|---|---|---|---|
| [0.0, 0.02) | 1087 | 0.001 | 0.001 |
| [0.02, 0.1) | 93 | 0.051 | 0.011 |
| [0.1, 0.25) | 76 | 0.175 | 0.224 |
| [0.25, 0.5) | 70 | 0.358 | 0.414 |
| [0.5, 0.75) | 62 | 0.610 | 0.516 |
| [0.75, 0.9) | 29 | 0.832 | 0.862 |
| [0.9, 0.98) | 14 | 0.932 | 1.000 |
| [0.98, 1.0) | 9 | 0.993 | 1.000 |

Brier score 0.0343 (always guessing the base rate: 0.0810; lower is better). Brier skill 0.58.

- From release 1: Brier 0.0384; teams given ≥50% that made it 38/54; teams given <10% that made it 2/581.
- From release 3: Brier 0.0302; teams given ≥50% that made it 42/60; teams given <10% that made it 0/599.

Limits: only 12 seasons, 10 of them with a four-team field, so the high-probability bins hold few teams; historical simulations use this model's own rating only (no FPI) and approximate each league's historical tiebreakers and title-game sites.
