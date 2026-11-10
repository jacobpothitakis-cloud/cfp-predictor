# Committee model backtest

Leave-one-season-out over 2014–2025: for each season the models are trained on every *other* season's rankings and then predict each of that season's 72 weekly committee rankings in total, using only games, polls and ratings that existed before each release. Ratings are the walk-forward team-strength model, so there is no look-ahead in the features.

Rank error is measured on the 25 teams the committee actually ranked; a team the model leaves outside its top 25 is scored at the rank the model gave it (26, 31, ...).

## 1. Overall, all weekly rankings

| Model | MAE (all 25) | MAE (top 12) | Spearman | Within 1 | Within 2 | Top-4 exact | Top-12 exact | Top-12 overlap |
|---|---|---|---|---|---|---|---|---|
| Ensemble (+polls, last week) | 1.30 | 0.78 | 0.97 | 0.69 | 0.84 | 0.65 | 0.51 | 0.95 |
| Pairwise logistic (+polls, last week) | 1.31 | 0.83 | 0.97 | 0.69 | 0.84 | 0.68 | 0.50 | 0.95 |
| LambdaRank (+polls, last week) | 1.61 | 0.88 | 0.97 | 0.66 | 0.81 | 0.68 | 0.40 | 0.94 |
| Ensemble (carry-over from last week, no polls) | 1.65 | 1.01 | 0.96 | 0.61 | 0.77 | 0.60 | 0.40 | 0.94 |
| AP poll | 1.78 | 0.92 | 0.96 | 0.60 | 0.77 | 0.64 | 0.46 | 0.95 |
| Prev rank + results | 1.85 | 0.96 | 0.96 | 0.61 | 0.76 | 0.71 | 0.44 | 0.94 |
| Ensemble (carry-over from first ranking) | 1.85 | 1.05 | 0.95 | 0.57 | 0.73 | 0.60 | 0.35 | 0.93 |
| Pairwise logistic (résumé only) | 2.17 | 1.50 | 0.94 | 0.49 | 0.67 | 0.33 | 0.18 | 0.91 |
| Ensemble (résumé only) | 2.21 | 1.43 | 0.94 | 0.50 | 0.67 | 0.39 | 0.22 | 0.92 |
| LambdaRank (résumé only) | 2.66 | 1.58 | 0.92 | 0.46 | 0.62 | 0.25 | 0.19 | 0.91 |
| First ranking adjusted for new losses | 3.11 | 1.55 | 0.91 | 0.47 | 0.61 | 0.67 | 0.28 | 0.91 |
| Strength of record | 4.39 | 2.55 | 0.83 | 0.30 | 0.45 | 0.32 | 0.03 | 0.84 |

*Résumé only* uses no polls and no previous committee ranking. *+polls, last week* adds the AP and Coaches polls released two days before and last week's committee ranking (the AP poll stands in for the first release of a season). *Carry-over* models use an earlier committee ranking from the same season but no polls: these are what the season simulation uses, because future polls do not exist.

## 2. By season: mean absolute rank error

| season | Ensemble (+polls, last week) | AP poll | Prev rank + results | Strength of record | Ensemble (résumé only) | Ensemble (carry-over from first ranking) |
|---|---|---|---|---|---|---|
| 2014 | 1.40 | 1.66 | 1.95 | 3.49 | 2.35 | 1.85 |
| 2015 | 1.51 | 1.91 | 2.35 | 3.89 | 2.27 | 1.88 |
| 2016 | 1.68 | 2.26 | 2.05 | 4.38 | 2.51 | 2.21 |
| 2017 | 1.18 | 1.73 | 1.89 | 3.65 | 1.77 | 1.67 |
| 2018 | 1.35 | 1.73 | 1.89 | 4.99 | 2.25 | 1.44 |
| 2019 | 0.93 | 1.53 | 1.71 | 4.51 | 1.73 | 1.63 |
| 2020 | 1.50 | 2.74 | 2.06 | 9.92 | 2.90 | 2.40 |
| 2021 | 1.52 | 2.42 | 1.83 | 4.41 | 2.39 | 1.79 |
| 2022 | 0.89 | 1.28 | 1.71 | 3.29 | 1.63 | 1.54 |
| 2023 | 1.07 | 1.69 | 1.53 | 4.94 | 2.24 | 2.10 |
| 2024 | 1.36 | 1.03 | 1.53 | 3.12 | 2.59 | 2.26 |
| 2025 | 1.20 | 1.58 | 1.67 | 3.16 | 1.93 | 1.53 |
| All | 1.30 | 1.78 | 1.85 | 4.39 | 2.21 | 1.85 |

**The model loses to the AP poll in 2024.** See section 9.


By release number within the season (1 = first ranking, when there is no previous committee ranking):

| release | AP poll | Ensemble (+polls, last week) | Prev rank + results |
|---|---|---|---|
| 1 | 2.01 | 1.73 | 2.41 |
| 2 | 1.84 | 1.42 | 1.89 |
| 3 | 1.89 | 1.30 | 2.02 |
| 4 | 1.53 | 1.05 | 1.52 |
| 5 | 1.45 | 1.09 | 1.58 |
| 6 | 1.96 | 1.16 | 1.62 |
| 7 | 2.12 | 1.52 | 2.08 |

## 3. Selection Day rankings only (12 seasons)

| Model | MAE (all 25) | Within 1 | Top-4 exact | Top-12 exact | Field correct | Field overlap | Seeds correct |
|---|---|---|---|---|---|---|---|
| Ensemble (+polls, last week) | 1.22 | 0.69 | 0.58 | 0.67 | 0.67 | 0.91 | 0.66 |
| Pairwise logistic (+polls, last week) | 1.22 | 0.74 | 0.50 | 0.67 | 0.58 | 0.90 | 0.72 |
| Ensemble (carry-over from last week, no polls) | 1.46 | 0.66 | 0.67 | 0.42 | 0.75 | 0.95 | 0.52 |
| LambdaRank (+polls, last week) | 1.55 | 0.65 | 0.58 | 0.58 | 0.67 | 0.91 | 0.65 |
| Prev rank + results | 1.68 | 0.58 | 0.33 | 0.33 | 0.33 | 0.84 | 0.53 |
| Ensemble (carry-over from first ranking) | 1.92 | 0.59 | 0.75 | 0.33 | 0.75 | 0.95 | 0.55 |
| Pairwise logistic (résumé only) | 1.96 | 0.56 | 0.42 | 0.17 | 0.50 | 0.88 | 0.44 |
| Ensemble (résumé only) | 2.02 | 0.54 | 0.50 | 0.42 | 0.58 | 0.91 | 0.47 |
| AP poll | 2.10 | 0.51 | 0.25 | 0.42 | 0.33 | 0.81 | 0.48 |
| LambdaRank (résumé only) | 2.46 | 0.50 | 0.33 | 0.33 | 0.33 | 0.84 | 0.35 |
| First ranking adjusted for new losses | 3.78 | 0.37 | 0.42 | 0.08 | 0.42 | 0.83 | 0.52 |
| Strength of record | 4.09 | 0.35 | 0.50 | 0.00 | 0.50 | 0.87 | 0.57 |

*Field correct*: the playoff field implied by the predicted ranking equals the field implied by the committee's ranking under that season's rules (top 4 through 2023; five champions plus seven at-large in 2024–25). *Seeds correct*: share of seed lines filled by the right team.

| season | Field correct — AP poll | Field correct — Ensemble (+polls, last week) | Field correct — Ensemble (carry-over from first ranking) | Seeds correct — AP poll | Seeds correct — Ensemble (+polls, last week) | Seeds correct — Ensemble (carry-over from first ranking) |
|---|---|---|---|---|---|---|
| 2014 | 0.00 | 0.00 | 1.00 | 0.25 | 0.75 | 1.00 |
| 2015 | 0.00 | 1.00 | 1.00 | 0.50 | 0.50 | 0.50 |
| 2016 | 1.00 | 1.00 | 1.00 | 0.50 | 0.50 | 0.50 |
| 2017 | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 | 0.00 |
| 2018 | 0.00 | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 |
| 2019 | 0.00 | 1.00 | 1.00 | 0.75 | 1.00 | 0.00 |
| 2020 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 0.50 |
| 2021 | 1.00 | 1.00 | 1.00 | 0.25 | 0.50 | 1.00 |
| 2022 | 0.00 | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 |
| 2023 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 |
| 2024 | 1.00 | 1.00 | 1.00 | 0.67 | 0.33 | 0.17 |
| 2025 | 0.00 | 0.00 | 0.00 | 0.33 | 0.33 | 0.42 |

## 4. Is the ranking a clean sheet each week? No.

Adding last week's rank and the current polls cuts the error from 2.21 to 1.30 places. In the pairwise model the weight on last week's committee position is 8.73 log-odds per 25 places (i.e. per full length of the ranking), on the AP poll 5.26 and on the Coaches poll 3.19; a loss this week by a previously ranked team is worth -2.41 on top of what the loss does to the résumé features. At a season's first release, with no previous committee ranking, the AP poll's weight rises by 1.40 and the Coaches poll's by 0.46. The committee says each ranking starts from scratch; the data say a team's previous position carries real weight after controlling for the résumé.

Carry-over decays with time. Weight on an earlier committee ranking, by how many releases ago it was: 1 release 9.17, 2 releases 7.85, 3+ releases 7.74; each loss since then -0.88 (plus -1.98 × earlier position).

## 5. What the résumé-only pairwise model learned

Weights are log-odds of being ranked ahead per one unit of the feature (z-scored features: per standard deviation across FBS that week; counts: per game). Base = 2014–2023; the next two columns add the learned shift for the 12-team era (2024+) and for 2025+ (new metrics).

| Feature | Base | 12-team era | 2025+ |
|---|---|---|---|
| fcs_losses | -2.07 |  |  |
| rating | 1.86 | 2.74 | 2.92 |
| postgame_wp | 1.51 |  |  |
| g6_wae | -1.51 |  |  |
| win_credit | -1.24 |  |  |
| best_win | 1.24 | 1.16 | 1.40 |
| is_g6 | 1.02 | 2.02 | 1.98 |
| margin_cap | -0.93 | -0.91 | -1.48 |
| losses_sq | -0.92 |  |  |
| ccg_loss | 0.82 |  |  |
| losses | -0.81 | -1.44 | -1.47 |
| ccg_played | 0.71 |  |  |
| played_week | -0.68 |  |  |
| best_win3 | 0.68 |  |  |
| last3_value | 0.66 |  |  |
| is_power | 0.65 |  |  |
| record_strength | 0.65 | 0.77 | 0.71 |
| conf_strength | 0.59 | 0.08 | 1.20 |
| w_top10 | 0.56 |  |  |
| w_power | 0.54 |  |  |
| loss_unranked | -0.50 | -0.50 | -0.52 |
| l_top10 | 0.48 |  |  |
| l500 | -0.47 |  |  |
| w_ranked_g | 0.46 |  |  |
| won_week | 0.43 |  |  |
| wins | 0.42 |  |  |
| loss_penalty | -0.42 |  |  |
| best_win2 | 0.38 |  |  |

### How priorities shifted by era

| Feature | Base | 12-team era | 2025+ | Shift 2014–23 → 2025+ |
|---|---|---|---|---|
| rating | 1.86 | 2.74 | 2.92 | 1.06 |
| is_g6 | 1.02 | 2.02 | 1.98 | 0.96 |
| sos | 0.18 | 0.09 | 0.85 | 0.67 |
| losses | -0.81 | -1.44 | -1.47 | -0.66 |
| conf_strength | 0.59 | 0.08 | 1.20 | 0.62 |
| margin_cap | -0.93 | -0.91 | -1.48 | -0.55 |
| sos_top4 | -0.22 | -0.36 | -0.77 | -0.55 |
| worst_loss | -0.02 | 0.59 | -0.41 | -0.39 |
| champion | -0.11 | -0.30 | -0.48 | -0.36 |
| wae | -0.24 | -0.12 | 0.08 | 0.32 |
| best_win | 1.24 | 1.16 | 1.40 | 0.16 |
| w500 | 0.31 | 0.41 | 0.16 | -0.15 |
| conf_sos_rel | 0.07 | 0.45 | -0.01 | -0.08 |
| record_strength | 0.65 | 0.77 | 0.71 | 0.07 |
| loss_unranked | -0.50 | -0.50 | -0.52 | -0.02 |

Only two seasons (12 rankings) exist in the 12-team era and one-plus under the 2025 metrics, so these shifts are estimated from little data and are regularised toward zero.

### Which groups of criteria carry predictive weight (ablation)

Individual weights above are entangled because many features measure overlapping things (rating, margin and postgame win expectancy; win credit and best win). A cleaner test: remove one whole group from the résumé-only model, refit, and see how much the leave-one-season-out error rises.

| Removed | MAE | MAE top 12 | Top-12 overlap | Within 1 | MAE change |
|---|---|---|---|---|---|
| Quality of wins | 2.486 | 1.637 | 0.899 | 0.463 | 0.319 |
| Team strength and dominance (rating, margin, game control) | 2.406 | 1.656 | 0.896 | 0.453 | 0.239 |
| Recency and this week's result | 2.217 | 1.521 | 0.912 | 0.499 | 0.050 |
| Losses and record | 2.209 | 1.534 | 0.917 | 0.479 | 0.042 |
| Head-to-head and common opponents | 2.207 | 1.525 | 0.916 | 0.482 | 0.041 |
| Brand (preseason AP rank) | 2.169 | 1.502 | 0.917 | 0.496 | 0.003 |
| (nothing removed) | 2.167 | 1.502 | 0.913 | 0.494 | 0.000 |
| Margin and game control only | 2.165 | 1.506 | 0.916 | 0.498 | -0.002 |
| Power / Group of Six / independent status | 2.164 | 1.508 | 0.916 | 0.497 | -0.003 |
| Schedule strength, strength of record, record strength | 2.163 | 1.486 | 0.917 | 0.498 | -0.004 |
| Quality of losses | 2.149 | 1.450 | 0.913 | 0.506 | -0.017 |
| Conference title and conference strength | 2.136 | 1.493 | 0.917 | 0.502 | -0.031 |

A group whose removal costs nothing is either unused by the committee or already implied by the other groups; it is not evidence the committee ignores it.

### Head-to-head and common opponents (pod effect)

Head-to-head winner bonus: 1.80 log-odds (résumé-only model), 2.12 (model with polls), applied with a closeness kernel of width 4 / 4 log-odds, so it fades once two teams are far apart. Common-opponent edge: 0.25 / 0.37 per net game.

## 6. Selection Day behaviour, 2014–2025

| Ranked team's last weekend | Teams | Mean places moved (+ = up) | Share up | Share down | Share unchanged |
|---|---|---|---|---|---|
| idle | 162 | 0.19 | 0.37 | 0.17 | 0.46 |
| lost title game | 66 | -2.05 | 0.03 | 0.82 | 0.15 |
| won title game | 72 | 1.31 | 0.65 | 0.03 | 0.32 |

| Top-12 teams only | Teams | Mean places moved | Share down |
|---|---|---|---|
| idle | 60 | -0.12 | 0.32 |
| lost title game | 32 | -2.12 | 0.81 |
| won title game | 52 | 1.02 | 0.04 |

Idle teams do get passed: 28 of 162 idle ranked teams finished lower than they started the weekend.

Title-game losses that knocked a team out of the playoff cut line (top 4 through 2023, top 12 after):

- 2015 Iowa: No. 4 before its title-game loss, No. 5 after (out of the top 4).
- 2017 Auburn: No. 2 before its title-game loss, No. 7 after (out of the top 4).
- 2017 Wisconsin: No. 4 before its title-game loss, No. 6 after (out of the top 4).
- 2018 Georgia: No. 4 before its title-game loss, No. 5 after (out of the top 4).
- 2019 Georgia: No. 4 before its title-game loss, No. 5 after (out of the top 4).
- 2022 USC: No. 4 before its title-game loss, No. 10 after (out of the top 4).
- 2023 Georgia: No. 1 before its title-game loss, No. 6 after (out of the top 4).

## 7. Uncertainty calibration

Each prediction ships with a rank distribution: Gaussian noise on team scores, re-ranked many times. The noise scale is set so a typical draw sits as far from the point prediction as the committee's actual ranking did in the backtest. Coverage is then checked on the same held-out predictions.

| Model | Noise σ (log-odds) | Observed MAE | 50% interval covers | 80% interval covers | 95% interval covers |
|---|---|---|---|---|---|
| Ensemble (+polls, last week) | 1.46 | 1.30 | 0.71 | 0.88 | 0.96 |
| Ensemble (carry-over from first ranking) | 1.73 | 1.85 | 0.64 | 0.86 | 0.96 |

Intervals are on integer ranks, so coverage above the nominal level is expected for teams whose rank is nearly certain. Calibration of simulated *playoff odds* is reported with the simulation.

## 8. 2026 so far (true out-of-sample)

**Release 1 (2026-11-03),** predicted with a model trained only on 2014–2025: MAE 1.88 (AP poll 2.12), within one spot 40%, within two 68%, top-4 overlap 4/4, top-12 overlap 12/12, top-25 overlap 24/25.

| Team | Actual | Predicted | AP | Record | Error |
|---|---|---|---|---|---|
| Miami | 1 | 2 | 1 | 8-0 | +1 |
| Georgia | 2 | 1 | 2 | 7-1 | -1 |
| Utah | 3 | 3 | 3 | 8-0 | +0 |
| Indiana | 4 | 4 | 6 | 8-1 | +0 |
| Alabama | 5 | 10 | 9 | 6-2 | +5 |
| Kentucky | 6 | 8 | 8 | 7-1 | +2 |
| Texas | 7 | 9 | 10 | 6-2 | +2 |
| UCLA | 8 | 5 | 5 | 8-0 | -3 |
| BYU | 9 | 6 | 7 | 7-1 | -3 |
| Ole Miss | 10 | 12 | 12 | 6-2 | +2 |
| Notre Dame | 11 | 7 | 4 | 7-1 | -4 |
| Ohio State | 12 | 11 | 11 | 6-2 | -1 |
| Tennessee | 13 | 16 | 16 | 6-2 | +3 |
| LSU | 14 | 15 | 15 | 6-2 | +1 |
| Texas Tech | 15 | 13 | 13 | 7-1 | -2 |
| Wake Forest | 16 | 14 | 14 | 7-1 | -2 |
| Houston | 17 | 19 | 17 | 6-2 | +2 |
| Missouri | 18 | 18 | 19 | 6-2 | +0 |
| Iowa | 19 | 20 | 21 | 6-2 | +1 |
| USC | 20 | 17 | 18 | 7-2 | -3 |
| South Carolina | 21 | 21 | 23 | 5-3 | +0 |
| Arizona | 22 | 24 | 24 | 6-2 | +2 |
| Oregon | 23 | 26 | NR | 5-3 | +3 |
| Florida | 24 | 23 | 25 | 5-3 | -1 |
| Boise State | 25 | 22 | 20 | 7-1 | -3 |
| Northwestern | NR | 25 | NR | 6-2 |  |

## 9. Biggest misses in the backtest

| Season | Release | Team | Actual | Predicted | Error |
|---|---|---|---|---|---|
| 2025 | 5 | Houston | 21 | 38 | 17 |
| 2021 | 1 | San Diego State | 24 | 38 | 14 |
| 2018 | 1 | NC State | 21 | 35 | 14 |
| 2014 | 4 | Minnesota | 25 | 35 | 10 |
| 2014 | 7 | Minnesota | 25 | 35 | 10 |
| 2021 | 2 | San Diego State | 22 | 32 | 10 |
| 2021 | 1 | Mississippi State | 17 | 27 | 10 |
| 2014 | 3 | Minnesota | 25 | 34 | 9 |
| 2016 | 2 | USC | 20 | 29 | 9 |
| 2021 | 2 | Utah | 24 | 33 | 9 |
| 2018 | 1 | Texas A&M | 20 | 29 | 9 |
| 2016 | 4 | Navy | 25 | 34 | 9 |
| 2020 | 3 | Missouri | 25 | 34 | 9 |
| 2016 | 5 | Utah | 20 | 28 | 8 |
| 2019 | 3 | Iowa State | 22 | 30 | 8 |

Error by position (the voting pools):

| Actual rank band | MAE |
|---|---|
| 1–4 | 0.40 |
| 5–8 | 0.86 |
| 9–12 | 1.07 |
| 13–16 | 1.30 |
| 17–25 | 1.99 |

Group of Five/Six teams: mean signed error +0.23 places, MAE 1.93 (202 team-weeks). Power-conference teams: MAE 1.22.

## 10. What the chair says versus what the data support

`data/committee_statements.csv` holds 1476 tagged sentences from 35 committee media calls (2020–2026; earlier transcripts were not obtainable). Mentions per season:

| factor | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | total |
|---|---|---|---|---|---|---|---|---|
| quality_wins | 19 | 22 | 28 | 11 | 16 | 29 | 12 | 137 |
| loss_quality | 13 | 20 | 14 | 5 | 46 | 26 | 7 | 131 |
| strength_of_schedule | 4 | 15 | 5 | 4 | 22 | 25 | 5 | 80 |
| conference_championship | 6 | 9 | 5 | 12 | 18 | 23 | 3 | 76 |
| eye_test | 15 | 16 | 4 | 5 | 11 | 12 | 7 | 70 |
| recency | 10 | 10 | 7 | 8 | 9 | 17 | 5 | 66 |
| head_to_head | 6 | 18 | 3 | 5 | 8 | 12 | 3 | 55 |
| body_of_work | 5 | 1 | 19 | 2 | 14 | 7 | 4 | 52 |
| road_wins | 1 | 6 | 1 | 2 | 14 | 11 | 10 | 45 |
| complete_team | 5 | 13 | 8 | 1 | 2 | 10 | 2 | 41 |
| availability | 5 | 2 | 5 | 4 | 6 | 9 | 5 | 36 |
| efficiency_stats | 4 | 6 | 6 | 3 | 3 | 11 | 0 | 33 |
| game_control | 2 | 5 | 1 | 1 | 7 | 12 | 2 | 30 |
| group_of_five | 0 | 2 | 1 | 5 | 0 | 7 | 4 | 19 |
| common_opponents | 2 | 6 | 1 | 0 | 1 | 4 | 0 | 14 |
| margin | 1 | 0 | 2 | 0 | 1 | 9 | 0 | 13 |
| line_of_scrimmage | 2 | 0 | 1 | 1 | 2 | 4 | 1 | 11 |
| record_strength | 0 | 0 | 0 | 0 | 2 | 7 | 0 | 9 |
| conference_strength | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |

How each stated factor shows up in the fitted résumé-only model (weight, base era):

| Stated factor | Mentions | Model features and weights |
|---|---|---|
| quality_wins | 137 | win_credit -1.24; best_win +1.24; best_win2 +0.38; w_ranked +0.32; w_top10 +0.56; w500 +0.31 |
| loss_quality | 131 | losses -0.81; loss_penalty -0.42; worst_loss -0.02; loss_unranked -0.50; loss_sub500 -0.34 |
| strength_of_schedule | 80 | sos +0.18; sos_top4 -0.22 |
| record_strength | 9 | wae -0.24; sor_z -0.25; record_strength +0.65 |
| head_to_head | 55 | h2h +1.80 |
| common_opponents | 14 | common +0.25 |
| conference_championship | 76 | champion -0.11; ccg_loss +0.82 |
| game_control | 30 | game_control +0.10 |
| margin | 13 | margin_cap -0.93; margin_loss_cap +0.31 |
| eye_test | 70 | rating +1.86 |
| road_wins | 45 | road_w_ranked -0.17; w_road -0.01 |
| recency | 66 | last3_value +0.66; trend +0.30; recent_losses -0.09; days_since_loss -0.21 |
| conference_strength | 1 | conf_strength +0.59; conf_sos_rel +0.07 |
| group_of_five | 19 | is_g6 +1.02; g6_wae -1.51 |
| availability | 36 | qb_out +0.00; qb_back +0.00 |
## 11. Diagnosis of the misses

**2026 release 1 (out of sample).** Every large miss points the same way: the model sat too close to the AP
poll and the committee did not. Alabama (actual 5, predicted 10, AP 9), Texas (7 / 9 / 10) and Tennessee
(13 / 16 / 16) are two-loss SEC teams the committee placed three to five spots above the polls; Notre Dame
(11 / 7 / AP 4), UCLA (8 / 5 / 5) and BYU (9 / 6 / 7) are zero- and one-loss teams it placed below them. The
chair's stated reasons on Nov 3 match the direction exactly: Alabama played "seven of their eight games against
teams that have a .500 record or better", Notre Dame had "one win against a team with a .500 record or better",
and UCLA's "schedule isn't as strong as other comparable teams". The model has these features (`w500`, `sos`,
`best_win`, `conf_strength`) and its 2025+ era terms already raise the weight on schedule and conference
strength, but one season of data under the 2025 metrics was not enough to move it as far as the committee
moved in 2026. The production model is now retrained with this ranking included, and the 2026 "Strength of
Conference Enhancement" is the most likely reason the shift is larger this year. Expect the same bias, smaller,
in the next prediction: two-loss SEC teams slightly too low, soft-schedule unbeatens slightly too high.

**2024 (the one season the AP poll beat the model).** 2024 was the first 12-team year and, when it is the
held-out season, the model has only 2025 to learn the new era from. Its misses are concentrated in Group of
Five teams the committee ranked much higher than any earlier committee would have: Boise State (9–12 actual,
predicted 12–16 all season), UNLV, Tulane and Army. The committee also tracked the AP poll unusually closely
that year (AP MAE 1.03 against 1.5–2.7 in other seasons). With 2024 in the training set the `is_g6` weight
roughly doubles in the 12-team era (section 5), which is the correction.

**The bottom of the ranking.** Error is 0.4 places for ranks 1–4 and 2.0 for ranks 17–25. The largest single
misses (Houston 2025, San Diego State 2021, NC State 2018, Minnesota 2014) are all teams ranked 20–25 that the
model had in the low 30s. The last voting rounds pick three teams at a time from a long list of similar
résumés; this is where the committee is least predictable and where the model's uncertainty bands are widest.

**Selection Day fields missed** (primary model, 4 of 12 seasons). 2014: the model had TCU 4th and Ohio State
6th; the committee had Ohio State 4th and TCU 6th. 2017: the model kept Wisconsin (actual 6th, after losing its
title game) in the top four; the committee put Alabama 4th. 2023: the model had Florida State 3rd and Georgia
4th; the committee had Texas 3rd, Alabama 4th, Florida State 5th and Georgia 6th. 2025: the model had Notre Dame
9th and Miami 12th; the committee had Miami 10th and Notre Dame 11th, which decided the last at-large spot.
In each case the model's field was one or two teams off, on a decision between teams the committee itself
ranked within two places of each other.

**What the features cannot see.** Starting-quarterback availability has zero weight because there is no
historical availability data to learn it from; it is a manual input (`data/availability.yaml`) whose effect is
an assumption, not an estimate. "Game control" is a quarter-score proxy, and postgame win expectancy is a
margin-based stand-in until the CFBD advanced stats are ingested.
