# Committee model backtest

Leave-one-season-out over 2014–2025: for each season the models are trained on every *other* season's rankings and then predict each of that season's 72 weekly committee rankings in total, using only games, polls and ratings that existed before each release. Ratings are the walk-forward team-strength model, so there is no look-ahead in the features.

Rank error is measured on the 25 teams the committee actually ranked; a team the model leaves outside its top 25 is scored at the rank the model gave it (26, 31, ...).

## 1. Overall, all weekly rankings

| Model | MAE (all 25) | MAE (top 12) | Spearman | Within 1 | Within 2 | Top-4 exact | Top-12 exact | Top-12 overlap |
|---|---|---|---|---|---|---|---|---|
| Ensemble (+polls, last week) | 1.21 | 0.73 | 0.97 | 0.72 | 0.85 | 0.69 | 0.51 | 0.95 |
| Pairwise logistic (+polls, last week) | 1.26 | 0.79 | 0.97 | 0.70 | 0.85 | 0.71 | 0.54 | 0.95 |
| LambdaRank (+polls, last week) | 1.51 | 0.89 | 0.97 | 0.67 | 0.81 | 0.65 | 0.42 | 0.95 |
| Ensemble (carry-over from last week, no polls) | 1.65 | 1.01 | 0.96 | 0.60 | 0.78 | 0.62 | 0.40 | 0.94 |
| AP poll | 1.78 | 0.92 | 0.96 | 0.60 | 0.77 | 0.64 | 0.46 | 0.95 |
| Ensemble (carry-over from first ranking) | 1.84 | 1.05 | 0.95 | 0.57 | 0.74 | 0.63 | 0.35 | 0.93 |
| Prev rank + results | 1.85 | 0.96 | 0.96 | 0.61 | 0.76 | 0.71 | 0.44 | 0.94 |
| Pairwise logistic (résumé only) | 2.15 | 1.48 | 0.94 | 0.50 | 0.68 | 0.36 | 0.21 | 0.92 |
| Ensemble (résumé only) | 2.19 | 1.42 | 0.94 | 0.50 | 0.67 | 0.42 | 0.24 | 0.92 |
| LambdaRank (résumé only) | 2.64 | 1.56 | 0.92 | 0.46 | 0.63 | 0.28 | 0.17 | 0.91 |
| First ranking adjusted for new losses | 3.11 | 1.55 | 0.91 | 0.47 | 0.61 | 0.67 | 0.28 | 0.91 |
| Strength of record | 4.39 | 2.55 | 0.83 | 0.30 | 0.45 | 0.32 | 0.03 | 0.84 |

*Résumé only* uses no polls and no previous committee ranking. *+polls, last week* adds the AP and Coaches polls released two days before and last week's committee ranking (the AP poll stands in for the first release of a season). *Carry-over* models use an earlier committee ranking from the same season but no polls: these are what the season simulation uses, because future polls do not exist.

## 2. By season: mean absolute rank error

| season | Ensemble (+polls, last week) | AP poll | Prev rank + results | Strength of record | Ensemble (résumé only) | Ensemble (carry-over from first ranking) |
|---|---|---|---|---|---|---|
| 2014 | 1.35 | 1.66 | 1.95 | 3.49 | 2.33 | 1.88 |
| 2015 | 1.56 | 1.91 | 2.35 | 3.89 | 2.19 | 1.85 |
| 2016 | 1.59 | 2.26 | 2.05 | 4.38 | 2.47 | 2.22 |
| 2017 | 1.08 | 1.73 | 1.89 | 3.65 | 1.77 | 1.77 |
| 2018 | 1.17 | 1.73 | 1.89 | 4.99 | 2.25 | 1.40 |
| 2019 | 0.95 | 1.53 | 1.71 | 4.51 | 1.76 | 1.65 |
| 2020 | 1.37 | 2.74 | 2.06 | 9.92 | 2.91 | 2.27 |
| 2021 | 1.38 | 2.42 | 1.83 | 4.41 | 2.33 | 1.73 |
| 2022 | 0.83 | 1.28 | 1.71 | 3.29 | 1.59 | 1.55 |
| 2023 | 1.01 | 1.69 | 1.53 | 4.94 | 2.30 | 2.11 |
| 2024 | 1.24 | 1.03 | 1.53 | 3.12 | 2.60 | 2.25 |
| 2025 | 0.95 | 1.58 | 1.67 | 3.16 | 1.84 | 1.53 |
| All | 1.21 | 1.78 | 1.85 | 4.39 | 2.19 | 1.84 |

**The model loses to the AP poll in 2024.** See section 9.


By release number within the season (1 = first ranking, when there is no previous committee ranking):

| release | AP poll | Ensemble (+polls, last week) | Prev rank + results |
|---|---|---|---|
| 1 | 2.01 | 1.62 | 2.41 |
| 2 | 1.84 | 1.36 | 1.89 |
| 3 | 1.89 | 1.16 | 2.02 |
| 4 | 1.53 | 0.95 | 1.52 |
| 5 | 1.45 | 1.01 | 1.58 |
| 6 | 1.96 | 1.11 | 1.62 |
| 7 | 2.12 | 1.40 | 2.08 |

## 3. Selection Day rankings only (12 seasons)

| Model | MAE (all 25) | Within 1 | Top-4 exact | Top-12 exact | Field correct | Field overlap | Seeds correct |
|---|---|---|---|---|---|---|---|
| Ensemble (+polls, last week) | 1.16 | 0.70 | 0.58 | 0.75 | 0.58 | 0.89 | 0.67 |
| Pairwise logistic (+polls, last week) | 1.19 | 0.73 | 0.58 | 0.75 | 0.58 | 0.91 | 0.72 |
| Ensemble (carry-over from last week, no polls) | 1.46 | 0.67 | 0.67 | 0.42 | 0.75 | 0.95 | 0.51 |
| LambdaRank (+polls, last week) | 1.47 | 0.67 | 0.50 | 0.58 | 0.58 | 0.89 | 0.56 |
| Prev rank + results | 1.68 | 0.58 | 0.33 | 0.33 | 0.33 | 0.84 | 0.53 |
| Pairwise logistic (résumé only) | 1.88 | 0.58 | 0.50 | 0.33 | 0.50 | 0.89 | 0.46 |
| Ensemble (carry-over from first ranking) | 1.90 | 0.60 | 0.75 | 0.33 | 0.75 | 0.95 | 0.55 |
| Ensemble (résumé only) | 2.01 | 0.54 | 0.50 | 0.33 | 0.58 | 0.91 | 0.49 |
| AP poll | 2.10 | 0.51 | 0.25 | 0.42 | 0.33 | 0.81 | 0.48 |
| LambdaRank (résumé only) | 2.45 | 0.50 | 0.42 | 0.25 | 0.42 | 0.86 | 0.43 |
| First ranking adjusted for new losses | 3.78 | 0.37 | 0.42 | 0.08 | 0.42 | 0.83 | 0.52 |
| Strength of record | 4.09 | 0.35 | 0.50 | 0.00 | 0.50 | 0.87 | 0.57 |

*Field correct*: the playoff field implied by the predicted ranking equals the field implied by the committee's ranking under that season's rules (top 4 through 2023; five champions plus seven at-large in 2024–25). *Seeds correct*: share of seed lines filled by the right team.

| season | Field correct — AP poll | Field correct — Ensemble (+polls, last week) | Field correct — Ensemble (carry-over from first ranking) | Seeds correct — AP poll | Seeds correct — Ensemble (+polls, last week) | Seeds correct — Ensemble (carry-over from first ranking) |
|---|---|---|---|---|---|---|
| 2014 | 0.00 | 0.00 | 1.00 | 0.25 | 0.75 | 1.00 |
| 2015 | 0.00 | 1.00 | 1.00 | 0.50 | 0.50 | 0.50 |
| 2016 | 1.00 | 1.00 | 1.00 | 0.50 | 0.50 | 0.50 |
| 2017 | 0.00 | 0.00 | 0.00 | 0.50 | 0.75 | 0.00 |
| 2018 | 0.00 | 0.00 | 1.00 | 0.75 | 0.75 | 1.00 |
| 2019 | 0.00 | 1.00 | 1.00 | 0.75 | 1.00 | 0.00 |
| 2020 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 0.50 |
| 2021 | 1.00 | 1.00 | 1.00 | 0.25 | 0.50 | 1.00 |
| 2022 | 0.00 | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 |
| 2023 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 |
| 2024 | 1.00 | 1.00 | 1.00 | 0.67 | 0.25 | 0.17 |
| 2025 | 0.00 | 0.00 | 0.00 | 0.33 | 0.50 | 0.42 |

## 4. Is the ranking a clean sheet each week? No.

Adding last week's rank and the current polls cuts the error from 2.19 to 1.21 places. In the pairwise model the weight on last week's committee position is 7.83 log-odds per 25 places (i.e. per full length of the ranking), on the AP poll 6.77 and on the Coaches poll 3.38; a loss this week by a previously ranked team is worth -2.31 on top of what the loss does to the résumé features. At a season's first release, with no previous committee ranking, the AP poll's weight rises by 1.63 and the Coaches poll's by 1.30. The committee says each ranking starts from scratch; the data say a team's previous position carries real weight after controlling for the résumé.

Carry-over decays with time. Weight on an earlier committee ranking, by how many releases ago it was: 1 release 8.87, 2 releases 8.09, 3+ releases 8.02; each loss since then -0.92 (plus -2.06 × earlier position).

## 5. What the résumé-only pairwise model learned

Weights are log-odds of being ranked ahead per one unit of the feature (z-scored features: per standard deviation across FBS that week; counts: per game). Base = 2014–2023; the next two columns add the learned shift for the 12-team era (2024+) and for 2025+ (new metrics).

| Feature | Base | 12-team era | 2025+ |
|---|---|---|---|
| fcs_losses | -2.15 |  |  |
| rating | 1.80 | 2.49 | 2.38 |
| postgame_wp | 1.77 |  |  |
| g6_wae | -1.48 |  |  |
| best_win | 1.20 | 1.14 | 1.37 |
| win_credit | -1.17 |  |  |
| margin_cap | -1.12 | -0.95 | -1.36 |
| is_g6 | 1.05 | 2.08 | 2.08 |
| losses_sq | -0.95 |  |  |
| ccg_loss | 0.83 |  |  |
| losses | -0.78 | -1.39 | -1.40 |
| ccg_played | 0.71 |  |  |
| record_strength | 0.69 | 0.79 | 0.70 |
| best_win3 | 0.69 |  |  |
| is_power | 0.67 |  |  |
| played_week | -0.62 |  |  |
| conf_strength | 0.61 | 0.13 | 1.35 |
| last3_value | 0.59 |  |  |
| w_power | 0.55 |  |  |
| w_top10 | 0.53 |  |  |
| loss_unranked | -0.50 | -0.50 | -0.51 |
| l_top10 | 0.48 |  |  |
| won_week | 0.47 |  |  |
| w_ranked_g | 0.45 |  |  |
| l500 | -0.45 |  |  |
| loss_penalty | -0.43 |  |  |
| wins | 0.40 |  |  |
| wae | -0.36 | -0.26 | -0.05 |

### How priorities shifted by era

| Feature | Base | 12-team era | 2025+ | Shift 2014–23 → 2025+ |
|---|---|---|---|---|
| is_g6 | 1.05 | 2.08 | 2.08 | 1.03 |
| sos | 0.14 | 0.12 | 1.01 | 0.87 |
| conf_strength | 0.61 | 0.13 | 1.35 | 0.73 |
| sos_top4 | -0.18 | -0.36 | -0.82 | -0.64 |
| losses | -0.78 | -1.39 | -1.40 | -0.61 |
| rating | 1.80 | 2.49 | 2.38 | 0.58 |
| worst_loss | -0.02 | 0.58 | -0.41 | -0.39 |
| champion | -0.12 | -0.31 | -0.48 | -0.36 |
| wae | -0.36 | -0.26 | -0.05 | 0.31 |
| margin_cap | -1.12 | -0.95 | -1.36 | -0.24 |
| best_win | 1.20 | 1.14 | 1.37 | 0.17 |
| w500 | 0.31 | 0.43 | 0.21 | -0.10 |
| conf_sos_rel | 0.07 | 0.45 | 0.04 | -0.03 |
| loss_unranked | -0.50 | -0.50 | -0.51 | -0.01 |
| record_strength | 0.69 | 0.79 | 0.70 | 0.01 |

Only two seasons (12 rankings) exist in the 12-team era and one-plus under the 2025 metrics, so these shifts are estimated from little data and are regularised toward zero.

### Which groups of criteria carry predictive weight (ablation)

Individual weights above are entangled because many features measure overlapping things (rating, margin and postgame win expectancy; win credit and best win). A cleaner test: remove one whole group from the résumé-only model, refit, and see how much the leave-one-season-out error rises.

| Removed | MAE | MAE top 12 | Top-12 overlap | Within 1 | MAE change |
|---|---|---|---|---|---|
| Quality of wins | 2.477 | 1.640 | 0.900 | 0.464 | 0.331 |
| Team strength and dominance (rating, margin, game control) | 2.390 | 1.659 | 0.899 | 0.461 | 0.243 |
| Recency and this week's result | 2.212 | 1.512 | 0.913 | 0.504 | 0.065 |
| Losses and record | 2.196 | 1.520 | 0.920 | 0.486 | 0.049 |
| Head-to-head and common opponents | 2.193 | 1.502 | 0.918 | 0.492 | 0.046 |
| Brand (preseason AP rank) | 2.155 | 1.487 | 0.916 | 0.503 | 0.008 |
| Margin and game control only | 2.147 | 1.478 | 0.917 | 0.499 | 0.001 |
| Power / Group of Six / independent status | 2.147 | 1.495 | 0.917 | 0.506 | 0.001 |
| (nothing removed) | 2.147 | 1.479 | 0.917 | 0.497 | 0.000 |
| Schedule strength, strength of record, record strength | 2.141 | 1.472 | 0.916 | 0.501 | -0.006 |
| Quality of losses | 2.131 | 1.434 | 0.912 | 0.512 | -0.016 |
| Conference title and conference strength | 2.129 | 1.481 | 0.918 | 0.507 | -0.018 |

A group whose removal costs nothing is either unused by the committee or already implied by the other groups; it is not evidence the committee ignores it.

### Head-to-head and common opponents (pod effect)

Head-to-head winner bonus: 1.81 log-odds (résumé-only model), 2.32 (model with polls), applied with a closeness kernel of width 4 / 4 log-odds, so it fades once two teams are far apart. Common-opponent edge: 0.27 / 0.41 per net game.

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
| Ensemble (+polls, last week) | 1.32 | 1.21 | 0.71 | 0.88 | 0.96 |
| Ensemble (carry-over from first ranking) | 1.73 | 1.84 | 0.64 | 0.86 | 0.96 |

Intervals are on integer ranks, so coverage above the nominal level is expected for teams whose rank is nearly certain. Calibration of simulated *playoff odds* is reported with the simulation.

## 8. 2026 so far (true out-of-sample)

**Release 1 (2026-11-03),** predicted with a model trained only on 2014–2025: MAE 1.76 (AP poll 2.12), within one spot 48%, within two 68%, top-4 overlap 4/4, top-12 overlap 12/12, top-25 overlap 25/25.

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
| Ole Miss | 10 | 11 | 12 | 6-2 | +1 |
| Notre Dame | 11 | 7 | 4 | 7-1 | -4 |
| Ohio State | 12 | 12 | 11 | 6-2 | +0 |
| Tennessee | 13 | 16 | 16 | 6-2 | +3 |
| LSU | 14 | 15 | 15 | 6-2 | +1 |
| Texas Tech | 15 | 13 | 13 | 7-1 | -2 |
| Wake Forest | 16 | 14 | 14 | 7-1 | -2 |
| Houston | 17 | 19 | 17 | 6-2 | +2 |
| Missouri | 18 | 18 | 19 | 6-2 | +0 |
| Iowa | 19 | 20 | 21 | 6-2 | +1 |
| USC | 20 | 17 | 18 | 7-2 | -3 |
| South Carolina | 21 | 21 | 23 | 5-3 | +0 |
| Arizona | 22 | 25 | 24 | 6-2 | +3 |
| Oregon | 23 | 24 | NR | 5-3 | +1 |
| Florida | 24 | 23 | 25 | 5-3 | -1 |
| Boise State | 25 | 22 | 20 | 7-1 | -3 |
**Release 2 (2026-11-10),** predicted with a model trained only on 2014–2025: MAE 1.84 (AP poll 2.76), within one spot 40%, within two 76%, top-4 overlap 2/4, top-12 overlap 11/12, top-25 overlap 24/25.

| Team | Actual | Predicted | AP | Record | Error |
|---|---|---|---|---|---|
| Georgia | 1 | 1 | 1 | 8-1 | +0 |
| Utah | 2 | 2 | 2 | 9-0 | +0 |
| Alabama | 3 | 6 | 7 | 7-2 | +3 |
| UCLA | 4 | 5 | 4 | 9-0 | +1 |
| Notre Dame | 5 | 3 | 3 | 8-1 | -2 |
| Texas | 6 | 8 | 8 | 7-2 | +2 |
| Miami | 7 | 7 | 5 | 8-1 | +0 |
| Tennessee | 8 | 9 | 9 | 7-2 | +1 |
| Indiana | 9 | 4 | 6 | 8-1 | -5 |
| BYU | 10 | 12 | 12 | 7-2 | +2 |
| Kentucky | 11 | 13 | 13 | 7-2 | +2 |
| Texas Tech | 12 | 10 | 10 | 8-1 | -2 |
| Wake Forest | 13 | 11 | 11 | 8-1 | -2 |
| Houston | 14 | 17 | 14 | 7-2 | +3 |
| Iowa | 15 | 14 | 16 | 7-2 | -1 |
| Oregon | 16 | 18 | 18 | 6-3 | +2 |
| South Carolina | 17 | 20 | 20 | 6-3 | +3 |
| USC | 18 | 16 | 15 | 7-2 | -2 |
| Ole Miss | 19 | 15 | 19 | 6-3 | -4 |
| Ohio State | 20 | 19 | 21 | 6-3 | -1 |
| LSU | 21 | 21 | 24 | 6-3 | +0 |
| Missouri | 22 | 24 | NR | 6-3 | +2 |
| Arizona | 23 | 22 | 22 | 7-2 | -1 |
| Boise State | 24 | 23 | 17 | 8-1 | -1 |
| Oklahoma State | 25 | 29 | NR | 6-3 | +4 |
| SMU | NR | 25 | 25 | 7-2 |  |

## 9. Biggest misses in the backtest

| Season | Release | Team | Actual | Predicted | Error |
|---|---|---|---|---|---|
| 2018 | 1 | NC State | 21 | 33 | 12 |
| 2021 | 1 | San Diego State | 24 | 36 | 12 |
| 2025 | 3 | Tulane | 24 | 34 | 10 |
| 2014 | 4 | Minnesota | 25 | 35 | 10 |
| 2016 | 5 | Utah | 20 | 29 | 9 |
| 2021 | 1 | Mississippi State | 17 | 26 | 9 |
| 2016 | 6 | Tennessee | 21 | 30 | 9 |
| 2018 | 1 | Texas A&M | 20 | 29 | 9 |
| 2019 | 3 | Iowa State | 22 | 31 | 9 |
| 2025 | 5 | Houston | 21 | 29 | 8 |
| 2021 | 2 | Utah | 24 | 32 | 8 |
| 2016 | 3 | Tennessee | 19 | 27 | 8 |
| 2016 | 1 | Penn State | 12 | 19 | 7 |
| 2016 | 1 | West Virginia | 20 | 13 | -7 |
| 2024 | 1 | Colorado | 20 | 27 | 7 |

Error by position (the voting pools):

| Actual rank band | MAE |
|---|---|
| 1–4 | 0.35 |
| 5–8 | 0.82 |
| 9–12 | 1.02 |
| 13–16 | 1.25 |
| 17–25 | 1.82 |

Group of Five/Six teams: mean signed error +0.44 places, MAE 1.81 (202 team-weeks). Power-conference teams: MAE 1.13.

## 10. What the chair says versus what the data support

`data/committee_statements.csv` holds 1461 tagged sentences from 33 committee media calls (2020–2026; earlier transcripts were not obtainable). Mentions per season:

| factor | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | total |
|---|---|---|---|---|---|---|---|---|
| quality_wins | 19 | 22 | 28 | 11 | 16 | 29 | 12 | 137 |
| loss_quality | 13 | 20 | 14 | 5 | 46 | 26 | 7 | 131 |
| strength_of_schedule | 4 | 15 | 5 | 4 | 22 | 25 | 3 | 78 |
| conference_championship | 6 | 9 | 3 | 12 | 18 | 23 | 0 | 71 |
| eye_test | 15 | 16 | 4 | 5 | 11 | 12 | 7 | 70 |
| recency | 10 | 10 | 7 | 8 | 9 | 17 | 2 | 63 |
| head_to_head | 6 | 18 | 3 | 5 | 8 | 12 | 3 | 55 |
| body_of_work | 5 | 1 | 19 | 2 | 14 | 7 | 4 | 52 |
| road_wins | 1 | 6 | 1 | 2 | 14 | 11 | 10 | 45 |
| complete_team | 5 | 13 | 8 | 1 | 2 | 10 | 2 | 41 |
| availability | 5 | 2 | 5 | 4 | 6 | 9 | 5 | 36 |
| efficiency_stats | 4 | 6 | 6 | 3 | 3 | 11 | 0 | 33 |
| game_control | 2 | 5 | 1 | 1 | 7 | 12 | 2 | 30 |
| group_of_five | 0 | 2 | 0 | 5 | 0 | 7 | 1 | 15 |
| common_opponents | 2 | 6 | 1 | 0 | 1 | 4 | 0 | 14 |
| margin | 1 | 0 | 2 | 0 | 1 | 9 | 0 | 13 |
| line_of_scrimmage | 2 | 0 | 1 | 1 | 2 | 4 | 1 | 11 |
| record_strength | 0 | 0 | 0 | 0 | 2 | 7 | 0 | 9 |
| conference_strength | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |

How each stated factor shows up in the fitted résumé-only model (weight, base era):

| Stated factor | Mentions | Model features and weights |
|---|---|---|
| quality_wins | 137 | win_credit -1.17; best_win +1.20; best_win2 +0.35; w_ranked +0.33; w_top10 +0.53; w500 +0.31 |
| loss_quality | 131 | losses -0.78; loss_penalty -0.43; worst_loss -0.02; loss_unranked -0.50; loss_sub500 -0.33 |
| strength_of_schedule | 78 | sos +0.14; sos_top4 -0.18 |
| record_strength | 9 | wae -0.36; sor_z -0.22; record_strength +0.69 |
| head_to_head | 55 | h2h +1.81 |
| common_opponents | 14 | common +0.27 |
| conference_championship | 71 | champion -0.12; ccg_loss +0.83 |
| game_control | 30 | game_control +0.13 |
| margin | 13 | margin_cap -1.12; margin_loss_cap +0.34 |
| eye_test | 70 | rating +1.80 |
| road_wins | 45 | road_w_ranked -0.15; w_road -0.01 |
| recency | 63 | last3_value +0.59; trend +0.32; recent_losses -0.11; days_since_loss -0.21 |
| conference_strength | 1 | conf_strength +0.61; conf_sos_rel +0.07 |
| group_of_five | 15 | is_g6 +1.05; g6_wae -1.48 |
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

