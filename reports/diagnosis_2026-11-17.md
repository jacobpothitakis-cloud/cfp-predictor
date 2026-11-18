# Diagnosis: the Nov 17, 2026 ranking

How the prediction for the committee's third ranking did, where it missed, and what follows. Every number here
was computed from the project database or `reports/2026_prediction_log.md`.

## 1. Result

Prediction stored Nov 16 (all Nov 14 games final, AP poll of Nov 16), scored against the committee's ranking of Nov 17.

| | Nov 17 | Nov 10 | 2014–2025 test average |
|---|---|---|---|
| Mean miss per team | **1.32 places** | 1.84 | 1.12 |
| Within one / two spots | 72% / 84% | 48% / 76% | 73% / — |
| Top 4 / top 12 / top 25 | **4 of 4 / 12 of 12** / 24 of 25 | 2 of 4 / 11 of 12 / 24 of 25 | |
| Actual rank inside the stated 80% range | 84% of teams | 76% | |
| AP poll, same measure | 1.92 places, top 4 3 of 4 | 2.32 | |
| Repeating the committee's previous ranking | 2.84 | 3.96 | |

Best week of the season so far. The model beat the AP poll on 12 teams and lost to it on 4. The top four and the
twelve teams in the top 12 were all right, and 8 of the top 12 were in the exact spot.

## 2. Where it missed

| Places | Mean miss |
|---|---|
| Ranks 1–12 | 0.67 |
| Ranks 13–16 | 1.00 |
| Ranks 17–25 | 2.33 |

All four misses of three or more were in the bottom nine:

| Team | Model | Actual | What happened |
|---|---|---|---|
| Ole Miss | 19 | 25 | Lost 17-24 at home to unranked Oklahoma, fourth loss. Fell six places; the model expected no fall. |
| Oklahoma | 26 | 22 | Beat Ole Miss. Went from unranked to 22, ahead of the team it beat. |
| South Carolina | 25 | 21 | Lost 6-34 at Arkansas, fourth loss, and fell only four places (17 to 21). |
| Arizona | 22 | 19 | Lost 39-40 to No. 2 Utah and *rose* four places (23 to 19). |

Two patterns, both ones the chair has described before:

- **A close loss to a top team cost nothing.** Arizona's one-point loss to Utah was treated as a result that helped,
  consistent with the Nov 3 statement that losses to teams "the committee thinks highly of" are not over-penalised.
- **Head-to-head decided the Ole Miss / Oklahoma pair.** Both are 6-4 SEC teams; the committee put the winner three
  places ahead. The model's head-to-head step only compares teams it already has close together, and it had
  Oklahoma outside the 25, seven places from Ole Miss, so the step never applied.

South Carolina is harder to explain from results alone: a 28-point loss to a team outside the ranking and only a
four-place fall. The committee has rated South Carolina's schedule above our rebuild all month (20 v 17 on Nov 10),
the same Alabama-type gap in the record-strength figure noted last week.

By conference, the model's average lean was SEC 0.6 places too low (1.0 on Nov 10), Big 12 0.4 too low (1.2),
Big Ten 0.3 too high (1.0), ACC 0.7 too high. The SEC/Big Ten tilt seen on Nov 3 and Nov 10 is still there but
about half its size, which is what the same-season weighting added on Nov 10 should do.

## 3. What the chair said

Nothing yet. The ranking was released at 8 p.m. Central; at the time of writing (9 p.m.) none of the coverage read
(Saturday Down South, Yahoo/Knoxville News Sentinel, NCAA.com) quoted Hunter Yurachek on any placement, and the
transcript was not in the ASAP Sports archive. Chair notes for this ranking should be added to
`data/chair_notes.yaml` once his remarks are published.

## 4. What changed

Nothing in the model. One good week is not evidence for a change, and none of the misses above had a fix that
was tested on 2014–2025 first. The ranking is now part of the training data, the models were retrained on 75
committee rankings, the prediction for ranking #4 (Nov 24) is stored for scoring, and the 10,000-season simulation
was rerun from this ranking.

## 5. Worth testing before Selection Day

- **Widening the head-to-head step for teams with the same record and conference**, so a pair like Ole Miss /
  Oklahoma is compared even when the base model has them several places apart. Needs a walk-forward test on
  2014–2025 before it ships; the risk is over-applying head-to-head in the top 12, where the model is now accurate.
- **How much a loss to a top-five team costs.** Arizona rose after one. Check the model's signed error for teams
  that lost a one-score game to a top-five opponent across 2014–2025.

## 6. Next week (ranking #4, Nov 24)

The prediction keeps the top 15 unchanged pending Saturday's games, with Boise State and Kentucky swapping at 16/17
and Louisville rising to 20. Simulated playoff odds, 10,000 seasons: Georgia 99%, Notre Dame 98%, Miami 97%,
Utah 96%, Texas 93%, Alabama 92%, Boise State 89%, Indiana 85%, Wake Forest 77%, UCLA 66%, Texas Tech 57%,
BYU 53%, Tennessee 52%. Full detail in `reports/weekly_2026-11-24.md`.
