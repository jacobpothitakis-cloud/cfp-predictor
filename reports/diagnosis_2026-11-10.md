# Diagnosis: the Nov 10, 2026 ranking

What went wrong, what the committee chair said, whether the outside criticism is fair, and what changed.
Every number here was computed from the project database or read from a cited source.

## 1. Result

Prediction stored Nov 9 (games through that evening), scored against the committee's ranking of Nov 10.

| | This week | 2014–2025 test average |
|---|---|---|
| Mean miss per team | 1.84 places | 1.30 |
| Exact spot | 4 of 25 | |
| Within one / two spots | 48% / 76% | 69% / 84% |
| Top 4 / top 12 / top 25 | 2 of 4 / 11 of 12 / 24 of 25 | |
| Actual rank inside the stated 80% range | 76% of teams | |
| Repeating the committee's previous ranking | 3.96 | |

Misses of three or more: Oklahoma State (model outside top 30, actual 25), Indiana (4 v 9), Ole Miss (15 v 19),
Alabama (6 v 3), Houston (17 v 14), South Carolina (20 v 17).

## 2. What the chair said

Nov 10 remarks are from press reports of Hunter Yurachek on ESPN's rankings show and to reporters; the full
transcript was not yet in the ASAP Sports archive. Nov 3 remarks are from the full transcript.

- **Alabama, No. 3.** First in the committee's record-strength and schedule-strength figures; four "quality wins";
  the Mississippi State win still counts because of how that team was playing at the time.
  ([ESPN](https://www.espn.com/college-football/story/_/id/50448424/georgia-takes-cfp-top-spot-notre-dame-leaps-no-5))
- **Indiana, No. 9.** "Probably a victim of having a bye week at the wrong time."
  ([AP via TSN](https://www.tsn.ca/ncaa/football/article/georgia-nabs-top-spot-in-cfp-rankings-while-notre-dame-jumps-ahead-of-miami-after-beating-canes/))
- **Miami, No. 7.** Lost as No. 1 while five teams behind had signature wins; looked poor early against Notre Dame.
  ([Canes Warning](https://caneswarning.com/miami-unsurprisingly-drops-to-no-7-in-cfp-rankings-but-a-first-round-bye-remains-in-reach))
- **Notre Dame, No. 5.** Looked like a No. 5 team while leading Miami 24-3; helped by losses ahead of them. (ESPN)
- **UCLA, No. 4.** The two-touchdown win at Minnesota made them look like the fourth-best team. (AP via TSN)
- **Tennessee, No. 8.** Behind Alabama on schedule strength, by judgment and by the figures.
  ([Yardbarker](https://www.yardbarker.com/college_football/articles/cfp_committee_chair_tells_why_alabama_moved_up_remains_ahead_of_tennessee/s1_17086_44545110))
- **Nov 3 call** ([transcript](https://asaptext.com/asap_media/media/1070/1122/transcripts/172609.html)): wins over
  teams "the committee thinks highly of" are rewarded; losses to such teams are not over-penalised; records
  against .500-or-better opponents were cited for Alabama, Indiana and Notre Dame; head-to-head only applies
  when the two teams are being compared in the same group; Utah was placed above UCLA on how the teams look
  when watched; a win is judged by how good the opponent was when the game was played.

Across the 33 chair calls in the archive since 2020, the factors cited most often are quality of wins (18% of
tagged statements), quality of losses (17%), schedule strength (10%), conference championships (9%), how a team
looks (9%), recent form (8%) and head-to-head (7%).

## 3. Miss by miss

| Team | Model | Actual | What happened | Could history have told us? |
|---|---|---|---|---|
| Indiana | 4 | 9 | Idle week; five teams with big wins went past. | No. Of 45 idle top-10 teams in 2014–2025, none fell more than one place. The résumé-only model also had them 5th. |
| Alabama | 6 | 3 | Won 31-13 at LSU. Chair: No. 1 in the committee's record and schedule strength. | Partly. Our schedule strength has them 4th, record strength 7th; ESPN's public figures 1st and 5th. The committee's private figure rates them higher than anything public. |
| Ole Miss | 15 | 19 | Lost 42-7 at Georgia, third loss. Fell nine places. | Partly. The model expected about a five-place fall. |
| Oklahoma State | 31+ | 25 | Won 66-28 at Kansas State. | No. Ranks 20–25 are where the model is weakest (average miss 2.0). ESPN has them 38th in strength of record. |
| Houston, South Carolina | 17, 20 | 14, 17 | Both won. | Small misses in the same direction as Alabama: the committee rates their schedules higher than we do. |

Two checks on whether the model is built wrong rather than unlucky:

- **Too anchored on last week or the polls?** No. The résumé-only model, which sees neither, had Indiana 5th and
  Alabama 7th. And across 2014–2025, pulling predictions toward the résumé-only ranking made them worse, not better.
- **Too slow to reward a big win?** No. Across 218 past cases of a top-quintile win, the model's average signed
  error is +0.09 places, and in 2024–2025 it was slightly too generous to such teams.

What does repeat is the direction first seen on Nov 3: SEC teams about a place too low, Big Ten teams about a
place too high. This committee is giving more weight to schedule strength and quality wins, and less to the
loss column, than earlier committees did.

## 4. Is the criticism fair?

| Claim | Verdict |
|---|---|
| "The simulations don't add knowledge; they replay the model's assumptions." | Fair for the season odds, which are only as good as the model behind them. Not relevant to this miss: the weekly ranking prediction does not use the simulation. (It is 10,000 runs, not 100,000.) |
| "Blind to who a team beat." | Not accurate. Quality of wins is twelve separate measures in the model, including best wins, wins over ranked teams and wins over .500-or-better teams. |
| "Blind to head-to-head results." | Not accurate. Head-to-head and common opponents are a separate stage, applied when two teams are close, which is how the chair describes the committee using them. |
| "Blind to injuries." | Fair. There is a manual input for a starting quarterback being out, with an assumed effect, and it is empty. The chair cited player availability for Mississippi State, Ohio State and Miami this month. |
| "Blind to how a loss looked on film." | Mostly fair. The model has margin, a quarter-by-quarter game-control proxy and a power rating, which stand in for it. The chair twice separated teams on how they look when watched. |
| "Judging a moving target; 24 of 25 is a solid result." | Fair, and generous. A normal week is closer than this one was. |

The criticism misses the largest real gap, which is neither film nor injuries: the committee uses its own
record-strength and schedule-strength figures, we can only rebuild them, and this week our rebuild and theirs
disagreed about Alabama.

## 5. What changed

- **Rankings already released this season now count six times as much** as an older ranking when the weekly
  model is retrained (`SEASON_BOOST` in `cfp/committee/model.py`). Tested walk-forward on 2014–2025, predicting
  each ranking from the second onward: mean miss 1.111 at the old weight, 1.075 at three, 1.059 at six, 1.060 at
  ten. Better in 10 of 12 seasons. On the Nov 10 ranking it would have scored 1.72 instead of 1.88, with 12 of 12
  in the top 12; Alabama moves from 6th to 5th and Indiana stays 4th. It is a modest gain, not a cure.
- **The Report card tab** now shows the chair's reasons next to what the model had, with sources.
- **A data error was fixed**: a Nov 9 media call about expanding the playoff had been filed as a chair's ranking
  call. It and one similar 2022 call are now excluded from `data/committee_statements.csv`.

## 6. What was tried and rejected

- Pulling the prediction toward the résumé-only ranking: made 2014–2025 worse (1.01 to 1.08 at a 20% blend).
- A penalty for idle weeks: no support in 158 past idle-team cases (average signed error +0.06).
- Hand-tuning the record-strength rebuild so Alabama comes out first: one data point, so it would be fitting
  to a single team.

## 7. Still open

- The committee's private record-strength figure. More chair statements naming where teams sit in it would
  let the rebuild be checked against more than one team.
- Injuries are a manual input with an assumed effect.
- The bottom five places of the ranking.
- The same-season weighting was tested on the weekly model only, not on the simulation's carry-over model.
