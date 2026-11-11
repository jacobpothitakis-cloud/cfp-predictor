# 2026 prediction log

Each prediction below was stored before the committee released that ranking, then scored against it. The AP poll is the one that was current when the prediction was made. A team a source did not list counts as rank 31.

Season to date: 1 scored release(s), mean absolute error 1.84 (AP poll 2.32).

## Release 1 (2026-11-03)

No pre-release prediction was stored (the project did not exist yet, or `cfp predict` was not run that week). A retrospective out-of-sample prediction for it is in `reports/backtest_report.md`, section 8.

## Release 2 (2026-11-10)

Prediction made 2026-11-10 01:54 UTC; AP poll dated 2026-11-08.

| | Model | AP poll |
|---|---|---|
| Mean absolute error (places) | 1.84 | 2.32 |
| Within one spot | 48% | 36% |
| Within two spots | 76% | 68% |
| Top 4 | 2/4 | 3/4 |
| Top 12 | 11/12 | 11/12 |
| Closer on (teams) | 8 | 4 |

Last week's committee ranking left unchanged: 3.96. The model's stated 80% range held the actual rank for 76% of teams.

| Team | Actual | Model | Error | AP | AP error | That week |
|---|---|---|---|---|---|---|
| Georgia | 1 | 1 | +0 | 1 | +0 | beat Ole Miss 42-7 (away) |
| Utah | 2 | 2 | +0 | 2 | +0 | beat BYU 28-20 |
| Alabama | 3 | 6 | +3 | 7 | +4 | beat LSU 31-13 (away) |
| UCLA | 4 | 5 | +1 | 4 | +0 | beat Minnesota 34-20 (away) |
| Notre Dame | 5 | 3 | -2 | 3 | -2 | beat Miami 31-24 |
| Texas | 6 | 8 | +2 | 8 | +2 | beat Missouri 34-20 (away) |
| Miami | 7 | 7 | +0 | 5 | -2 | lost to Notre Dame 24-31 (away) |
| Tennessee | 8 | 9 | +1 | 9 | +1 | beat Kentucky 34-6 |
| Indiana | 9 | 4 | -5 | 6 | -3 | did not play |
| BYU | 10 | 11 | +1 | 12 | +2 | lost to Utah 20-28 (away) |
| Kentucky | 11 | 13 | +2 | 13 | +2 | lost to Tennessee 6-34 (away) |
| Texas Tech | 12 | 10 | -2 | 10 | -2 | beat West Virginia 53-48 |
| Wake Forest | 13 | 12 | -1 | 11 | -2 | beat Merrimack 77-0 |
| Houston | 14 | 17 | +3 | 14 | +0 | beat Cincinnati 31-24 |
| Iowa | 15 | 14 | -1 | 16 | +1 | beat Northwestern 24-10 (away) |
| Oregon | 16 | 18 | +2 | 18 | +2 | beat Ohio State 41-28 (away) |
| South Carolina | 17 | 20 | +3 | 20 | +3 | beat Texas A&M 17-9 |
| USC | 18 | 16 | -2 | 15 | -3 | did not play |
| Ole Miss | 19 | 15 | -4 | 19 | +0 | lost to Georgia 7-42 |
| Ohio State | 20 | 19 | -1 | 21 | +1 | lost to Oregon 28-41 |
| LSU | 21 | 21 | +0 | 24 | +3 | lost to Alabama 13-31 |
| Missouri | 22 | 24 | +2 | NR | +9 | lost to Texas 20-34 |
| Arizona | 23 | 22 | -1 | 22 | -1 | beat TCU 29-28 |
| Boise State | 24 | 23 | -1 | 17 | -7 | beat Colorado State 64-38 (away) |
| Oklahoma State | 25 | NR | +6 | NR | +6 | beat Kansas State 66-28 (away) |

Biggest misses:

- Oklahoma State (6-3): actual 25, predicted NR, 6 too low (AP NR, previous committee rank NR; beat Kansas State 66-28 (away)).
- Indiana (8-1): actual 9, predicted 4, 5 too high (AP 6, previous committee rank 4; did not play).
- Ole Miss (6-3): actual 19, predicted 15, 4 too high (AP 19, previous committee rank 10; lost to Georgia 7-42).
- Alabama (7-2): actual 3, predicted 6, 3 too low (AP 7, previous committee rank 5; beat LSU 31-13 (away)).
- Houston (7-2): actual 14, predicted 17, 3 too low (AP 14, previous committee rank 17; beat Cincinnati 31-24).
- South Carolina (6-3): actual 17, predicted 20, 3 too low (AP 20, previous committee rank 21; beat Texas A&M 17-9).
