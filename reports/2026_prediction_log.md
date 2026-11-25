# 2026 prediction log

Each prediction below was stored before the committee released that ranking, then scored against it. The AP poll is the one that was current when the prediction was made. A team a source did not list counts as rank 31.

Season to date: 3 scored release(s), mean absolute error 1.36 (AP poll 1.93).

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
- South Carolina (6-3): actual 17, predicted 20, 3 too low (AP 20, previous committee rank 21; beat Texas A&M 17-9).
- Houston (7-2): actual 14, predicted 17, 3 too low (AP 14, previous committee rank 17; beat Cincinnati 31-24).

## Release 3 (2026-11-17)

Prediction made 2026-11-16 23:16 UTC; AP poll dated 2026-11-16.

| | Model | AP poll |
|---|---|---|
| Mean absolute error (places) | 1.32 | 1.92 |
| Within one spot | 72% | 52% |
| Within two spots | 84% | 80% |
| Top 4 | 4/4 | 3/4 |
| Top 12 | 12/12 | 12/12 |
| Closer on (teams) | 12 | 4 |

Last week's committee ranking left unchanged: 2.84. The model's stated 80% range held the actual rank for 84% of teams.

| Team | Actual | Model | Error | AP | AP error | That week |
|---|---|---|---|---|---|---|
| Georgia | 1 | 1 | +0 | 1 | +0 | beat Missouri 48-21 |
| Utah | 2 | 2 | +0 | 2 | +0 | beat Arizona 40-39 (away) |
| Alabama | 3 | 4 | +1 | 7 | +4 | beat Vanderbilt 21-6 (away) |
| UCLA | 4 | 3 | -1 | 4 | +0 | beat Illinois 22-14 |
| Texas | 5 | 7 | +2 | 8 | +3 | beat LSU 22-21 (away) |
| Notre Dame | 6 | 5 | -1 | 3 | -3 | beat Boston College 41-7 |
| Miami | 7 | 6 | -1 | 5 | -2 | beat Duke 36-25 |
| Indiana | 8 | 8 | +0 | 6 | -2 | beat USC 27-13 |
| Tennessee | 9 | 9 | +0 | 9 | +0 | beat Texas A&M 22-7 (away) |
| BYU | 10 | 10 | +0 | 12 | +2 | beat Baylor 28-7 |
| Wake Forest | 11 | 12 | +1 | 10 | -1 | beat SMU 49-7 (away) |
| Texas Tech | 12 | 11 | -1 | 11 | -1 | beat Oklahoma State 52-27 (away) |
| Oregon | 13 | 14 | +1 | 14 | +1 | beat Michigan 31-7 |
| Ohio State | 14 | 15 | +1 | 16 | +2 | beat Northwestern 28-19 |
| Iowa | 15 | 13 | -2 | 13 | -2 | beat Purdue 28-7 |
| Kentucky | 16 | 16 | +0 | 17 | +1 | lost to Florida 14-21 |
| Boise State | 17 | 18 | +1 | 15 | -2 | beat Oregon State 46-41 |
| USC | 18 | 17 | -1 | 18 | +0 | lost to Indiana 13-27 (away) |
| Arizona | 19 | 22 | +3 | 19 | +0 | lost to Utah 39-40 |
| Houston | 20 | 20 | +0 | 20 | +0 | lost to Colorado 32-42 (away) |
| South Carolina | 21 | 25 | +4 | NR | +10 | lost to Arkansas 6-34 (away) |
| Oklahoma | 22 | 26 | +4 | NR | +9 | beat Ole Miss 24-17 |
| Louisville | 23 | 21 | -2 | 21 | -2 | beat North Carolina 16-14 (away) |
| Florida | 24 | 24 | +0 | 23 | -1 | beat Kentucky 21-14 (away) |
| Ole Miss | 25 | 19 | -6 | 25 | +0 | lost to Oklahoma 17-24 (away) |

Biggest misses:

- Ole Miss (6-4): actual 25, predicted 19, 6 too high (AP 25, previous committee rank 19; lost to Oklahoma 17-24 (away)).
- South Carolina (6-4): actual 21, predicted 25, 4 too low (AP NR, previous committee rank 17; lost to Arkansas 6-34 (away)).
- Oklahoma (6-4): actual 22, predicted 26, 4 too low (AP NR, previous committee rank NR; beat Ole Miss 24-17).
- Arizona (7-3): actual 19, predicted 22, 3 too low (AP 19, previous committee rank 23; lost to Utah 39-40).

## Release 4 (2026-11-24)

Prediction made 2026-11-23 16:41 UTC; AP poll dated 2026-11-22.

| | Model | AP poll |
|---|---|---|
| Mean absolute error (places) | 0.92 | 1.56 |
| Within one spot | 80% | 60% |
| Within two spots | 92% | 84% |
| Top 4 | 4/4 | 3/4 |
| Top 12 | 12/12 | 11/12 |
| Closer on (teams) | 12 | 4 |

Last week's committee ranking left unchanged: 3.24. The model's stated 80% range held the actual rank for 92% of teams.

| Team | Actual | Model | Error | AP | AP error | That week |
|---|---|---|---|---|---|---|
| Georgia | 1 | 1 | +0 | 1 | +0 | beat South Carolina 34-6 (away) |
| Utah | 2 | 2 | +0 | 2 | +0 | beat TCU 38-17 (away) |
| Texas | 3 | 3 | +0 | 5 | +2 | beat Arkansas 45-3 |
| Notre Dame | 4 | 4 | +0 | 3 | -1 | beat SMU 24-3 |
| Tennessee | 5 | 7 | +2 | 7 | +2 | beat LSU 27-13 |
| Alabama | 6 | 5 | -1 | 6 | +0 | beat Chattanooga 47-0 |
| Miami | 7 | 6 | -1 | 4 | -3 | beat Virginia Tech 28-20 |
| Wake Forest | 8 | 8 | +0 | 8 | +0 | beat Georgia Tech 50-29 (away) |
| UCLA | 9 | 11 | +2 | 10 | +1 | lost to Michigan 16-18 (away) |
| BYU | 10 | 9 | -1 | 11 | +1 | beat Kansas 31-29 (away) |
| Texas Tech | 11 | 10 | -1 | 9 | -2 | beat Baylor 20-10 (away) |
| Oregon | 12 | 12 | +0 | 14 | +2 | beat Michigan State 45-7 (away) |
| Iowa | 13 | 14 | +1 | 13 | +0 | beat Illinois 34-17 (away) |
| Indiana | 14 | 13 | -1 | 12 | -2 | lost to Washington 24-27 (away) |
| USC | 15 | 15 | +0 | 15 | +0 | beat Maryland 35-21 |
| Arizona | 16 | 16 | +0 | 16 | +0 | beat Kansas State 35-28 (away) |
| Oklahoma | 17 | 18 | +1 | 19 | +2 | beat Texas A&M 28-7 |
| Houston | 18 | 17 | -1 | 17 | -1 | beat West Virginia 34-31 (away) |
| Louisville | 19 | 19 | +0 | 18 | -1 | beat Pittsburgh 42-10 |
| Florida | 20 | 20 | +0 | 20 | +0 | beat Vanderbilt 34-27 |
| Ohio State | 21 | 21 | +0 | 25 | +4 | lost to Nebraska 28-31 (away) |
| Ole Miss | 22 | 22 | +0 | 23 | +1 | beat Wofford 59-17 |
| Missouri | 23 | 25 | +2 | 24 | +1 | beat Kentucky 28-14 |
| Washington | 24 | 28 | +4 | NR | +7 | beat Indiana 27-24 |
| Nebraska | 25 | 30 | +5 | NR | +6 | beat Ohio State 31-28 |

Biggest misses:

- Nebraska (7-4): actual 25, predicted 30, 5 too low (AP NR, previous committee rank NR; beat Ohio State 31-28).
- Washington (7-4): actual 24, predicted 28, 4 too low (AP NR, previous committee rank NR; beat Indiana 27-24).
