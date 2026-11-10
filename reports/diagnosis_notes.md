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
