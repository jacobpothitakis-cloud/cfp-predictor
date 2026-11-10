# Team-strength validation

Walk-forward: each game is predicted from the rating snapshot taken before kickoff. 2014 is burn-in (no prior season in the database) and is excluded.

Parameters: `RatingParams(cap=42.0, hfa=2.6, ridge=3.0, carry=0.74, fcs_prior=-19.0, new_fbs_prior=-12.0, sigma=14.0, recency=0.0)`

The five parameters were chosen by grid search on 2015–2025 walk-forward log loss. The surface is flat near the optimum (log loss 0.4956–0.4967 across the 12 best settings), so the in-sample selection effect is small; the split test below quantifies it.

## By season

| Season | Games | Accuracy | Log loss | Brier | Margin MAE | FBS-v-FBS acc | FBS-v-FBS log loss |
|---|---|---|---|---|---|---|---|
| 2015 | 870 | 0.772 | 0.478 | 0.157 | 14.0 | 0.753 | 0.505 |
| 2016 | 873 | 0.740 | 0.528 | 0.177 | 14.1 | 0.716 | 0.557 |
| 2017 | 874 | 0.748 | 0.491 | 0.164 | 13.4 | 0.729 | 0.517 |
| 2018 | 884 | 0.761 | 0.471 | 0.157 | 13.6 | 0.736 | 0.513 |
| 2019 | 888 | 0.766 | 0.464 | 0.153 | 13.1 | 0.735 | 0.511 |
| 2020 | 568 | 0.734 | 0.525 | 0.177 | 13.2 | 0.719 | 0.548 |
| 2021 | 887 | 0.736 | 0.515 | 0.171 | 13.7 | 0.709 | 0.551 |
| 2022 | 896 | 0.731 | 0.527 | 0.178 | 13.3 | 0.700 | 0.573 |
| 2023 | 910 | 0.760 | 0.486 | 0.161 | 13.1 | 0.730 | 0.536 |
| 2024 | 919 | 0.734 | 0.503 | 0.170 | 13.2 | 0.702 | 0.555 |
| 2025 | 934 | 0.749 | 0.474 | 0.158 | 12.7 | 0.715 | 0.527 |
| 2026 | 685 | 0.745 | 0.493 | 0.164 | 13.9 | 0.699 | 0.563 |
| **2015–25** | 9503 | 0.749 | 0.496 | 0.165 | 13.4 | 0.722 | 0.535 |

## Held-out seasons

Parameters re-tuned on 2015–2020 only gave cap=42, carry=0.7, sigma=13; on the untouched 2021–2025 seasons: accuracy 0.744, log loss 0.502, Brier 0.168 over 4546 games. Shipped parameters on the same seasons: log loss 0.501.

## Win-probability calibration (2015–2025, favourite's side)

| Predicted bin | Games | Mean predicted | Actual win rate |
|---|---|---|---|
| (0.499, 0.55] | 978 | 0.525 | 0.526 |
| (0.55, 0.6] | 1016 | 0.576 | 0.560 |
| (0.6, 0.65] | 984 | 0.626 | 0.631 |
| (0.65, 0.7] | 943 | 0.674 | 0.676 |
| (0.7, 0.75] | 960 | 0.724 | 0.727 |
| (0.75, 0.8] | 884 | 0.776 | 0.768 |
| (0.8, 0.85] | 890 | 0.825 | 0.831 |
| (0.85, 0.9] | 887 | 0.875 | 0.892 |
| (0.9, 0.95] | 889 | 0.925 | 0.918 |
| (0.95, 1.0] | 1072 | 0.976 | 0.981 |

## Against closing lines

Not enough closing lines in the database to compare. CFBD lines have not been ingested.
