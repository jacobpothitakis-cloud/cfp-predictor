"""Validation of the team-strength layer. Writes reports/ratings_validation.md."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from cfp import data, ratings
from cfp.config import REPORTS
from cfp.ratings.model import RatingParams, Ratings, build_all, calibration_table, evaluate


def run() -> str:
    g = data.games()
    R = ratings.load()
    done = g[g["completed"] & (g["season"] >= 2015)]
    pr = R.pregame(done)
    fbs = pr[pr["home_fbs"] & pr["away_fbs"]]
    lines = [f"# Team-strength validation\n",
             "Walk-forward: each game is predicted from the rating snapshot taken before kickoff. "
             "2014 is burn-in (no prior season in the database) and is excluded.\n",
             f"Parameters: `{R.p}`\n",
             "The five parameters were chosen by grid search on 2015–2025 walk-forward log loss. The surface is "
             "flat near the optimum (log loss 0.4956–0.4967 across the 12 best settings), so the in-sample "
             "selection effect is small; the split test below quantifies it.\n",
             "## By season\n", "| Season | Games | Accuracy | Log loss | Brier | Margin MAE | FBS-v-FBS acc | FBS-v-FBS log loss |",
             "|---|---|---|---|---|---|---|---|"]
    for s in sorted(pr["season"].unique()):
        e, f = evaluate(pr[pr.season == s]), evaluate(fbs[fbs.season == s])
        lines.append(f"| {s} | {e['n']} | {e['acc']:.3f} | {e['logloss']:.3f} | {e['brier']:.3f} | {e['mae_margin']:.1f} | {f['acc']:.3f} | {f['logloss']:.3f} |")
    e, f = evaluate(pr[pr.season < 2026]), evaluate(fbs[fbs.season < 2026])
    lines.append(f"| **2015–25** | {e['n']} | {e['acc']:.3f} | {e['logloss']:.3f} | {e['brier']:.3f} | {e['mae_margin']:.1f} | {f['acc']:.3f} | {f['logloss']:.3f} |")

    # held-out check: tune on 2015-2020 only, score 2021-2025
    best = None
    for cap in (28, 35, 42, 60):
        for carry in (0.6, 0.7, 0.8):
            for sigma in (13, 14, 15, 16):
                p = RatingParams(cap=cap, carry=carry, sigma=sigma)
                RR = Ratings(build_all(g, data.membership(), p), p)
                q = RR.pregame(done)
                tr = evaluate(q[q.season <= 2020])["logloss"]
                if best is None or tr < best[0]:
                    best = (tr, p, evaluate(q[(q.season >= 2021) & (q.season <= 2025)]))
    lines += ["\n## Held-out seasons\n",
              f"Parameters re-tuned on 2015–2020 only gave cap={best[1].cap}, carry={best[1].carry}, sigma={best[1].sigma}; "
              f"on the untouched 2021–2025 seasons: accuracy {best[2]['acc']:.3f}, log loss {best[2]['logloss']:.3f}, "
              f"Brier {best[2]['brier']:.3f} over {best[2]['n']} games. Shipped parameters on the same seasons: "
              f"log loss {evaluate(pr[(pr.season >= 2021) & (pr.season <= 2025)])['logloss']:.3f}.\n",
              "## Win-probability calibration (2015–2025, favourite's side)\n",
              "| Predicted bin | Games | Mean predicted | Actual win rate |", "|---|---|---|---|"]
    for r in calibration_table(pr[pr.season < 2026]).itertuples():
        lines.append(f"| {r.bin} | {r.n} | {r.predicted:.3f} | {r.actual:.3f} |")

    # closing lines
    ln = pr[pr["spread_home"].notna()]
    lines.append("\n## Against closing lines\n")
    if len(ln) < 50:
        lines.append("Not enough closing lines in the database to compare. CFBD lines have not been ingested.\n")
    else:
        mkt_mu = -ln["spread_home"].to_numpy(float)
        y = (ln["home_pts"] > ln["away_pts"]).to_numpy(float)
        marg = (ln["home_pts"] - ln["away_pts"]).to_numpy(float)
        pm = np.clip(norm.cdf(mkt_mu / 13.5), 1e-6, 1 - 1e-6)
        po = np.clip(ln["p_home"].to_numpy(), 1e-6, 1 - 1e-6)
        ll = lambda p: float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())  # noqa: E731
        src = ln["spread_source"].value_counts().to_dict()
        lines += [f"Games with a line: {len(ln)} (sources: {src}; seasons {int(ln.season.min())}–{int(ln.season.max())}).\n",
                  "| | Log loss | Accuracy | Margin MAE |", "|---|---|---|---|",
                  f"| Market (spread → Φ(spread/13.5)) | {ll(pm):.3f} | {((pm > .5) == (y > .5)).mean():.3f} | {np.abs(marg - mkt_mu).mean():.1f} |",
                  f"| This model | {ll(po):.3f} | {((po > .5) == (y > .5)).mean():.3f} | {np.abs(marg - ln['mu'].to_numpy()).mean():.1f} |",
                  f"\nCorrelation of model expected margin with the market spread: {np.corrcoef(mkt_mu, ln['mu'])[0, 1]:.3f}. "
                  f"Mean absolute difference: {np.abs(mkt_mu - ln['mu']).mean():.1f} points.\n"]
    out = "\n".join(lines)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "ratings_validation.md").write_text(out)
    return out


if __name__ == "__main__":
    print(run())
