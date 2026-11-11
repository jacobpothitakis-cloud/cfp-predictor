"""`cfp update`: fetch, rebuild, retrain, score past predictions, predict, simulate, write the weekly report."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cfp import data, ratings
from cfp.committee import backtest as bt
from cfp.committee import dataset
from cfp.committee.train import train_production
from cfp.config import CURRENT_SEASON, DATA, REPORTS
from cfp.ingest import espn, http, load

OUT = DATA / "outputs"
PRED_DIR = REPORTS / "predictions"


def clear_caches() -> None:
    data.clear_cache()
    data.games.cache_clear()
    ratings.load.cache_clear()
    dataset.all_releases.cache_clear()
    from cfp.features import external
    external.clear()


def refresh(force: bool = True) -> dict[str, int]:
    """Re-fetch current-season pages and rebuild the database. Raises FetchError on any failed fetch."""
    espn.FORCE_CURRENT = force
    try:
        counts = load.build()
    finally:
        espn.FORCE_CURRENT = False
    clear_caches()
    return counts


def save_prediction(pred) -> Path:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    p = PRED_DIR / f"{pred.rel.season}_release_{pred.release_no}.csv"
    t = pred.table.drop(columns=["idx"]).copy()
    t.insert(0, "made_at_utc", str(pd.Timestamp.now(tz="UTC"))[:19])
    t.insert(1, "games_through_utc", str(pred.rel.st.cutoff)[:19])
    t.to_csv(p, index=False)
    return p


def score_predictions(season: int = CURRENT_SEASON) -> str:
    """Score every stored pre-release prediction whose actual ranking is now in the database, and write
    reports/{season}_prediction_log.md with errors against the AP poll, biggest misses and context for each."""
    from cfp.scoring import scored_releases
    L = [f"# {season} prediction log\n",
         "Each prediction below was stored before the committee released that ranking, then scored against it. "
         "The AP poll is the one that was current when the prediction was made. A team a source did not list counts as rank 31.\n"]
    rows = []
    show = lambda v: str(v) if v <= 30 else "NR"   # noqa: E731
    for e in scored_releases(season):
        if e["status"] == "no prediction":
            L.append(f"## Release {e['release']} ({e['date']})\n\nNo pre-release prediction was stored "
                     "(the project did not exist yet, or `cfp predict` was not run that week). A retrospective "
                     "out-of-sample prediction for it is in `reports/backtest_report.md`, section 8.\n")
            continue
        if e["status"] != "scored":
            L.append(f"## Release {e['release']}\n\nStored prediction was made after the release; not scored.\n")
            continue
        m, a, d = e["model"], e["ap"], e["table"]
        rows.append({"release": e["release"], "date": e["date"], "mae": m["mae"], "within1": m["within1"], "within2": m["within2"],
                     "top4": m["top4"], "top12": m["top12"], "ap_mae": a["mae"], "ap_within1": a["within1"], "ap_within2": a["within2"],
                     "ap_top4": a["top4"], "ap_top12": a["top12"], "prev_mae": e["prev"]["mae"] if e["prev"] else np.nan,
                     "closer_model": e["closer_model"], "closer_ap": e["closer_ap"], "in_range": e["in_range"]})
        L.append(f"## Release {e['release']} ({e['date']})\n")
        L.append(f"Prediction made {e['made']} UTC; AP poll dated {e['ap_date']}.\n")
        L += ["| | Model | AP poll |", "|---|---|---|",
              f"| Mean absolute error (places) | {m['mae']:.2f} | {a['mae']:.2f} |",
              f"| Within one spot | {m['within1']:.0%} | {a['within1']:.0%} |",
              f"| Within two spots | {m['within2']:.0%} | {a['within2']:.0%} |",
              f"| Top 4 | {m['top4']}/4 | {a['top4']}/4 |", f"| Top 12 | {m['top12']}/12 | {a['top12']}/12 |",
              f"| Closer on (teams) | {e['closer_model']} | {e['closer_ap']} |", ""]
        if e["prev"]:
            L.append(f"Last week's committee ranking left unchanged: {e['prev']['mae']:.2f}. "
                     f"The model's stated 80% range held the actual rank for {e['in_range']:.0%} of teams.\n")
        L += ["| Team | Actual | Model | Error | AP | AP error | That week |", "|---|---|---|---|---|---|---|"]
        L += [f"| {x.team} | {x.actual} | {show(x.model)} | {x.err_model:+d} | {show(x.ap)} | {x.err_ap:+d} | {x.week} |" for x in d.itertuples()]
        big = d[d.err_model.abs() >= 3].sort_values("err_model", key=abs, ascending=False).head(8)
        if len(big):
            L.append("\nBiggest misses:\n")
            for x in big.itertuples():
                side = "too low" if x.err_model > 0 else "too high"
                L.append(f"- {x.team} ({x.record}): actual {x.actual}, predicted {show(x.model)}, {abs(x.err_model)} {side} "
                         f"(AP {show(x.ap)}, previous committee rank {show(x.prev)}; {x.week}).")
        L.append("")
    if rows:
        s = pd.DataFrame(rows)
        L.insert(2, f"Season to date: {len(s)} scored release(s), mean absolute error {s.mae.mean():.2f} (AP poll {s.ap_mae.mean():.2f}).\n")
        s.to_csv(REPORTS / f"{season}_prediction_scores.csv", index=False)
    out = "\n".join(L)
    (REPORTS / f"{season}_prediction_log.md").write_text(out)
    return out


def save_outputs(pred, sim) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sim_small = {"teams": sim.teams, "seeds": sim.seeds, "games": sim.games, "swing": sim.swing,
                 "title_games": sim.title_games, "proxy_log": sim.proxy_log, "n_sims": sim.n_sims, "seed": sim.seed,
                 "notes": sim.notes}
    from cfp.fan import FAN_TEAMS, breakdown
    fan = {t: b for t in FAN_TEAMS if getattr(sim, "raw", None) and (b := breakdown(sim, t))}
    (OUT / "latest.pkl").write_bytes(pickle.dumps({"pred": pred, "sim": sim_small, "fan": fan,
                                                   "made": str(pd.Timestamp.now(tz='UTC'))[:19]}))


def load_outputs() -> dict | None:
    p = OUT / "latest.pkl"
    return pickle.loads(p.read_bytes()) if p.exists() else None


def run_update(n_sims: int = 10000, seed: int = 2026, fetch: bool = True) -> dict:
    from cfp.predict import predict_next
    from cfp.report_weekly import write_weekly
    from cfp.simulate.season import simulate
    log: list[str] = []
    if fetch:
        counts = refresh()
        log.append(f"Fetched and rebuilt: {counts['games']:,} games, {counts['releases']} committee rankings, "
                   f"{counts['score_conflicts']} score conflicts between sources.")
        if not os.environ.get("CFBD_API_KEY", "").strip():
            log.append("CFBD_API_KEY is not set: ESPN data only (no closing lines, SP+, SRS, Elo or advanced stats; "
                       "no two-source score reconciliation).")
    bundle = train_production()
    log.append(f"Committee models retrained on {bundle['n_releases']} rankings (through {bundle['trained_through']}).")
    score_predictions()
    pred = predict_next(bundle=bundle)
    save_prediction(pred)
    sim = simulate(n_sims=n_sims, seed=seed, bundle=bundle)
    save_outputs(pred, sim)
    path = write_weekly(pred, sim, log)
    return {"pred": pred, "sim": sim, "report": path, "log": log}
