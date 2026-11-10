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
    reports/{season}_prediction_log.md with errors, biggest misses and a diagnosis line for each."""
    rel = data.releases()
    rel = rel[rel["season"] == season].sort_values("release_no")
    names = data.team_names()
    L = [f"# {season} prediction log\n",
         "Each prediction below was stored before the committee released that ranking, then scored against it.\n"]
    rows = []
    for r in rel.itertuples():
        f = PRED_DIR / f"{season}_release_{r.release_no}.csv"
        act = data.cfp_ranking(season, r.cutoff)
        actual = {names[int(t)]: int(k) for t, k in zip(act["team_id"], act["rank"])}
        if not f.exists():
            L.append(f"## Release {r.release_no} ({str(r.cutoff)[:10]})\n\nNo pre-release prediction was stored "
                     "(the project did not exist yet, or `cfp predict` was not run that week). A retrospective "
                     "out-of-sample prediction for it is in `reports/backtest_report.md`, section 8.\n")
            continue
        p = pd.read_csv(f)
        if pd.Timestamp(p["games_through_utc"].iloc[0], tz="UTC") > r.cutoff + pd.Timedelta(hours=20):
            L.append(f"## Release {r.release_no}\n\nStored prediction was made after the release; not scored.\n")
            continue
        pr = dict(zip(p["team"], p["rank"]))
        d = pd.DataFrame({"team": list(actual), "actual": list(actual.values())})
        d["predicted"] = d["team"].map(pr).fillna(31).astype(int)
        d["error"] = d["predicted"] - d["actual"]
        d = d.sort_values("actual")
        mae = d["error"].abs().mean()
        top12 = len(set(p[p["rank"] <= 12]["team"]) & {t for t, k in actual.items() if k <= 12})
        top4 = len(set(p[p["rank"] <= 4]["team"]) & {t for t, k in actual.items() if k <= 4})
        rows.append({"release": r.release_no, "date": str(r.cutoff)[:10], "mae": mae, "within1": (d.error.abs() <= 1).mean(),
                     "within2": (d.error.abs() <= 2).mean(), "top4": top4, "top12": top12})
        L.append(f"## Release {r.release_no} ({str(r.cutoff)[:10]})\n")
        L.append(f"Mean absolute error {mae:.2f} places; within one spot {(d.error.abs() <= 1).mean():.0%}; within two "
                 f"{(d.error.abs() <= 2).mean():.0%}; top 4 {top4}/4; top 12 {top12}/12. Prediction made {p['made_at_utc'].iloc[0]} UTC.\n")
        L += ["| Team | Actual | Predicted | Error |", "|---|---|---|---|"]
        L += [f"| {x.team} | {x.actual} | {x.predicted if x.predicted <= 30 else '31+'} | {x.error:+d} |" for x in d.itertuples()]
        big = d[d.error.abs() >= 3].reindex(d.error.abs().sort_values(ascending=False).index).dropna().head(8)
        if len(big):
            L.append("\nBiggest misses:\n")
            for x in big.itertuples():
                row = p[p["team"] == x.team]
                ctx = "" if row.empty else f" (AP {row['ap'].iloc[0] if pd.notna(row['ap'].iloc[0]) else 'NR'}, previous committee rank {row['prev'].iloc[0] if pd.notna(row['prev'].iloc[0]) else 'NR'}, record {row['record'].iloc[0]})"
                side = "too low" if x.error > 0 else "too high"
                L.append(f"- {x.team}: actual {x.actual}, predicted {x.predicted if x.predicted <= 30 else 'outside 30'}, {abs(x.error)} {side}{ctx}.")
        L.append("")
    if rows:
        s = pd.DataFrame(rows)
        L.insert(2, f"Season to date: {len(s)} scored release(s), mean absolute error {s.mae.mean():.2f}.\n")
        s.to_csv(REPORTS / f"{season}_prediction_scores.csv", index=False)
    out = "\n".join(L)
    (REPORTS / f"{season}_prediction_log.md").write_text(out)
    return out


def save_outputs(pred, sim) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sim_small = {"teams": sim.teams, "seeds": sim.seeds, "games": sim.games, "swing": sim.swing,
                 "title_games": sim.title_games, "proxy_log": sim.proxy_log, "n_sims": sim.n_sims, "seed": sim.seed,
                 "notes": sim.notes}
    (OUT / "latest.pkl").write_bytes(pickle.dumps({"pred": pred, "sim": sim_small,
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
