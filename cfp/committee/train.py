"""Production models: trained on every release available, with calibrated noise scales."""
from __future__ import annotations

import pickle
from pathlib import Path

from cfp.committee import dataset
from cfp.committee.backtest import PRIMARY, SIM_MODEL, fit_models
from cfp.config import CURRENT_SEASON, DATA

MODEL_PATH = DATA / "models" / "committee.pkl"
SIGMA_PATH = DATA / "models" / "sigma.json"   # noise scales calibrated by the backtest report


def calibrated_sigmas() -> dict[str, float]:
    import json
    if SIGMA_PATH.exists():
        return json.loads(SIGMA_PATH.read_text())
    return {"primary": 1.4, "anchor": 1.7}


def save_sigmas(sig: dict[str, float]) -> None:
    import json
    SIGMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIGMA_PATH.write_text(json.dumps(sig, indent=1))


def train_production(sigmas: dict[str, float] | None = None, save: bool = True) -> dict:
    rels = list(dataset.all_releases())
    models = fit_models(rels, CURRENT_SEASON)
    bundle = {"primary": models[PRIMARY], "anchor": models["_anchor"],
              "level": models["Ensemble (résumé only)"],
              "pair_full": models["Pairwise logistic (+polls, last week)"],
              "pair_level": models["Pairwise logistic (résumé only)"], "pair_anchor": models["_anchor_pair"],
              "sigma": sigmas or calibrated_sigmas(), "n_releases": len(rels),
              "trained_through": max((r.season, r.release_no) for r in rels)}
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_bytes(pickle.dumps(bundle))
    return bundle


def load_production() -> dict:
    if not MODEL_PATH.exists():
        return train_production()
    return pickle.loads(MODEL_PATH.read_bytes())
