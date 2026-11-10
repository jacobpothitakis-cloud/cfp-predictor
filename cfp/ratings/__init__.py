"""Team-strength layer."""
from __future__ import annotations

from functools import lru_cache

from cfp.ratings.model import FCS_ID, RatingParams, Ratings, build_all, win_prob  # noqa: F401


@lru_cache(maxsize=1)
def load() -> Ratings:
    """Walk-forward ratings for every season in the database (rebuilt in ~2 s, so not persisted)."""
    from cfp import data
    p = RatingParams()
    return Ratings(build_all(data.games(), data.membership(), p), p)
