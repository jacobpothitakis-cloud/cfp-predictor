"""Disk-cached HTTP GET. A response that is on disk is never fetched again unless force=True."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests

from cfp.config import CACHE

UA = "cfp-predictor/0.1 (python-requests)"
_LOG = CACHE / "fetch_log.jsonl"


class FetchError(RuntimeError):
    """A fetch failed. Callers must stop the step and report; never substitute remembered data."""


def _key(url: str, params: dict[str, Any] | None) -> str:
    blob = url + "?" + json.dumps(params or {}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def cache_path(namespace: str, name: str, url: str, params: dict[str, Any] | None, ext: str) -> Path:
    d = CACHE / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}_{_key(url, params)}.{ext}"


def get(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    namespace: str,
    name: str,
    headers: dict[str, str] | None = None,
    as_json: bool = True,
    force: bool = False,
    max_age_s: float | None = None,
    retries: int = 3,
) -> Any:
    """GET with a permanent disk cache.

    max_age_s: if set, a cached copy older than this is refreshed (used only for
    current-season data that changes week to week).
    """
    path = cache_path(namespace, name, url, params, "json" if as_json else "txt")
    if path.exists() and not force:
        fresh = max_age_s is None or (time.time() - path.stat().st_mtime) < max_age_s
        if fresh:
            text = path.read_text(encoding="utf-8")
            return json.loads(text) if as_json else text
    hdrs = {"User-Agent": UA, "Accept": "application/json" if as_json else "*/*"}
    hdrs.update(headers or {})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=hdrs, timeout=60)
            if r.status_code == 429:
                raise FetchError(f"429 rate limited: {url}")
            if r.status_code in (401, 403):
                raise FetchError(f"{r.status_code} not authorised: {url}")
            r.raise_for_status()
            if as_json:
                data = r.json()
                path.write_text(json.dumps(data), encoding="utf-8")
            else:
                data = r.text
                path.write_text(data, encoding="utf-8")
            remaining = r.headers.get("X-CallLimit-Remaining")
            with _LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "url": url, "params": params or {},
                                    "namespace": namespace, "remaining": remaining}) + "\n")
            return data
        except FetchError:
            raise
        except Exception as e:  # network / 5xx: retry
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise FetchError(f"fetch failed after {retries} tries: {url} params={params} ({last})")
