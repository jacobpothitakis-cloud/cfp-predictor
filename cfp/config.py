"""Paths, environment and season constants."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(os.environ.get("CFP_ROOT") or Path(__file__).resolve().parent.parent)
if not (ROOT / "data").exists() and (Path.cwd() / "data").exists():
    ROOT = Path.cwd()          # installed as a package and run from the project folder (hosted dashboard)
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
CACHE = DATA / "cache"
DB_PATH = DATA / "cfp.duckdb"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

FIRST_SEASON = 2014
CURRENT_SEASON = 2026

CFBD_BASE = "https://api.collegefootballdata.com"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"


def cfbd_key() -> str:
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "CFBD_API_KEY is not set. Put it in .env (see .env.example); it is never committed."
        )
    return key
