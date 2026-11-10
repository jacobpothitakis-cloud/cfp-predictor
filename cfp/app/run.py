"""Entry point: `cfp-dashboard` starts the Streamlit app."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).with_name("dashboard.py")
    root = app.parent.parent.parent          # project folder, so .streamlit/config.toml (theme) is picked up
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]], cwd=root))
