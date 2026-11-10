"""Committee-chair media-conference transcripts (ASAP Sports archive, Nov 2020 onward).

The archive index lists every CFP media conference; we keep only the selection-committee
calls (ranking weeks and Selection Day), not team press conferences.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from cfp.config import CACHE
from cfp.ingest import http

ASAP_ROOT = "https://asaptext.com/asap_media/media/1070/1122/"
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}


def fetch_index(force: bool = False) -> list[dict]:
    html = http.get(ASAP_ROOT, namespace="transcripts", name="asap_index", as_json=False,
                    headers=BROWSER_UA, force=force, max_age_s=6 * 3600)
    soup = BeautifulSoup(html, "lxml")
    toks: list[tuple[str, str]] = []
    for el in soup.body.descendants:
        if getattr(el, "name", None) == "a" and el.get("href", "").startswith("browse_file"):
            m = re.search(r"(\d+)\.html", el["href"])
            if m:
                toks.append(("L", m.group(1)))
        elif isinstance(el, str) and el.strip():
            toks.append(("T", el.strip()))
    rows, buf, cur_date = [], [], None
    for kind, val in toks:
        if kind == "T":
            if val not in ("HTML", "PDF"):
                buf.append(val)
            continue
        dates = [b for b in buf if re.match(r"^[A-Z][a-z]+ \d{2}, \d{4}$", b)]
        if dates:
            cur_date = dates[-1]
        rest = [b for b in buf if b not in dates and not re.match(r"^\d\d:\d\d:\d\d$", b)]
        rows.append({"id": val, "date": datetime.strptime(cur_date, "%B %d, %Y").date().isoformat()
                     if cur_date else None, "label": " | ".join(rest)})
        buf = []
    return rows


def is_committee_call(row: dict) -> bool:
    lab = row["label"]
    if " - " in lab.split("|")[0]:          # "Ohio State Buckeyes - CFP Media Conference"
        return False
    if not row["date"]:
        return False
    month = int(row["date"][5:7])
    return month in (11, 12) and ("CFP" in lab or "College Football Playoff" in lab)


def fetch_transcript(tid: str) -> str:
    url = f"{ASAP_ROOT}transcripts/{tid}.html"
    html = http.get(url, namespace="transcripts", name=f"asap_{tid}", as_json=False, headers=BROWSER_UA)
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))


def fetch_all() -> list[dict]:
    out = []
    for row in fetch_index():
        if not is_committee_call(row):
            continue
        text = fetch_transcript(row["id"])
        out.append({**row, "url": f"{ASAP_ROOT}transcripts/{row['id']}.html", "text": text})
    (CACHE / "transcripts").mkdir(parents=True, exist_ok=True)
    Path(CACHE / "transcripts" / "committee_calls.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    calls = fetch_all()
    print(len(calls), "committee calls")
    for c in calls:
        print(c["date"], c["id"], len(c["text"]), c["label"][:60])
