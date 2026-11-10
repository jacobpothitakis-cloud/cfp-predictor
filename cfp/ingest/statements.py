"""data/committee_statements.csv: what the committee chair said, sentence by sentence.

Built automatically from the ASAP Sports transcripts (Nov 2020 onward): every sentence spoken by a CFP
official (not the moderator, not a reporter) that matches a factor in the taxonomy below becomes one row
per (factor, team mentioned). `direction` is a keyword heuristic. Rows are tagged method=auto-keyword;
this is a searchable index of real quotes with their source URL, not a hand-coded dataset.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from cfp import data
from cfp.config import CACHE, DATA

FACTORS = {
    "strength_of_schedule": r"strength of schedule|schedule strength|schedule (isn't|is not|wasn't) as strong|tough(er|est)? schedule|who they('ve| have) played",
    "record_strength": r"record strength|strength of record",
    "quality_wins": r"quality win|signature win|good win|great win|big win|wins? (against|over|versus|vs\.?) (a |the )?(ranked|top)|ranked wins?|wins? over .{0,40}(ranked|top[- ]?\d+)|top[- ]?25 (win|team|opponent)|\.500|winning record",
    "loss_quality": r"(bad|only|lone|one|two|three) loss|loss(es)? (to|against|at|versus)|lost to|unranked",
    "head_to_head": r"head[- ]to[- ]head",
    "common_opponents": r"common opponent",
    "conference_championship": r"conference champion|championship game|won the (sec|acc|big ten|big 12|pac-12|league|conference)|champions?\b",
    "game_control": r"game control|controlled the game|in control|dominat",
    "eye_test": r"eye test|watch(ed|ing)? (them|the games?|.{0,25} play)|on (film|tape|video)|when (you|we) watch|looked (good|impressive|great)",
    "complete_team": r"complete team|both sides of the ball|balanced|complementary football|all three phases",
    "line_of_scrimmage": r"lines? of scrimmage|up front|offensive line|defensive line|in the trenches",
    "body_of_work": r"body of work|full (resume|résumé|profile)|entire (resume|résumé|season)|whole (resume|résumé|body)",
    "margin": r"margin of victory|scoring margin|margin|blowout|won by \d+|\d+-plus points",
    "availability": r"injur|unavailab|without (their|its|his) |quarterback (was|is|went) (out|down)|backup quarterback|full strength|suspension|opt(ed|ing)? out",
    "road_wins": r"on the road|road win|true road|away from home",
    "recency": r"playing (really )?(well|their best)|right now|of late|lately|momentum|improv|trending|down the stretch|last (three|four|few) (games|weeks)",
    "efficiency_stats": r"relative scoring|offensive efficiency|defensive efficiency|statistic|scoring (offense|defense)|yards per",
    "conference_strength": r"strength of (the |their )?conference|conference strength|strength of the (league|sec|big ten|big 12|acc)",
    "group_of_five": r"group of (five|5|six|6)|g5|g6",
}
POS = r"impressive|strong|quality|credit|reward|dominant|dominat|great|really good|well|respect|signature|better"
NEG = r"\bloss|lost|struggl|not as strong|isn't as strong|lack|concern|penaliz|drop|fell|behind|inconsisten|weak|didn't|did not"
SPEAKER = re.compile(r"^([A-Z][A-Z\.\' -]{3,40}):\s*(.*)$")


def _answers(text: str) -> list[tuple[str, str]]:
    """(speaker, paragraph) for every paragraph spoken by someone other than the moderator/reporters."""
    out, speaker = [], None
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Q.") or line == "Q":
            speaker = None
            continue
        m = SPEAKER.match(line)
        if m:
            speaker = None if "MODERATOR" in m.group(1) else m.group(1).title()
            line = m.group(2)
        if speaker and line:
            out.append((speaker, line))
    return out


def build() -> pd.DataFrame:
    calls = json.loads((CACHE / "transcripts" / "committee_calls.json").read_text())
    mem = data.membership()
    names = sorted(set(mem["team"]), key=len, reverse=True)
    alias = {"Southern Cal": "USC", "Southern California": "USC", "Mississippi": "Ole Miss", "Pitt": "Pittsburgh"}
    pat = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(n) for n in names + list(alias)) + r")(?![A-Za-z])")
    rel = data.releases()
    rows = []
    for c in calls:
        date = pd.Timestamp(c["date"], tz="UTC")
        season = date.year if date.month >= 8 else date.year - 1
        r = rel[(rel["season"] == season) & (rel["cutoff"] <= date + pd.Timedelta(days=1))]
        release_no = int(r["release_no"].max()) if len(r) else None
        week = int(r.sort_values("cutoff")["week"].iloc[-1]) if len(r) else None
        for speaker, para in _answers(c["text"]):
            para_teams = {alias.get(t, t) for t in pat.findall(para)}
            for sent in re.split(r"(?<=[\.\?\!])\s+", para):
                low = sent.lower()
                hits = [f for f, rx in FACTORS.items() if re.search(rx, low)]
                if not hits:
                    continue
                teams = {alias.get(t, t) for t in pat.findall(sent)}
                if not teams and len(para_teams) == 1:
                    teams = para_teams
                pos, neg = bool(re.search(POS, low)), bool(re.search(NEG, low))
                direction = "positive" if pos and not neg else "negative" if neg and not pos else "mixed" if pos else "neutral"
                for f in hits:
                    for t in (sorted(teams) or [""]):
                        rows.append({"season": season, "week": week, "release_no": release_no, "date": c["date"],
                                     "speaker": speaker, "team": t, "factor": f, "direction": direction,
                                     "quote": sent.strip()[:400], "url": c["url"], "method": "auto-keyword"})
    df = pd.DataFrame(rows).drop_duplicates(["date", "team", "factor", "quote"])
    df.to_csv(DATA / "committee_statements.csv", index=False)
    return df


if __name__ == "__main__":
    d = build()
    print(len(d), "rows;", d["date"].nunique(), "calls;", d["season"].min(), "-", d["season"].max())
    print(d.groupby(["factor", "season"]).size().unstack(fill_value=0).to_string())
    print(d.speaker.value_counts().head(12).to_string())
