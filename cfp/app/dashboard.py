"""Local or hosted dashboard:  streamlit run cfp/app/dashboard.py   (or: cfp-dashboard)"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

warnings.filterwarnings("ignore")
from cfp import data  # noqa: E402
from cfp.config import REPORTS  # noqa: E402
from cfp.pipeline import OUT, load_outputs  # noqa: E402
from cfp.predict import QF_DATE, bowl_paths, game_log  # noqa: E402
from cfp.ratings.model import RatingParams  # noqa: E402

st.set_page_config(page_title="CFP Committee Predictor", page_icon="🏈", layout="wide")

NAVY, GOLD, GREEN, RED, GREY = "#14213d", "#b8862b", "#1f8a4c", "#c0392b", "#8a94a6"
CONF = {"sec": "SEC", "big10": "Big Ten", "big12": "Big 12", "acc": "ACC", "ind": "Independent",
        "american": "American", "mwest": "Mountain West", "pac12": "Pac-12", "usa": "Conference USA",
        "midam": "MAC", "belt": "Sun Belt"}
FACTOR = {
    "last week's committee position": "Where the committee had them last week",
    "AP and Coaches polls": "Standing in the AP and Coaches polls",
    "losses and record": "Win-loss record",
    "team strength and dominance": "How dominant they have looked",
    "quality of wins": "Quality of wins",
    "quality of losses": "Who they lost to",
    "schedule strength and strength of record": "Schedule difficulty",
    "recent form and this week's result": "Recent form",
    "conference strength and title game": "Conference strength",
    "power / Group of Six status": "Conference tier",
    "preseason reputation": "Preseason reputation",
}

st.markdown(f"""
<style>
  .block-container, [data-testid="stMainBlockContainer"] {{ padding: 3.6rem 2.2rem 3rem 2.2rem !important; max-width: 1560px !important; }}
  #MainMenu, footer {{ visibility: hidden; }}
  .hero {{ background: linear-gradient(120deg, {NAVY} 0%, #24365f 70%, #2f4a80 100%); color: #fff;
          border-radius: 14px; padding: 22px 28px; margin-bottom: 14px; }}
  .hero h1 {{ color: #fff; font-size: 1.9rem; margin: 0 0 4px 0; letter-spacing: .2px; }}
  .hero p {{ color: #d5dcec; margin: 0; font-size: .95rem; }}
  .hero .tag {{ display: inline-block; background: {GOLD}; color: #fff; border-radius: 999px; padding: 2px 12px;
               font-size: .75rem; font-weight: 600; letter-spacing: .6px; text-transform: uppercase; margin-bottom: 8px; }}
  .kpi {{ background: #fff; border: 1px solid #e3e7ee; border-radius: 12px; padding: 12px 16px; height: 100%; }}
  .kpi .l {{ color: {GREY}; font-size: .72rem; text-transform: uppercase; letter-spacing: .7px; font-weight: 600; }}
  .kpi .v {{ color: {NAVY}; font-size: 1.25rem; font-weight: 700; display: flex; align-items: center; gap: 8px; margin-top: 2px; }}
  .kpi .s {{ color: {GREY}; font-size: .8rem; }}
  .kpi img {{ height: 28px; }}
  .sec {{ color: {NAVY}; font-size: 1.15rem; font-weight: 700; margin: 18px 0 6px 0; border-left: 4px solid {GOLD}; padding-left: 10px; }}
  .note {{ color: {GREY}; font-size: .85rem; margin-bottom: 8px; }}
  .chip {{ display: inline-block; border-radius: 8px; padding: 6px 10px; margin: 3px 4px 3px 0; font-size: .86rem; line-height: 1.3; }}
  .up {{ background: #e9f6ee; color: #14613a; border: 1px solid #cfe9d9; }}
  .down {{ background: #fbebe9; color: #8f2a20; border: 1px solid #f3cfca; }}
  .bracket {{ display: flex; gap: 14px; align-items: stretch; overflow-x: auto; padding-bottom: 6px; }}
  .round {{ flex: 1 1 0; min-width: 215px; display: flex; flex-direction: column; justify-content: space-around; gap: 12px; }}
  .round h4 {{ color: {GREY}; font-size: .72rem; text-transform: uppercase; letter-spacing: .8px; margin: 0 0 2px 2px; }}
  .game {{ background: #fff; border: 1px solid #e3e7ee; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 2px rgba(20,33,61,.05); }}
  .game .meta {{ background: #f1f3f7; color: {GREY}; font-size: .7rem; padding: 4px 10px; text-transform: uppercase; letter-spacing: .5px; font-weight: 600; }}
  .slot {{ display: flex; align-items: center; gap: 8px; padding: 7px 10px; font-size: .92rem; color: {NAVY}; font-weight: 600; }}
  .slot + .slot {{ border-top: 1px solid #eef0f4; }}
  .slot .sd {{ background: {NAVY}; color: #fff; border-radius: 5px; font-size: .7rem; min-width: 22px; text-align: center; padding: 2px 0; }}
  .slot img {{ height: 22px; width: 22px; object-fit: contain; }}
  .slot.tbd {{ color: {GREY}; font-weight: 500; font-style: italic; }}
  .th {{ display: flex; align-items: center; gap: 16px; background: #fff; border: 1px solid #e3e7ee; border-radius: 14px; padding: 16px 20px; }}
  .th img {{ height: 72px; }}
  .th h2 {{ margin: 0; color: {NAVY}; font-size: 1.6rem; }}
  .th p {{ margin: 2px 0 0 0; color: {GREY}; }}
  .fan {{ display: flex; align-items: center; gap: 18px; background: linear-gradient(120deg, #111 0%, #2a2a2a 100%); border-radius: 14px;
          padding: 18px 24px; border-bottom: 5px solid #ffcd00; }}
  .fan img {{ height: 84px; }}
  .fan h2 {{ margin: 0; color: #ffcd00; font-size: 1.7rem; }}
  .fan p {{ margin: 3px 0 0 0; color: #e6e6e6; font-size: .95rem; }}
  .step {{ background: #fff; border: 1px solid #e3e7ee; border-left: 5px solid #ffcd00; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;
           color: {NAVY}; font-size: .95rem; line-height: 1.4; }}
  .step b.n {{ display: inline-block; background: #111; color: #ffcd00; border-radius: 50%; width: 22px; height: 22px; text-align: center;
               font-size: .8rem; line-height: 22px; margin-right: 8px; }}
  table.seedgrid {{ border-collapse: collapse; width: 100%; font-size: .85rem; }}
  table.seedgrid th {{ color: {GREY}; font-weight: 600; padding: 4px 6px; text-align: center; }}
  table.seedgrid td {{ padding: 5px 6px; text-align: center; border-bottom: 1px solid #eef0f4; color: {NAVY}; }}
  table.seedgrid td.t {{ text-align: left; font-weight: 600; white-space: nowrap; }}
  table.seedgrid td.t img {{ height: 18px; vertical-align: middle; margin-right: 6px; }}
  @media (max-width: 640px) {{
    .hero {{ padding: 16px 18px; }} .hero h1 {{ font-size: 1.3rem; }} .hero p {{ font-size: .85rem; }}
    .bracket {{ flex-direction: column; }} .round {{ min-width: 0; }}
    .th img {{ height: 48px; }} .th h2 {{ font-size: 1.2rem; }} .kpi {{ margin-bottom: 8px; }}
  }}
  div[data-testid="stMetric"] {{ background: #fff; border: 1px solid #e3e7ee; border-radius: 12px; padding: 10px 14px; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _outputs(stamp: float):
    """Loaded once per data refresh and shared by every visitor (keeps CPU use low on a shared host).
    A hosted app keeps running when new data is pushed, so the in-memory copies of the database tables
    are dropped here; otherwise the site would keep showing the previous week's committee ranking."""
    from cfp.pipeline import clear_caches
    clear_caches()
    return load_outputs()


@st.cache_data(show_spinner=False)
def _last_ranking(season: int, stamp: float) -> tuple[pd.DataFrame, str]:
    rel = data.releases()
    rel = rel[rel.season == season].sort_values("release_no")
    if rel.empty:
        return pd.DataFrame(), ""
    names = data.team_names()
    last = data.cfp_ranking(season, rel.cutoff.iloc[-1])
    return pd.DataFrame({"rank": last["rank"].astype(int).to_numpy(), "team": [names[int(x)] for x in last["team_id"]],
                         "record": last["record"].to_numpy()}), str(rel.cutoff.iloc[-1])[:10]


@st.cache_data(show_spinner=False)
def _scored(season: int, stamp: float) -> list[dict]:
    from cfp.scoring import scored_releases
    return [e for e in scored_releases(season) if e["status"] == "scored"]


_f = OUT / "latest.pkl"
from cfp.config import DB_PATH  # noqa: E402
STAMP = max(_f.stat().st_mtime if _f.exists() else 0.0, DB_PATH.stat().st_mtime if DB_PATH.exists() else 0.0)
out = _outputs(STAMP) if _f.exists() else None
if out is None:
    st.error("No results yet. Run `cfp update` first.")
    st.stop()
pred, sim = out["pred"], SimpleNamespace(**out["sim"])
S = pred.rel.st
IDX = {n: i for i, n in enumerate(S.names)}
last_df, last_date = _last_ranking(S.season, STAMP)
last_rank = dict(zip(last_df.get("team", []), last_df.get("rank", [])))


def logo(team: str) -> str:
    return f"https://a.espncdn.com/i/teamlogos/ncaa/500/{int(S.team_ids[IDX[team]])}.png" if team in IDX else ""


def pct(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return "<1%" if 0 < x < 0.005 else f"{x:.0%}"


def section(title: str, note: str = "") -> None:
    st.markdown(f'<div class="sec">{title}</div>' + (f'<div class="note">{note}</div>' if note else ""), unsafe_allow_html=True)


def kpi(col, label: str, value: str, sub: str = "", team: str | None = None) -> None:
    img = f'<img src="{logo(team)}">' if team else ""
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{img}{value}</div><div class="s">{sub}</div></div>',
                 unsafe_allow_html=True)


def strength(v: float) -> str:
    a = abs(v)
    return "major factor" if a >= 2.5 else "clear factor" if a >= 1.0 else "minor factor"


def reasons(e: dict) -> str:
    html = ""
    for kind, items in (("up", e["up"]), ("down", e["down"])):
        for g, v, fact in items:
            if g.startswith("recent form") and ((v < 0 and "won this week" in fact) or (v > 0 and "lost this week" in fact)):
                fact = ""
            arrow = "▲" if kind == "up" else "▼"
            html += f'<span class="chip {kind}"><b>{arrow} {FACTOR.get(g, g)}</b> · {strength(v)}{"<br>" + fact if fact else ""}</span>'
    return html or '<span class="note">Nothing stands out either way.</span>'


def move(team: str, rank: int) -> str:
    p = last_rank.get(team)
    if p is None:
        return "NEW"
    return f"▲ {p - rank}" if p > rank else f"▼ {rank - p}" if p < rank else "—"


# ----------------------------------------------------------------------------- header
made = pd.Timestamp(out["made"], tz="UTC").tz_convert("America/New_York").strftime("%b %-d, %-I:%M %p ET")
label = "Selection Day" if pred.is_final else pd.Timestamp(pred.release_date).strftime("%A, %b %-d")
st.markdown(f"""<div class="hero"><span class="tag">{S.season} season · ranking #{pred.release_no}</span>
<h1>College Football Playoff Committee Predictor</h1>
<p>What the selection committee is expected to do: its next Top 25 ({label}), the 12-team field and every team's playoff odds.
Updated {made}.</p></div>""", unsafe_allow_html=True)

fld = pred.field
tbl = pred.table
field_names = [S.names[t] for t in fld.seeds] if fld and len(fld.seeds) == 12 else []
at_large = [n for n in field_names if IDX[n] not in fld.auto] if field_names else []
first_out = next((r.team for r in tbl.itertuples() if r.team not in field_names), "")
fav = sim.teams.sort_values("p_champion", ascending=False).iloc[0]
k = st.columns(5)
kpi(k[0], "Projected No. 1", tbl.team.iloc[0], f"{tbl.record.iloc[0]} · {pct(tbl.p_exact.iloc[0])} to be No. 1", tbl.team.iloc[0])
kpi(k[1], "Last team in", at_large[-1] if at_large else "n/a", "lowest at-large seed today", at_large[-1] if at_large else None)
kpi(k[2], "First team out", first_out or "n/a", "next in the predicted ranking", first_out or None)
kpi(k[3], "Title favourite", fav.team, f"{fav.p_champion:.0%} to win it all", fav.team)
kpi(k[4], "Simulated seasons", f"{sim.n_sims:,}", "rest of season played out")
st.write("")

FAN = next(iter(out.get("fan") or {}), None)
tabs = st.tabs(["Rankings", "Bracket", "Playoff odds", "Teams", "What-if", "Report card", "Track record"] + ([FAN] if FAN else []))

# ----------------------------------------------------------------------------- rankings
with tabs[0]:
    c1, c2 = st.columns([11, 4])
    with c1:
        section("Predicted committee Top 25", "The range is where the team lands in 8 of 10 cases.")
        t = tbl.head(25)
        show = pd.DataFrame({
            "#": t["rank"], " ": [logo(x) for x in t.team], "Team": t.team, "Record": t.record,
            "Conf": [CONF.get(c, c) for c in t.conf], "Move": [move(a, b) for a, b in zip(t.team, t["rank"])],
            "AP": t["ap"].map(lambda x: "NR" if pd.isna(x) else str(int(x))),
            "This exact spot": t.p_exact * 100, "Range": t.low.astype(str) + "–" + t.high.astype(str),
            "Top 4": t.p_top4 * 100, "Top 12": t.p_top12 * 100})
        prog = lambda: st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)   # noqa: E731
        st.dataframe(show, hide_index=True, use_container_width=True, height=915,
                     column_config={"#": st.column_config.NumberColumn(width=38), " ": st.column_config.ImageColumn(width=44),
                                    "Team": st.column_config.TextColumn(width=150), "Record": st.column_config.TextColumn(width=62),
                                    "Conf": st.column_config.TextColumn(width=104), "Move": st.column_config.TextColumn(width=58),
                                    "AP": st.column_config.TextColumn(width=44), "Range": st.column_config.TextColumn(width=66),
                                    "This exact spot": prog(), "Top 4": prog(), "Top 12": prog()})
        nxt = tbl.iloc[25:30]
        st.markdown('<div class="note">Just outside: ' + ", ".join(f"{r.team} ({pct(r.p_top25)} to be ranked)" for r in nxt.itertuples()) + "</div>",
                    unsafe_allow_html=True)
    with c2:
        section(f"Committee's last ranking", last_date and pd.Timestamp(last_date).strftime("Released %b %-d"))
        if len(last_df):
            st.dataframe(pd.DataFrame({"#": last_df["rank"], " ": [logo(x) for x in last_df.team], "Team": last_df.team,
                                       "Record": last_df.record}),
                         hide_index=True, use_container_width=True, height=915,
                         column_config={"#": st.column_config.NumberColumn(width=38), " ": st.column_config.ImageColumn(width=44),
                                        "Team": st.column_config.TextColumn(width=150), "Record": st.column_config.TextColumn(width=62)})
    section("Why each team is where it is", "Green helps the team, red holds it back. Based on how past committees have weighed the same things.")
    cols = st.columns(2)
    for i, (nm, e) in enumerate(pred.explanations.items()):
        with cols[i % 2].expander(f"{e['rank']}.  {nm}", expanded=i < 4):
            st.markdown(reasons(e), unsafe_allow_html=True)
    with st.expander("Team against team: why each is ahead of the next"):
        for p in pred.pairs:
            txt = p.split(" (gap ")[0]
            call = "close call" if "close call" in p else "clear" if "clear)" in p else "moderate gap"
            for g, lab in FACTOR.items():
                txt = txt.replace(g, lab.lower())
            st.markdown(f"- {txt} — *{call}*")
    section("How sure is the model about one team?")
    pick = st.selectbox("Team", list(pred.dist.index), label_visibility="collapsed")
    dd = pred.dist.loc[pick]
    d = pd.DataFrame({"Rank": list(dd.index), "Chance": dd.to_numpy(), "o": range(len(dd))})
    d = d[d.Chance > 0.002]
    st.altair_chart(alt.Chart(d).mark_bar(color=NAVY, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("Rank:N", sort=alt.SortField("o"), title="Committee rank"),
        y=alt.Y("Chance:Q", axis=alt.Axis(format="%"), title=None), tooltip=[alt.Tooltip("Rank:N"), alt.Tooltip("Chance:Q", format=".0%")]
    ).properties(height=230), use_container_width=True)

# ----------------------------------------------------------------------------- bracket
with tabs[1]:
    section("The bracket if the season ended today",
            "Predicted ranking, with each conference's current leader standing in as champion. Which bowl each of the top three seeds picks is our assumption.")
    if field_names:
        seeds = fld.seeds
        bowls = bowl_paths(seeds, S)

        def slot(seed: int) -> str:
            n = S.names[seeds[seed - 1]]
            return f'<div class="slot"><span class="sd">{seed}</span><img src="{logo(n)}">{n}</div>'

        def tbd(text: str) -> str:
            return f'<div class="slot tbd">{text}</div>'

        def game(meta: str, a: str, b: str) -> str:
            return f'<div class="game"><div class="meta">{meta}</div>{a}{b}</div>'

        r1 = "".join(game("Dec 18–19 · on campus", slot(h), slot(v)) for h, v in ((8, 9), (5, 12), (7, 10), (6, 11)))
        qf = "".join(game(f"{bowls[top]} Bowl · {QF_DATE[bowls[top]]}", slot(top), tbd(f"Winner of {h} v {v}"))
                     for top, h, v in ((1, 8, 9), (4, 5, 12), (2, 7, 10), (3, 6, 11)))
        sf = game("Semifinal · Jan 14 or 15", tbd("1 / 8 / 9 side"), tbd("4 / 5 / 12 side")) + \
            game("Semifinal · Jan 14 or 15", tbd("2 / 7 / 10 side"), tbd("3 / 6 / 11 side"))
        fin = game("Championship · Jan 25 · Las Vegas", tbd("Semifinal winner"), tbd("Semifinal winner"))
        st.markdown(f'<div class="bracket"><div class="round"><h4>First round</h4>{r1}</div>'
                    f'<div class="round"><h4>Quarterfinals</h4>{qf}</div><div class="round"><h4>Semifinals</h4>{sf}</div>'
                    f'<div class="round"><h4>Final</h4>{fin}</div></div>', unsafe_allow_html=True)
        st.write("")
        a, b = st.columns(2)
        with a:
            section("Automatic bids")
            for t, why in fld.auto.items():
                w = why.replace("acc", "ACC").replace("big10", "Big Ten").replace("big12", "Big 12").replace("sec", "SEC")
                st.markdown(f"- **{S.names[t]}** — {w}")
        with b:
            section("On the outside")
            outs = [r.team for r in tbl.itertuples() if r.team not in field_names][:4]
            for n in outs:
                bumped = IDX[n] in fld.bumped
                st.markdown(f"- **{n}** — predicted No. {int(tbl[tbl.team == n]['rank'].iloc[0])}" +
                            (", inside the top 12 but pushed out by an automatic bid" if bumped else ""))
        st.caption("Semifinals are the Orange Bowl (Jan 14) and Sugar Bowl (Jan 15); the No. 1 seed chooses which one its side plays in. "
                   "There is no reseeding.")
    section("Where each team is likely to be seeded", f"Share of {sim.n_sims:,} simulated seasons.")
    top = list(sim.teams[sim.teams.p_field >= 0.05]["team"])
    rows = ""
    for n in top:
        cells = "".join(f'<td style="background: rgba(20,33,61,{min(v * 2.2, .85):.2f}); color: {"#fff" if v > .22 else NAVY}">{pct(v) if v >= .005 else ""}</td>'
                        for v in sim.seeds.loc[n])
        rows += f'<tr><td class="t"><img src="{logo(n)}">{n}</td>{cells}</tr>'
    st.markdown('<div style="overflow-x:auto"><table class="seedgrid"><tr><th></th>' + "".join(f"<th>{k}</th>" for k in range(1, 13)) +
                f"</tr>{rows}</table></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- odds
with tabs[2]:
    section("Chance of making the 12-team field", f"From {sim.n_sims:,} simulated seasons: every remaining game, conference race and title game, then the committee's final ranking.")
    t = sim.teams[sim.teams.p_field >= 0.02].copy()
    t["status"] = np.where(t.p_field >= 0.75, "Likely in", np.where(t.p_field >= 0.25, "On the bubble", "Long shot"))
    chart = alt.Chart(t).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        y=alt.Y("team:N", sort=alt.SortField("p_field", order="descending"), title=None),
        x=alt.X("p_field:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title=None),
        color=alt.Color("status:N", scale=alt.Scale(domain=["Likely in", "On the bubble", "Long shot"], range=[GREEN, GOLD, GREY]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip("p_field:Q", format=".0%", title="Make field"),
                 alt.Tooltip("p_bye:Q", format=".0%", title="Bye"), alt.Tooltip("p_champion:Q", format=".1%", title="Win title")])
    rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4], color=NAVY).encode(x="x:Q")
    st.altair_chart((chart + rule).properties(height=max(26 * len(t), 300)), use_container_width=True)
    section("The full picture")
    full = sim.teams[sim.teams.p_field >= 0.005]
    d = pd.DataFrame({" ": [logo(x) for x in full.team], "Team": full.team, "Make field": full.p_field * 100, "First-round bye": full.p_bye * 100,
                      "Host a game": full.p_host * 100, "Win conference": full.p_conf_champ * 100,
                      "Win title": full.p_champion * 100, "Typical final rank": full.mean_rank.round(1),
                      "In if they win out": full.field_if_win_out.map(pct), "In with one more loss": full.field_if_one_more_loss.map(pct)})
    prog = lambda: st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)   # noqa: E731
    st.dataframe(d, hide_index=True, use_container_width=True, height=min(38 * len(d) + 40, 760),
                 column_config={" ": st.column_config.ImageColumn(width="small"), "Make field": prog(), "First-round bye": prog(),
                                "Host a game": prog(), "Win conference": prog(),
                                "Win title": st.column_config.NumberColumn(format="%.1f%%")})
    st.caption("'n/a' means too few simulations met the condition to give a fair number.")
    a, b = st.columns(2)
    with a:
        section("Games that matter most", "How much the playoff picture changes depending on who wins.")
        if len(sim.games):
            g = sim.games.head(12)
            st.dataframe(pd.DataFrame({"Game": g.game, "Date": pd.to_datetime(g.date).dt.strftime("%b %-d"),
                                       "Home win": g.p_home_win * 100, "Impact": g.expected_impact.round(2),
                                       "Biggest effect": g.most_affected.str.split(";").str[0]}),
                         hide_index=True, use_container_width=True,
                         column_config={"Home win": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})
    with b:
        section("Most likely conference title games", "Top seed listed first.")
        tg = sim.title_games.copy()
        tg = tg.loc[tg.groupby("conf")["p"].idxmax()].sort_values("p", ascending=False)
        st.dataframe(pd.DataFrame({"Conference": [CONF.get(c, c) for c in tg.conf], "Matchup": tg.matchup, "Chance": tg.p * 100,
                                   "Site": tg.site}), hide_index=True, use_container_width=True,
                     column_config={"Chance": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})

# ----------------------------------------------------------------------------- teams
with tabs[3]:
    ranked = list(tbl.team)
    nm = st.selectbox("Choose a team", ranked + sorted(n for n in S.names if n not in ranked))
    ti = IDX[nm]
    raw = pred.rel.raw.iloc[ti]
    prow = tbl[tbl.team == nm]
    srow = sim.teams[sim.teams.team == nm]
    prk = f"No. {int(prow['rank'].iloc[0])}" if len(prow) else "Unranked"
    st.markdown(f"""<div class="th"><img src="{logo(nm)}"><div><h2>{nm}</h2>
<p>{int(raw.wins)}–{int(raw.losses)} overall · {int(raw.conf_wins)}–{int(raw.conf_losses)} {CONF.get(S.conf[ti], S.conf[ti])} · predicted {prk}</p></div></div>""",
                unsafe_allow_html=True)
    st.write("")
    m = st.columns(6)
    m[0].metric("Predicted rank", prk.replace("No. ", "#"))
    m[1].metric("Committee last week", f"#{last_rank[nm]}" if nm in last_rank else "NR")
    m[2].metric("AP poll", "NR" if np.isnan(S.ap[ti]) else f"#{int(S.ap[ti])}")
    if len(srow):
        r = srow.iloc[0]
        m[3].metric("Make the playoff", pct(r.p_field))
        m[4].metric("First-round bye", pct(r.p_bye))
        m[5].metric("Win the title", f"{r.p_champion:.1%}")
    else:
        m[3].metric("Make the playoff", "<1%")
    if nm in pred.explanations:
        section("Why the model has them here")
        st.markdown(reasons(pred.explanations[nm]), unsafe_allow_html=True)
    a, b = st.columns([3, 2])
    with a:
        section("Results")
        logs = game_log(S, SimpleNamespace(p=RatingParams()))[ti]
        gl = pd.DataFrame([{"": "W" if g["won"] else "L", "Opponent": ("at " if g["loc"] == "road" else "vs ") + g["opp"] +
                            (" (neutral)" if g["loc"] == "neutral" else "") + (" · title game" if g["title_game"] else ""),
                            "Score": g["score"], "Opp. record": g["opp_record"],
                            "Opp. ranked then": f"#{g['opp_rank_then']}" if g["opp_rank_then"] else ""} for g in logs])
        if len(gl):
            st.dataframe(gl.style.map(lambda v: f"color: {GREEN}; font-weight: 700" if v == "W" else f"color: {RED}; font-weight: 700" if v == "L" else "",
                                      subset=[""]), hide_index=True, use_container_width=True, height=38 * len(gl) + 40)
        fut = S.future
        rows = [g for g in range(len(fut)) if ti in (fut.gh[g], fut.ga[g])]
        if rows:
            section("Still to play")
            P = RatingParams()
            rr = []
            for g in rows:
                home = fut.gh[g] == ti
                o = fut.ga[g] if home else fut.gh[g]
                mu = (S.rating[ti] - S.rating[o]) + (0 if fut.neutral[g] else (P.hfa if home else -P.hfa))
                day = pd.Timestamp(f"{S.season}-08-01") + pd.Timedelta(days=float(fut.day[g]) - 0.34)
                rr.append({"Date": day.strftime("%b %-d"), "Opponent": ("vs " if home else "at ") + (S.names[o] if o < S.n else "FCS opponent"),
                           "Win chance": float(norm.cdf(mu / P.sigma)) * 100})
            st.dataframe(pd.DataFrame(rr), hide_index=True, use_container_width=True,
                         column_config={"Win chance": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})
    with b:
        section("Résumé at a glance", "Rank among all 138 FBS teams in brackets.")
        rk = lambda c: int(pred.rel.raw[c].rank(ascending=False, method="min").iloc[ti])   # noqa: E731
        facts = [("Power rating", f"{raw.rating:+.1f} pts vs average ({rk('rating')})"),
                 ("Strength of record", f"{raw.wae:+.2f} wins vs a top-10 team's expectation ({rk('wae')})"),
                 ("Schedule difficulty", f"{rk('sos')} toughest played"),
                 ("Vs teams ranked at kickoff", f"{int(raw.w_ranked_g)}–{int(raw.l_ranked_g)}"),
                 ("Vs teams .500 or better", f"{int(raw.w500)}–{int(raw.l500)}"),
                 ("Vs power-conference teams", f"{int(raw.w_power)}–{int(raw.l_power)}"),
                 ("Home / road / neutral", f"{int(raw.w_home)}–{int(raw.l_home)} / {int(raw.w_road)}–{int(raw.l_road)} / {int(raw.w_neutral)}–{int(raw.l_neutral)}"),
                 ("Average margin (capped at 24)", f"{raw.margin_cap:+.1f}"),
                 ("Losses to unranked-calibre teams", f"{int(raw.loss_unranked)}")]
        st.dataframe(pd.DataFrame(facts, columns=["Measure", "Value"]), hide_index=True, use_container_width=True, height=38 * len(facts) + 40)
        if len(srow):
            r = srow.iloc[0]
            section("What they need")
            st.markdown(f"- **Win out** ({pct(r.p_win_out)} likely): in the field {pct(r.field_if_win_out)} of the time\n"
                        f"- **One more loss**: in {pct(r.field_if_one_more_loss)} of the time\n"
                        f"- **Win their conference**: {pct(r.p_conf_champ)}")
            sw = sim.swing[sim.swing.team == nm].head(5) if len(sim.swing) else []
            if len(sw):
                section("Games that swing their odds")
                for x in sw.itertuples():
                    st.markdown(f"- **{x.game}** ({pd.Timestamp(x.date).strftime('%b %-d')}): {pct(x.field_if_home_wins)} if the home team wins, "
                                f"{pct(x.field_if_away_wins)} if not")

# ----------------------------------------------------------------------------- what-if
with tabs[4]:
    section("What if...?", "Pick winners for any upcoming games, then re-run the season. Everything you leave alone is simulated as normal.")
    from cfp.simulate.season import Forced, simulate
    fut, n = S.future, S.n
    opts = []
    for g in np.argsort(fut.day):
        if fut.gh[g] < n and fut.ga[g] < n and not fut.ccg[g]:
            day = pd.Timestamp(f"{S.season}-08-01") + pd.Timedelta(days=float(fut.day[g]) - 0.34)
            opts.append((f"{day.strftime('%b %-d')} · {S.names[fut.ga[g]]} at {S.names[fut.gh[g]]}", S.names[fut.gh[g]], S.names[fut.ga[g]]))
    interesting = set(sim.teams[sim.teams.p_field >= 0.02].team)
    only = st.toggle("Only show games involving playoff contenders", True)
    shown = [o for o in opts if (not only) or o[1] in interesting or o[2] in interesting]
    chosen = st.multiselect("Games to decide", [o[0] for o in shown], placeholder="Choose one or more games")
    forced = []
    for lab in chosen:
        o = next(x for x in shown if x[0] == lab)
        c1, c2 = st.columns([3, 1])
        w = c1.radio(lab, [o[1], o[2]], horizontal=True, key="w" + lab)
        mg = c2.number_input("Winning margin (0 = any)", 0, 70, 0, key="m" + lab)
        forced.append(Forced(winner=w, loser=o[2] if w == o[1] else o[1], margin=int(mg) or None))
    contenders = sorted(sim.teams[sim.teams.p_title_game >= 0.03].team)
    c1, c2 = st.columns(2)
    tl = c1.multiselect("Loses its conference title game", contenders, placeholder="Optional")
    tw = c2.multiselect("Wins its conference title game", [c for c in contenders if c not in tl], placeholder="Optional")
    forced += [Forced(winner=t, title_game=True, wins=False) for t in tl] + [Forced(winner=t, title_game=True, wins=True) for t in tw]
    ns = st.select_slider("Simulated seasons (more is steadier but slower)", [500, 1000, 2000], 1000)
    if st.button("Run it", type="primary", disabled=not forced):
        with st.spinner("Playing out the season..."):
            w = simulate(n_sims=ns, seed=sim.seed, forced=forced)
        b = sim.teams.set_index("team")
        t = w.teams[(w.teams.p_field >= 0.005) | w.teams.team.map(lambda x: b.p_field.get(x, 0) >= 0.02)].copy()
        t["base"] = t.team.map(b.p_field).fillna(0)
        t["chg"] = (t.p_field - t.base) * 100
        section("Result", "Change is against the standard forecast. Small changes (a point or two) are just simulation noise.")
        st.dataframe(pd.DataFrame({" ": [logo(x) for x in t.team], "Team": t.team, "Make field": t.p_field * 100,
                                   "Change": t.chg.round(0), "First-round bye": t.p_bye * 100,
                                   "Typical final rank": t.mean_rank.round(1),
                                   "Most likely seed": [f"{int(w.seeds.loc[x].idxmax())}" if w.seeds.loc[x].max() > 0 else "" for x in t.team]}),
                     hide_index=True, use_container_width=True, height=min(38 * len(t) + 40, 760),
                     column_config={" ": st.column_config.ImageColumn(width="small"),
                                    "Make field": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                                    "First-round bye": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                                    "Change": st.column_config.NumberColumn(format="%+.0f pts")})
        st.caption("A title-game pick only applies in the seasons where that team reaches its title game.")
    elif not forced:
        st.info("Choose at least one game or title-game result above to run a scenario.")


# ----------------------------------------------------------------------------- report card
with tabs[5]:
    scored = _scored(S.season, STAMP)
    if not scored:
        st.info("No prediction has been scored yet. The first one is checked after the committee's next ranking.")
    else:
        pick = scored[-1]
        if len(scored) > 1:
            lab_ = {f"Ranking #{e['release']} ({pd.Timestamp(e['date']).strftime('%b %-d')})": e for e in scored}
            pick = lab_[st.selectbox("Which ranking", list(lab_)[::-1])]
        d, m = pick["table"].copy(), pick["model"]
        nT = len(d)
        rdate = pd.Timestamp(pick["date"]).strftime("%b %-d")
        section(f"The committee's ranking of {rdate} against our prediction",
                f"The prediction was stored before the ranking came out (using games through {pd.Timestamp(pick['made'], tz='UTC').tz_convert('America/New_York').strftime('%b %-d, %-I:%M %p ET')}).")
        long_run = None
        bk = REPORTS / "backtest_metrics.csv"
        if bk.exists():
            mdf_ = pd.read_csv(bk)
            long_run = mdf_[mdf_.model == "Ensemble (+polls, last week)"]
        c = st.columns(6)
        kpi(c[0], "Average miss", f"{m['mae']:.2f} places", f"2014–2025 test average: {long_run.mae.mean():.2f}" if long_run is not None else "per team")
        kpi(c[1], "Exactly right", f"{m['exact']} of {nT}", "teams in the exact spot")
        kpi(c[2], "Within one spot", f"{m['within1']:.0%}", f"test average: {long_run.within1.mean():.0%}" if long_run is not None else "")
        kpi(c[3], "Top 4", f"{m['top4']} of 4", "right teams, any order")
        kpi(c[4], "Top 12", f"{m['top12']} of 12", "right teams, any order")
        kpi(c[5], "Top 25", f"{m['top25']} of {nT}", "teams the model also had ranked")
        st.write("")
        worse = long_run is not None and m["mae"] > long_run.mae.mean() + 0.15
        better = long_run is not None and m["mae"] < long_run.mae.mean() - 0.15
        verdict = ("A weaker week than usual" if worse else "A better week than usual" if better else "About a normal week")
        st.markdown(f"**{verdict}.** The model was off by {m['mae']:.2f} places per team, with {m['exact']} teams in the exact spot and "
                    f"{int(round(m['within2'] * nT))} of {nT} within two. The committee's rank fell inside the model's stated 80% range for "
                    f"{pick['in_range']:.0%} of teams."
                    + (f" Simply repeating the committee's previous ranking would have missed by {pick['prev']['mae']:.2f}." if pick["prev"] else ""))

        a, b = st.columns([5, 6])
        with a:
            section("Predicted against actual", "Each row is a team, in the committee's order. The line shows how far the model was from the committee.")
            d["model_plot"] = d.model.clip(upper=28)
            d["Model had"] = ["Not ranked" if v > 30 else f"No. {v}" for v in d.model]
            d["dir"] = np.where(d.err_model == 0, "Exact", np.where(d.err_model > 0, "We had them too low", "We had them too high"))
            order = list(d.team)
            base = alt.Chart(d).encode(y=alt.Y("team:N", sort=order, title=None, axis=alt.Axis(labelLimit=150)))
            rule = base.mark_rule(strokeWidth=3, opacity=.55).encode(
                x=alt.X("actual:Q", title="Rank", scale=alt.Scale(domain=[0.5, 28.5]), axis=alt.Axis(values=[1, 4, 8, 12, 16, 20, 25])), x2="model_plot:Q",
                color=alt.Color("dir:N", scale=alt.Scale(domain=["We had them too low", "We had them too high", "Exact"], range=[GREEN, RED, GREY]),
                                legend=alt.Legend(orient="top", title=None)))
            tip = ["team:N", alt.Tooltip("actual:Q", title="Committee"), "Model had:N", alt.Tooltip("week:N", title="That week")]
            pa = base.mark_point(filled=True, size=90, color=NAVY, opacity=1).encode(x="actual:Q", tooltip=tip)
            pm = base.mark_point(filled=False, size=90, color=GOLD, strokeWidth=2.5, opacity=1).encode(x="model_plot:Q", tooltip=tip)
            st.altair_chart((rule + pm + pa).properties(height=26 * nT + 40), use_container_width=True)
            st.caption("Filled dot: the committee. Open gold ring: our prediction. A ring at the far right means the model did not have the team in its top 30.")
        with b:
            section("Team by team")
            miss = lambda e: "Exact" if e == 0 else f"{abs(e)} too {'low' if e > 0 else 'high'}"   # noqa: E731
            st.dataframe(pd.DataFrame({" ": [logo(t_) for t_ in d.team], "Committee": d.actual, "Team": d.team, "Record": d.record,
                                       "We predicted": ["NR" if v > 30 else str(v) for v in d.model], "Miss": [miss(e) for e in d.err_model],
                                       "Their week": d.week}).style.map(
                             lambda v: f"color: {GREEN}; font-weight: 700" if v == "Exact" else f"color: {RED}; font-weight: 700" if isinstance(v, str) and v[:1].isdigit() and int(v.split()[0]) >= 3 else "",
                             subset=["Miss"]),
                         hide_index=True, use_container_width=True, height=38 * nT + 40,
                         column_config={" ": st.column_config.ImageColumn(width="small"), "Committee": st.column_config.NumberColumn(width="small"),
                                        "We predicted": st.column_config.TextColumn(width="small")})

        a, b = st.columns(2)
        how = lambda x: ("left them where they were" if x.prev == x.actual else   # noqa: E731
                         f"moved them {'up' if x.prev > x.actual else 'down'} {abs(x.prev - x.actual)} from No. {x.prev}" if x.prev <= 30 else "brought them in from outside the top 25")
        with a:
            section("Biggest misses", "Three or more places off.")
            big = d[d.err_model.abs() >= 3].sort_values("err_model", key=abs, ascending=False)
            if big.empty:
                st.markdown("No team was three or more places off.")
            for x in big.itertuples():
                st.markdown(f'<span class="chip down"><b>{x.team}</b> · committee No. {x.actual}, we said {"not ranked" if x.model > 30 else "No. " + str(x.model)}'
                            f'<br>That week they {x.week}. The committee {how(x)}.</span>',
                            unsafe_allow_html=True)
        with b:
            section("What it got right", "Exact, or one place off.")
            good = d[d.err_model.abs() <= 1]
            ex = good[good.err_model == 0]
            if len(ex):
                st.markdown('<span class="chip up"><b>Exact spot</b><br>' + ", ".join(f"{x.team} ({x.actual})" for x in ex.itertuples()) + "</span>", unsafe_allow_html=True)
            one = good[good.err_model != 0]
            if len(one):
                st.markdown('<span class="chip up"><b>One place off</b><br>' + ", ".join(f"{x.team} ({x.actual}, we said {x.model})" for x in one.itertuples()) + "</span>",
                            unsafe_allow_html=True)
            movers = d[(d.prev <= 30) & ((d.prev - d.actual).abs() >= 4) & (d.err_model.abs() <= 2)]
            if len(movers):
                st.markdown('<span class="chip up"><b>Big moves it saw coming</b> (committee moved them four or more, model within two)<br>'
                            + ", ".join(f"{x.team} ({x.prev} → {x.actual}, we said {x.model})" for x in movers.itertuples()) + "</span>", unsafe_allow_html=True)

        section("Where the model leaned", "Average direction of the miss for different kinds of team. One ranking is a small sample, so read these as hints, not findings.")
        d["res"] = np.where(d.week.str.contains("lost to"), "Lost that week", np.where(d.week == "did not play", "Did not play", "Won that week"))
        d["conf"] = [CONF.get(S.conf[IDX[t_]], S.conf[IDX[t_]]) if t_ in IDX else "" for t_ in d.team]
        d["losses"] = d.record.str.split("-").str[1].astype(int).map(lambda v: f"{v} loss{'es' if v != 1 else ''}")
        lean = []
        for title_, col in (("By result that week", "res"), ("By conference", "conf"), ("By record", "losses")):
            for key, grp in d.groupby(col):
                if len(grp) >= 3:
                    e = float(grp.err_model.clip(-8, 8).mean())
                    lean.append({"Group": title_, "Teams like": key, "How many": len(grp),
                                 "Model's lean": "about right" if abs(e) < 0.5 else f"{abs(e):.1f} places too {'low' if e > 0 else 'high'}",
                                 "Average miss": float(grp.err_model.abs().clip(upper=8).mean())})
        st.dataframe(pd.DataFrame(lean), hide_index=True, use_container_width=True, height=38 * len(lean) + 40,
                     column_config={"Average miss": st.column_config.NumberColumn(format="%.1f places")})

        section("What happens with this")
        st.markdown(f"- The committee model is retrained on every committee ranking to date before each new prediction, so this ranking is already part of what it learns from "
                    f"({len(data.releases())} rankings since 2014).\n"
                    f"- The committee's new ranking is also the starting point for the next prediction and for the season simulation.\n"
                    f"- The next prediction, for {label}, is on the Rankings tab and will be scored here once that ranking is out.")
        if len(scored) > 1:
            section(f"{S.season} so far")
            st.dataframe(pd.DataFrame({"Ranking": [e["release"] for e in scored], "Date": [e["date"] for e in scored],
                                       "Average miss": [round(e["model"]["mae"], 2) for e in scored],
                                       "Exact": [e["model"]["exact"] for e in scored],
                                       "Within one spot": [f"{e['model']['within1']:.0%}" for e in scored],
                                       "Top 4": [f"{e['model']['top4']} of 4" for e in scored], "Top 12": [f"{e['model']['top12']} of 12" for e in scored]}),
                         hide_index=True, use_container_width=True)
        st.caption("A team the model did not have in its top 30 is counted as 31st when measuring the miss.")

# ----------------------------------------------------------------------------- track record
with tabs[6]:
    section("How accurate is it?", "Tested on 2014–2025: for each season the model was trained on every other season, then asked to predict that season's 72 weekly rankings in total.")
    bk = REPORTS / "backtest_metrics.csv"
    if bk.exists():
        mdf = pd.read_csv(bk)
        P, A = "Ensemble (+polls, last week)", "AP poll"
        mp, ma = mdf[mdf.model == P], mdf[mdf.model == A]
        c = st.columns(4)
        c[0].metric("Average miss per team", f"{mp.mae.mean():.1f} places", f"{mp.mae.mean() - ma.mae.mean():+.1f} vs AP poll", delta_color="inverse")
        c[1].metric("Teams within one spot", f"{mp.within1.mean():.0%}", f"{(mp.within1.mean() - ma.within1.mean()) * 100:+.0f} pts vs AP poll")
        c[2].metric("Top-12 teams identified", f"{mp.top12_overlap.mean():.0%}")
        fp, fa = mp[mp.is_final], ma[ma.is_final]
        c[3].metric("Exact playoff field", f"{int(fp.field_correct.sum())} of {len(fp)} seasons", f"AP poll: {int(fa.field_correct.sum())} of {len(fa)}", delta_color="off")
        section("Average miss by season", "Lower is better.")
        names_map = {P: "This model", A: "AP poll", "Prev rank + results": "Last week's ranking, adjusted for losses"}
        d = mdf[mdf.model.isin(names_map)].groupby(["season", "model"], as_index=False)["mae"].mean()
        d["Model"] = d.model.map(names_map)
        st.altair_chart(alt.Chart(d).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X("season:O", title=None), y=alt.Y("mae:Q", title="Places off, per team"),
            color=alt.Color("Model:N", scale=alt.Scale(domain=list(names_map.values()), range=[NAVY, GOLD, GREY]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["season:O", "Model:N", alt.Tooltip("mae:Q", format=".2f")]).properties(height=300), use_container_width=True)
    section(f"{S.season}: model against the committee")
    f = REPORTS / f"{S.season}_prediction_scores.csv"
    if f.exists():
        s = pd.read_csv(f)
        st.dataframe(pd.DataFrame({"Ranking": s.release, "Date": s.date, "Average miss": s.mae.round(2), "Within one spot": (s.within1 * 100).round(0),
                                   "Top 4 right": s.top4.astype(str) + " of 4", "Top 12 right": s.top12.astype(str) + " of 12"}),
                     hide_index=True, use_container_width=True)
    else:
        st.info("The first live prediction will be scored after the committee's next release.")
    with st.expander("How it works"):
        st.markdown("""
- **Team strength.** A rating built from every game's score, adjusted for opponent and venue.
- **Résumé.** About 80 measures per team: record, quality of wins and losses, schedule difficulty, head-to-head, recent form, conference.
- **Committee model.** Trained on every committee ranking since 2014 to learn how those measures, the polls and last week's ranking turn into the committee's order.
- **Simulation.** Plays out the rest of the season thousands of times, settles each conference race with its tiebreakers, then applies the 2026 selection and seeding rules.

The committee is thirteen people, so no model can be exact. Every number here comes with a range for that reason.""")
    with st.expander("What the model cannot know"):
        st.markdown("""
- How the committee will weigh things in the room on the day. Ranges are based on how far past committees strayed from the model.
- The committee's own schedule-strength and record-strength figures, which are private. Ours are reconstructions.
- Injuries, unless entered by hand.
- Which bowl each of the top three seeds will choose.""")
    st.caption("Data: ESPN public feeds. Team logos are loaded from ESPN and belong to their respective schools. Not affiliated with the College Football Playoff.")


# ----------------------------------------------------------------------------- fan page
if FAN:
    with tabs[7]:
        F = out["fan"][FAN]
        ti = IDX[FAN]
        raw = pred.rel.raw.iloc[ti]
        now, wo, G = F["now"], F["win_out"], F["games"]
        prow = tbl[tbl.team == FAN]
        prk = int(prow["rank"].iloc[0]) if len(prow) else None
        k = len(G)
        nword = {1: "its last game", 2: "both remaining games", 3: "all three remaining games", 4: "all four remaining games"}.get(k, "out")
        rel_day = pd.Timestamp(pred.release_date).normalize()
        wk_end = rel_day + pd.Timedelta(days=6)
        when = lambda d: pd.Timestamp(d).strftime("%a %b %-d")   # noqa: E731
        vs = lambda g: ("vs " if g["home"] else "at ") + g["opp"]   # noqa: E731
        st.markdown(f"""<div class="fan"><img src="{logo(FAN)}"><div><h2>{FAN}: the road to the playoff</h2>
<p>{int(raw.wins)}–{int(raw.losses)} overall · {int(raw.conf_wins)}–{int(raw.conf_losses)} {CONF.get(S.conf[ti], S.conf[ti])} ·
committee had them {"No. " + str(last_rank[FAN]) if FAN in last_rank else "unranked"} last week ·
predicted {"No. " + str(prk) if prk else "unranked"} on {label}</p></div></div>""", unsafe_allow_html=True)
        st.write("")
        c = st.columns(5)
        kpi(c[0], "Make the playoff", pct(now["p_field"]), "as things stand today")
        kpi(c[1], "Home playoff game", pct(now["p_host"]), "finish as a 5–8 seed")
        kpi(c[2], f"Win {nword}", pct(F["p_win_out"]), "chance it happens")
        kpi(c[3], "Playoff if they do", pct(wo.get("p_field")), "the whole season comes down to this")
        kpi(c[4], "Home game if they do", pct(wo.get("p_host")), f"bye {pct(wo.get('p_bye'))} · road game {pct(wo.get('p_road'))}")

        # ---- the plan
        root_all = [r for r in (F["rooting_win_out"] or F["rooting"]) if abs(r["z"]) >= 3]
        section("What we want to happen", "In order of importance. Every number comes from the 10,000 simulated seasons.")
        steps = []
        if G:
            g0 = G[0]
            steps.append(f"<b>Beat {g0['opp']}</b> ({when(g0['date'])}, {'home' if g0['home'] else 'away'}). {FAN} wins {pct(g0['p_win'])} of the time. "
                         f"A win keeps the playoff chance at {pct(g0['if_win'].get('p_field'))}; a loss drops it to {pct(g0['if_lose'].get('p_field'))}.")
            if k > 1:
                rest = " and ".join(f"{'beat' if g['home'] else 'win at'} {g['opp']} ({pct(g['p_win'])})" for g in G[1:])
                one = max((s.get("p_field", 0) for s in F["scenarios"] if s["n_losses"] == 1), default=0)
                steps.append(f"<b>Then {rest}.</b> " + ("There is no room for a slip: with" if one < 0.1 else "With")
                             + f" one more loss {FAN} makes the field {pct(one)} of the time at best.")
            steps.append(f"<b>Win {nword} and the odds flip.</b> {FAN} would finish around No. {wo['median_rank']:.0f} "
                         f"(usually between {wo['rank_p10']:.0f} and {wo['rank_p90']:.0f}) and make the playoff {pct(wo['p_field'])} of the time, "
                         f"with a {pct(wo['p_host'])} chance of a home game.")
        if root_all:
            names_ = []
            for r in root_all:
                loser = r["away"] if r["root_for"] == r["home"] else r["home"]
                if loser not in names_:
                    names_.append(loser)
            steps.append(f"<b>Get help.</b> Even winning out leaves about a {pct(1 - wo['p_field'])} chance of missing. The teams whose losses help most: "
                         f"{', '.join(names_[:6])}. Details in the rooting guide below.")
        st.markdown("".join(f'<div class="step"><b class="n">{i}</b>{t}</div>' for i, t in enumerate(steps, 1)), unsafe_allow_html=True)

        a, b = st.columns([1, 1])
        with a:
            if G:
                g0 = G[0]
                section(f"This week: {g0['opp']} {'at ' + FAN if g0['home'] else 'hosts ' + FAN}", f"{when(g0['date'])} · the week of the committee's ranking on {label}.")
                oi = g0["opp_idx"]
                orow = pred.rel.raw.iloc[oi] if oi < S.n else None
                wi, lo_ = g0["if_win"], g0["if_lose"]
                tw = pd.DataFrame({"": ["Playoff chance", "Home playoff game", "Typical final ranking"],
                                   f"If {FAN} wins ({pct(g0['p_win'])})": [pct(wi.get("p_field")), pct(wi.get("p_host")), f"No. {wi['median_rank']:.0f}"],
                                   f"If {FAN} loses ({pct(1 - g0['p_win'])})": [pct(lo_.get("p_field")), pct(lo_.get("p_host")), f"No. {lo_['median_rank']:.0f}"]})
                st.dataframe(tw, hide_index=True, use_container_width=True)
                bits = []
                if orow is not None:
                    bits.append(f"{g0['opp']} comes in {int(orow.wins)}–{int(orow.losses)} with a power rating of {orow.rating:+.1f} points against an average team; "
                                f"{FAN}'s is {raw.rating:+.1f}.")
                if prk:
                    bits.append(f"On {label} the model expects the committee to put {FAN} at No. {prk} (likely range {int(prow.low.iloc[0])}–{int(prow.high.iloc[0])})"
                                + (f", up from No. {last_rank[FAN]}" if FAN in last_rank and last_rank[FAN] > prk else "") + ".")
                if wi.get("p_field", 0) - now["p_field"] < 0.1 and lo_.get("p_field", 1) < now["p_field"] / 2:
                    bits.append("Winning this game does not raise the odds much on its own. It is the game the rest of the plan depends on.")
                st.markdown(" ".join(bits))
            section("The games left", "Playoff chance after each result, averaged over everything else that could happen.")
            st.dataframe(pd.DataFrame({"Date": [when(g["date"]) for g in G], "Game": [vs(g) for g in G],
                                       "Win chance": [g["p_win"] * 100 for g in G],
                                       "Playoff if win": [g["if_win"].get("p_field", 0) * 100 for g in G],
                                       "Playoff if lose": [g["if_lose"].get("p_field", 0) * 100 for g in G]}),
                         hide_index=True, use_container_width=True,
                         column_config={c_: st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)
                                        for c_ in ("Win chance", "Playoff if win", "Playoff if lose")})
        with b:
            section("Every way the rest of the season can go", f"All {2 ** k} combinations of {FAN}'s results.")
            sc = [s for s in F["scenarios"] if s["n"] > 0]
            st.dataframe(pd.DataFrame({"Finish": [s["label"] for s in sc], "How likely": [s["share"] * 100 for s in sc],
                                       "Make playoff": [s["p_field"] * 100 for s in sc], "Home game": [s["p_host"] * 100 for s in sc],
                                       "Typical final rank": [f"No. {s['median_rank']:.0f}" for s in sc]}),
                         hide_index=True, use_container_width=True, height=38 * len(sc) + 40,
                         column_config={c_: st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)
                                        for c_ in ("How likely", "Make playoff", "Home game")})
            st.caption(f"Record before any Big Ten title game. {FAN} reaches that game in {pct(now['p_title_game'])} of seasons "
                       f"({pct(wo.get('p_title_game'))} if it wins out).")

        # ---- home game
        section("A playoff game at home",
                "Seeds 5 to 8 host the first round on campus on Dec 18–19. Seeding follows the committee's final ranking, so hosting means finishing about 5th to 8th. "
                "Seeds 9 to 11 play that round on the road; seed 12 goes to the lowest-ranked conference champion.")
        a, b = st.columns([3, 2])
        with a:
            sd = pd.DataFrame([{"Seed": s_, "Chance": v * 100, "Case": lab_,
                                "Kind": "Bye" if s_ <= 4 else "Home game" if s_ <= 8 else "Road game"}
                               for lab_, d in (("As things stand", now["seeds"]), (f"If {FAN} wins {nword}", wo.get("seeds", {})))
                               for s_, v in d.items()])
            ch = alt.Chart(sd).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("Seed:O", axis=alt.Axis(labelAngle=0, title="Playoff seed")),
                y=alt.Y("Chance:Q", title="Chance (%)"),
                color=alt.Color("Kind:N", scale=alt.Scale(domain=["Bye", "Home game", "Road game"], range=[NAVY, "#e0a800", GREY]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["Case", "Seed", alt.Tooltip("Chance:Q", format=".1f"), "Kind"],
            ).properties(height=210).facet(column=alt.Column("Case:N", title=None, sort=["As things stand"]))
            st.altair_chart(ch.resolve_scale(y="shared").configure_view(strokeWidth=0), use_container_width=True)
        with b:
            m = st.columns(2)
            m[0].metric("Home game today", pct(now["p_host"]))
            m[1].metric(f"If {FAN} wins out", pct(wo.get("p_host")))
            ho = F["host_opponents"]
            if ho:
                st.markdown("**Most likely visitors** (share of the seasons where " + FAN + " hosts): "
                            + ", ".join(f"{o['team']} {pct(o['share'])}" for o in ho[:5]) + "." + (" No opponent stands out: the 9–12 seeds are wide open." if ho[0]["share"] < 0.15 else ""))
            ro = F["road_opponents"]
            if ro:
                st.markdown("**If it is a road game instead**, the most likely trips: " + ", ".join(f"{o['team']} {pct(o['share'])}" for o in ro[:5]) + ".")
            short = int(wo["median_rank"]) - 8
            if short > 0:
                st.markdown(f"Winning out puts {FAN} at No. {wo['median_rank']:.0f} on a typical run, {short} spot{'s' if short > 1 else ''} short of "
                            f"hosting, so the home game also needs teams above to lose.")

        # ---- who is in the way
        if prk and field_names:
            lastin_rank = int(tbl[tbl.team == at_large[-1]]["rank"].iloc[0]) if at_large else 12
            lo_r, hi_r = min(lastin_rank, prk) - 3, prk + 3
            near = tbl[(tbl["rank"] >= lo_r) & (tbl["rank"] <= hi_r)]
            gl = sim.games.set_index("game")
            fut = S.future
            def nxt(i: int) -> tuple[str, float]:
                rows = [g for g in np.argsort(fut.day) if i in (fut.gh[g], fut.ga[g]) and not fut.ccg[g]]
                if not rows:
                    return "season complete", np.nan
                g = rows[0]
                h, a_ = int(fut.gh[g]), int(fut.ga[g])
                hn, an = (S.names[h] if h < S.n else "FCS opponent"), (S.names[a_] if a_ < S.n else "FCS opponent")
                ph = gl.p_home_win.get(f"{an} at {hn}", np.nan)
                day = pd.Timestamp(f"{S.season}-08-01") + pd.Timedelta(days=float(fut.day[g]) - 0.34)
                return f"{day.strftime('%b %-d')} {'vs ' + an if h == i else 'at ' + hn}", (ph if h == i else 1 - ph) * 100
            sb = sim.teams.set_index("team")
            section("Who is in the way", f"The teams around {FAN} in the predicted ranking. The last at-large place today belongs to {at_large[-1]} "
                                         f"(No. {lastin_rank}); {FAN} has to pass {max(prk - lastin_rank, 0)} teams or see them fall.")
            nx = [nxt(IDX[t_]) for t_ in near.team]
            st.dataframe(pd.DataFrame({" ": [logo(t_) for t_ in near.team], "Predicted": near["rank"].to_numpy(), "Team": near.team.to_numpy(),
                                       "Record": near.record.to_numpy(),
                                       "Status today": ["In the field" if t_ in field_names else "◀ " + FAN if t_ == FAN else "Out" for t_ in near.team],
                                       "Playoff chance": [float(sb.p_field.get(t_, 0)) * 100 for t_ in near.team],
                                       "Next game": [x[0] for x in nx], "Chance they win it": [x[1] for x in nx]}),
                         hide_index=True, use_container_width=True, height=38 * len(near) + 40,
                         column_config={" ": st.column_config.ImageColumn(width="small"),
                                        "Playoff chance": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                                        "Chance they win it": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})

        # ---- rooting guide
        def guide(rows: list[dict], what: str) -> None:
            if not rows:
                st.info("Nothing clears the noise threshold.")
                return
            st.dataframe(pd.DataFrame({"Date": [when(r["date"]) for r in rows], "Game": [r["game"] for r in rows],
                                       "Root for": [r["root_for"] for r in rows], "Chance it happens": [r["p_root"] * 100 for r in rows],
                                       f"{what} if it does": [max(r["if_home"], r["if_away"]) * 100 for r in rows],
                                       "If it does not": [min(r["if_home"], r["if_away"]) * 100 for r in rows],
                                       "Worth": [abs(r["swing"]) * 100 for r in rows]}),
                         hide_index=True, use_container_width=True, height=38 * len(rows) + 40,
                         column_config={"Chance it happens": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                                        f"{what} if it does": st.column_config.NumberColumn(format="%.0f%%"),
                                        "If it does not": st.column_config.NumberColumn(format="%.0f%%"),
                                        "Worth": st.column_config.NumberColumn(format="+%.0f pts")})

        cond = bool(F["rooting_win_out"])
        base_txt = f"Assumes {FAN} wins {nword}, since little else matters otherwise. " if cond else ""
        this_wk = sorted([r for r in root_all if r["date"] <= wk_end], key=lambda r: (r["date"], -abs(r["swing"])))
        later = sorted([r for r in root_all if r["date"] > wk_end], key=lambda r: (r["date"], -abs(r["swing"])))
        a, b = st.columns(2)
        with a:
            section("Rooting guide: this week", base_txt + f"How much each result moves {FAN}'s playoff chance.")
            guide(this_wk, "Playoff")
        with b:
            section("Rooting guide: the weeks after", "Same reading. These will firm up as results come in.")
            guide(later, "Playoff")
        hr = sorted([r for r in F["rooting_host_win_out"] if abs(r["z"]) >= 3.5], key=lambda r: (r["date"], -abs(r["swing"])))
        if hr:
            section("Results that help bring a home game", base_txt + f"Same idea, measured on {FAN}'s chance of hosting.")
            guide(hr, "Home game")
        st.caption(f"How to read this page: each figure is a share of the {F['n_sims']:,} simulated seasons that match the condition. "
                   "The rooting tables only list games whose effect is at least three times its sampling error, so small or doubtful effects are left out; "
                   "with about 200 games tested, one listed game could still be chance. The simulation's playoff odds were well calibrated on 2014–2025 "
                   "but there are only two seasons of the 12-team format to learn from.")
