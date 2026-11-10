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
  .block-container {{ padding-top: 1.2rem; max-width: 1280px; }}
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
    """Loaded once per data refresh and shared by every visitor (keeps CPU use low on a shared host)."""
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


_f = OUT / "latest.pkl"
out = _outputs(_f.stat().st_mtime) if _f.exists() else None
if out is None:
    st.error("No results yet. Run `cfp update` first.")
    st.stop()
pred, sim = out["pred"], SimpleNamespace(**out["sim"])
S = pred.rel.st
IDX = {n: i for i, n in enumerate(S.names)}
last_df, last_date = _last_ranking(S.season, _f.stat().st_mtime)
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
kpi(k[4], "Simulated seasons", f"{sim.n_sims:,}", "every remaining game played out")
st.write("")

tabs = st.tabs(["Rankings", "Bracket", "Playoff odds", "Teams", "What-if", "Track record"])

# ----------------------------------------------------------------------------- rankings
with tabs[0]:
    c1, c2 = st.columns([5, 2])
    with c1:
        section("Predicted committee Top 25", "The range is where the team lands in 8 of 10 cases.")
        t = tbl.head(25)
        show = pd.DataFrame({
            "#": t["rank"], " ": [logo(x) for x in t.team], "Team": t.team, "Record": t.record,
            "Conf": [CONF.get(c, c) for c in t.conf], "Move": [move(a, b) for a, b in zip(t.team, t["rank"])],
            "AP": t["ap"].map(lambda x: "NR" if pd.isna(x) else str(int(x))),
            "This exact spot": t.p_exact * 100, "Likely range": t.low.astype(str) + "–" + t.high.astype(str),
            "Top 4": t.p_top4 * 100, "Top 12": t.p_top12 * 100})
        prog = lambda: st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)   # noqa: E731
        st.dataframe(show, hide_index=True, use_container_width=True, height=915,
                     column_config={" ": st.column_config.ImageColumn(width="small"), "#": st.column_config.NumberColumn(width="small"),
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
                         column_config={" ": st.column_config.ImageColumn(width="small")})
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

# ----------------------------------------------------------------------------- track record
with tabs[5]:
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
