"""Local dashboard:  streamlit run cfp/app/dashboard.py   (or: cfp-dashboard)"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
from cfp import data  # noqa: E402
from cfp.config import REPORTS  # noqa: E402
from cfp.pipeline import load_outputs  # noqa: E402
from cfp.predict import bowl_paths, bracket_text, team_sheet  # noqa: E402

st.set_page_config(page_title="CFP committee predictor", layout="wide")


@st.cache_resource(show_spinner=False)
def _outputs(stamp: float):
    """Loaded once per data refresh and shared by every visitor (keeps CPU use low on a shared host)."""
    return load_outputs()


from cfp.pipeline import OUT  # noqa: E402

_f = OUT / "latest.pkl"
out = _outputs(_f.stat().st_mtime) if _f.exists() else None
if out is None:
    st.error("No outputs yet. Run `cfp update` first.")
    st.stop()
pred, sim = out["pred"], SimpleNamespace(**out["sim"])
S = pred.rel.st
names = data.team_names()


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else ("<1%" if 0 < x < 0.005 else f"{x:.0%}")


st.title("CFP committee predictor")
st.caption(f"Ranking #{pred.release_no}, {'Selection Day' if pred.is_final else pred.release_date}. Computed {out['made']} UTC. "
           + " ".join(pred.notes))
tabs = st.tabs(["Predicted Top 25", "Bracket", "Playoff odds", "Team résumé", "What-if", "Accuracy"])

with tabs[0]:
    rel = data.releases(); rel = rel[rel.season == S.season].sort_values("release_no")
    last = data.cfp_ranking(S.season, rel.cutoff.iloc[-1]) if len(rel) else pd.DataFrame()
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Model: next ranking")
        t = pred.table.head(25).copy()
        show = pd.DataFrame({"Pred": t["rank"], "Team": t["team"], "Record": t["record"],
                             "Last": t["prev"].map(lambda x: "NR" if pd.isna(x) else int(x)),
                             "AP": t["ap"].map(lambda x: "NR" if pd.isna(x) else int(x)),
                             "This spot": t["p_exact"].map(pct), "80% range": t["low"].astype(str) + "–" + t["high"].astype(str),
                             "Top 4": t["p_top4"].map(pct), "Top 12": t["p_top12"].map(pct)})
        st.dataframe(show, hide_index=True, use_container_width=True, height=920)
    with c2:
        st.subheader(f"Committee: last actual ({str(rel.cutoff.iloc[-1])[:10]})" if len(rel) else "No committee ranking yet")
        if len(last):
            st.dataframe(pd.DataFrame({"Rank": last["rank"].astype(int), "Team": [names[int(x)] for x in last["team_id"]],
                                       "Record": last["record"]}), hide_index=True, use_container_width=True, height=920)
    st.subheader("Rank distribution")
    pick = st.selectbox("Team", list(pred.dist.index), key="dist_team")
    dd = pred.dist.loc[pick]
    st.bar_chart(pd.DataFrame({"probability": dd.to_numpy()},
                              index=pd.Index([c.zfill(2) if c.isdigit() else c for c in dd.index], name="rank")))
    st.subheader("Why")
    for nm, e in pred.explanations.items():
        with st.expander(f"{e['rank']}. {nm}"):
            for g, v, f in e["up"]:
                st.markdown(f"- **Helping:** {g} ({v:+.1f}){': ' + f if f else ''}")
            for g, v, f in e["down"]:
                st.markdown(f"- **Hurting:** {g} ({v:+.1f}){': ' + f if f else ''}")
    with st.expander("Neighbour by neighbour"):
        for p in pred.pairs:
            st.markdown(f"- {p}")

with tabs[1]:
    st.subheader("If the season ended today")
    st.caption("Predicted ranking, with each league's current leader standing in as its champion. Bowl choices by the top three seeds are an assumption.")
    st.code(bracket_text(pred.field, S))
    st.subheader("Simulated seeds")
    top = sim.teams[sim.teams.p_field >= 0.03]["team"]
    sd = sim.seeds.loc[top].copy(); sd.columns = [f"Seed {c}" for c in sd.columns]
    st.dataframe(sd.style.format(lambda v: "" if v < 0.005 else f"{v:.0%}"), use_container_width=True)
    st.subheader("Most likely title games")
    st.dataframe(sim.title_games.assign(p=sim.title_games.p.map(pct)), hide_index=True, use_container_width=True)

with tabs[2]:
    st.subheader(f"Simulated Selection Day: {sim.n_sims:,} seasons, seed {sim.seed}")
    for n in sim.notes:
        st.caption(n)
    t = sim.teams[sim.teams.p_field >= 0.005]
    cols = {"team": "Team", "p_field": "Make field", "p_bye": "Bye", "p_host": "Host", "p_auto": "Auto bid",
            "p_conf_champ": "Win league", "mean_rank": "Mean rank", "p_semifinal": "Semifinal", "p_champion": "Title",
            "p_win_out": "Wins out", "field_if_win_out": "In if wins out", "field_if_one_more_loss": "In with one more loss"}
    d = t[list(cols)].rename(columns=cols)
    st.dataframe(d.style.format({c: "{:.0%}" for c in d.columns if c not in ("Team", "Mean rank", "Title")} | {"Mean rank": "{:.1f}", "Title": "{:.1%}"}, na_rep="n/a"),
                 hide_index=True, use_container_width=True, height=800)
    st.subheader("Games that move the picture most")
    st.dataframe(sim.games.head(25), hide_index=True, use_container_width=True)

with tabs[3]:
    nm = st.selectbox("Team", sorted(S.names), index=sorted(S.names).index(pred.table.team.iloc[0]))

    @st.cache_data(show_spinner=False)
    def _sheet(name: str, stamp: str) -> str:
        return team_sheet(name, pred=pred, sim=sim)

    st.markdown(_sheet(nm, out["made"]))

with tabs[4]:
    st.subheader("What-if builder")
    st.caption("Pick winners of upcoming games (optionally a margin), then simulate. Unpicked games are simulated normally.")
    from cfp.simulate.season import Forced, simulate
    fut = S.future
    n = S.n
    opts = []
    for g in np.argsort(fut.day):
        if fut.gh[g] < n and fut.ga[g] < n and not fut.ccg[g]:
            day = pd.Timestamp(f"{S.season}-08-01") + pd.Timedelta(days=float(fut.day[g]) - 0.34)
            opts.append((f"{day.date()}  {S.names[fut.ga[g]]} at {S.names[fut.gh[g]]}", S.names[fut.gh[g]], S.names[fut.ga[g]]))
    interesting = set(sim.teams[sim.teams.p_field >= 0.02].team)
    only = st.checkbox("Only games involving playoff contenders", True)
    shown = [o for o in opts if (not only) or o[1] in interesting or o[2] in interesting]
    chosen = st.multiselect("Games", [o[0] for o in shown])
    forced = []
    for lab in chosen:
        o = next(x for x in shown if x[0] == lab)
        c1, c2 = st.columns([3, 1])
        w = c1.radio(lab, [o[1], o[2]], horizontal=True, key="w" + lab)
        m = c2.number_input("Margin (0 = simulate)", 0, 70, 0, key="m" + lab)
        forced.append(Forced(winner=w, loser=o[2] if w == o[1] else o[1], margin=int(m) or None))
    contenders = sorted(sim.teams[sim.teams.p_title_game >= 0.03].team)
    tl = st.multiselect("Loses its conference title game (if it gets there)", contenders)
    tw = st.multiselect("Wins its conference title game (if it gets there)", [c for c in contenders if c not in tl])
    forced += [Forced(winner=t, title_game=True, wins=False) for t in tl] + [Forced(winner=t, title_game=True, wins=True) for t in tw]
    ns = st.select_slider("Simulations (more = steadier numbers, slower)", [500, 1000, 2000], 1000)
    st.caption("Capped at 2,000 here to stay within the free host's CPU limit. Use `cfp whatif` on your own machine for 10,000.")
    if st.button("Simulate", type="primary"):
        with st.spinner("Simulating..."):
            w = simulate(n_sims=ns, seed=sim.seed, forced=forced)
        b = sim.teams.set_index("team")
        t = w.teams[w.teams.p_field >= 0.005].copy()
        d = pd.DataFrame({"Team": t.team, "Make field": t.p_field.map(pct), "Baseline": t.team.map(b.p_field).map(pct),
                          "Bye": t.p_bye.map(pct), "Mean final rank": t.mean_rank.round(1),
                          "Baseline rank": t.team.map(b.mean_rank).round(1),
                          "Most likely seed": [f"{int(w.seeds.loc[x].idxmax())} ({pct(w.seeds.loc[x].max())})" if w.seeds.loc[x].max() > 0 else "" for x in t.team]})
        st.dataframe(d, hide_index=True, use_container_width=True, height=700)
        # bracket from the what-if: teams by mean simulated rank, most likely champions
        st.caption("Baseline uses the last full update's simulation.")

with tabs[5]:
    st.subheader("Model versus committee")
    f = REPORTS / f"{S.season}_prediction_scores.csv"
    if f.exists():
        st.dataframe(pd.read_csv(f), hide_index=True)
    else:
        st.info("No pre-release prediction has been scored yet this season.")
    lg = REPORTS / f"{S.season}_prediction_log.md"
    if lg.exists():
        st.markdown(lg.read_text())
    bk = REPORTS / "backtest_metrics.csv"
    if bk.exists():
        m = pd.read_csv(bk)
        st.subheader("Backtest 2014–2025: mean absolute rank error by season")
        show = ["Ensemble (+polls, last week)", "AP poll", "Prev rank + results", "Ensemble (carry-over from first ranking)"]
        st.line_chart(m[m.model.isin(show)].pivot_table(index="season", columns="model", values="mae"))
    cal = REPORTS / "simulation_calibration.md"
    if cal.exists():
        st.markdown(cal.read_text())
