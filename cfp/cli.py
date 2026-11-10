"""Command line: cfp update | predict | simulate | whatif | backtest | team."""
from __future__ import annotations

import warnings

import typer
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")
app = typer.Typer(add_completion=False, help="College Football Playoff committee ranking predictor")
con = Console()


def _pct(x) -> str:
    import numpy as np
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return "<1%" if 0 < x < 0.005 else f"{x:.0%}"


def _show_pred(pred) -> None:
    t = Table(title=f"Predicted committee ranking #{pred.release_no} ({pred.release_date})")
    for c in ("Pred", "Team", "Record", "Last", "AP", "This spot", "80% range", "Top 4", "Top 12"):
        t.add_column(c)
    for r in pred.table.head(25).itertuples():
        t.add_row(str(r.rank), r.team, r.record, "NR" if r.prev != r.prev or r.prev is None else str(int(r.prev)),
                  "NR" if r.ap != r.ap or r.ap is None else str(int(r.ap)), _pct(r.p_exact), f"{r.low}-{r.high}",
                  _pct(r.p_top4), _pct(r.p_top12))
    con.print(t)
    for n in pred.notes:
        con.print(f"[dim]{n}[/dim]")


def _show_sim(sim, top: int = 30) -> None:
    t = Table(title=f"Simulated Selection Day ({sim.n_sims:,} seasons, seed {sim.seed})")
    for c in ("Team", "Field", "Bye", "Host", "Auto", "League", "Mean rank", "Title", "Win out -> in", "1 more loss -> in"):
        t.add_column(c)
    for r in sim.teams.head(top).itertuples():
        t.add_row(r.team, _pct(r.p_field), _pct(r.p_bye), _pct(r.p_host), _pct(r.p_auto), _pct(r.p_conf_champ),
                  f"{r.mean_rank:.1f}", f"{r.p_champion:.1%}", _pct(r.field_if_win_out), _pct(r.field_if_one_more_loss))
    con.print(t)
    for n in sim.notes:
        con.print(f"[dim]{n}[/dim]")


@app.command()
def update(sims: int = typer.Option(10000, help="simulated seasons"), seed: int = 2026,
           no_fetch: bool = typer.Option(False, help="skip the network fetch and reuse the database")):
    """Fetch new results and rankings, recompute everything, write the weekly report."""
    from cfp.ingest.http import FetchError
    from cfp.pipeline import run_update
    try:
        out = run_update(n_sims=sims, seed=seed, fetch=not no_fetch)
    except FetchError as e:
        con.print(f"[red]UPDATE STOPPED, a fetch failed:[/red] {e}\nNothing was filled in from memory. Fix and re-run.")
        raise typer.Exit(1)
    for x in out["log"]:
        con.print(f"- {x}")
    _show_pred(out["pred"]); _show_sim(out["sim"], 20)
    con.print(f"Weekly report: {out['report']}")


@app.command()
def predict(explain: bool = typer.Option(True, help="print the explanation for each team")):
    """Predict the committee's next ranking with uncertainty and explanations."""
    from cfp.pipeline import save_prediction
    from cfp.predict import bracket_text, predict_next
    pred = predict_next()
    save_prediction(pred)
    _show_pred(pred)
    con.print("\n" + bracket_text(pred.field, pred.rel.st, "Bracket if the season ended today:"))
    if explain:
        con.print()
        for nm, e in pred.explanations.items():
            up = "; ".join(f"{g} ({v:+.1f}{', ' + f if f else ''})" for g, v, f in e["up"]) or "-"
            dn = "; ".join(f"{g} ({v:+.1f}{', ' + f if f else ''})" for g, v, f in e["down"]) or "-"
            con.print(f"[bold]{e['rank']}. {nm}[/bold]\n   helping: {up}\n   hurting: {dn}", highlight=False)
        con.print()
        for p in pred.pairs:
            con.print(p, highlight=False)


@app.command()
def simulate(sims: int = 10000, seed: int = 2026):
    """Simulate the rest of the season and the playoff."""
    from cfp.simulate.season import simulate as run
    sim = run(n_sims=sims, seed=seed)
    _show_sim(sim)
    con.print(sim.title_games.to_string(index=False))
    con.print(sim.games.head(15).to_string(index=False))


@app.command()
def whatif(result: list[str] = typer.Option(..., "--result", "-r", help='e.g. "Ohio State over Michigan 27-24" or "Georgia loses SEC title game"'),
           sims: int = 5000, seed: int = 2026):
    """Force any set of results and see the simulated ranking, field and odds."""
    from cfp.simulate.season import parse_result, simulate as run
    forced = [parse_result(r) for r in result]
    base = run(n_sims=sims, seed=seed)
    sim = run(n_sims=sims, seed=seed, forced=forced)
    con.print("Forced: " + "; ".join(result))
    t = Table(title="What-if versus baseline")
    for c in ("Team", "Field", "was", "Bye", "was", "Mean rank", "was", "Most likely seed"):
        t.add_column(c)
    b = base.teams.set_index("team")
    for r in sim.teams.head(24).itertuples():
        sd = sim.seeds.loc[r.team]
        t.add_row(r.team, _pct(r.p_field), _pct(b.p_field.get(r.team)), _pct(r.p_bye), _pct(b.p_bye.get(r.team)),
                  f"{r.mean_rank:.1f}", f"{b.mean_rank.get(r.team, float('nan')):.1f}",
                  f"{int(sd.idxmax())} ({_pct(sd.max())})" if sd.max() > 0 else "")
    con.print(t)
    con.print("Note: a title-game condition applies in the simulations where that team reaches its title game.")


@app.command()
def backtest():
    """Leave-one-season-out backtest; writes reports/backtest_report.md."""
    from cfp.committee.report import run
    run()
    con.print("Wrote reports/backtest_report.md")


@app.command()
def team(name: str, sims: int = typer.Option(0, help="also simulate (0 = reuse the last update's simulation)")):
    """Full résumé sheet for one team, like the committee would see."""
    from types import SimpleNamespace

    from cfp.pipeline import load_outputs
    from cfp.predict import team_sheet
    sim = None
    if sims:
        from cfp.simulate.season import simulate as run
        sim = run(n_sims=sims)
    else:
        o = load_outputs()
        if o:
            sim = SimpleNamespace(**o["sim"])
    con.print(team_sheet(name, sim=sim), highlight=False, markup=False)


if __name__ == "__main__":
    app()
