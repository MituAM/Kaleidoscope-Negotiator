"""
CAI Showcase - Kaleidoscope Negotiation Agent
Collaborative Artificial Intelligence - ANL 2026

Run:  python showcase.py
      python showcase.py --quick        (skips the tournament)
      python showcase.py --scenario Laptop
"""

from __future__ import annotations

import sys
import io
import random
from pathlib import Path

# Force UTF-8 output on Windows so Rich can render safely
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Make sure local modules (examples/, mynegotiator, etc.) are on the path
sys.path.insert(0, str(Path(__file__).parent))

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

from negmas.inout import Scenario
from negmas.preferences import compare_ufuns
from negmas.sao import SAOMechanism

from mynegotiator import MyNegotiator

console = Console(force_terminal=True)
app = typer.Typer(add_completion=False)

SCENARIOS_DIR = Path(__file__).parent / "scenarios"

# Opponents used in the showcase tournament
OPPONENTS: list[tuple[str, str]] = [
    ("negmas.sao.BoulwareTBNegotiator", "Boulware"),
    ("examples.boa.BOANeg", "BOA"),
    ("examples.simple.SimpleNegotiator", "Simple"),
    ("examples.map.MAPNeg", "MAP"),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_class(dotted: str):
    """Import a class given its fully-qualified name."""
    parts = dotted.rsplit(".", 1)
    mod = __import__(parts[0], fromlist=[parts[1]])
    return getattr(mod, parts[1])


def calc_scores(m: SAOMechanism) -> dict[str, dict[str, float]]:
    """ANL 2026 scoring: Score = Advantage + Concealing."""
    agreement = m.agreement
    negotiators = [n.__class__.__name__ for n in m.negotiators]

    if agreement is None:
        return {
            name: {"Advantage": 0.0, "Concealing": 0.0, "Score": 0.0}
            for name in negotiators
        }

    advantages = [
        float(n.ufun(agreement)) - float(n.ufun.reserved_value)
        if n.ufun
        else 0.0
        for n in m.negotiators
    ]

    ufuns = [n.ufun for n in m.negotiators]
    models = [n.opponent_ufun for n in m.negotiators]
    models.reverse()  # model[i] is the opponent's model of ufuns[i]

    # Pass the true ufun's outcome_space so compare_ufuns can sample outcomes
    # (opponent models like LambdaMultiFun may not carry their own outcome_space)
    accuracies = [
        (1 + compare_ufuns(u, mod, method="kendall", outcome_space=u.outcome_space)) / 2
        for u, mod in zip(ufuns, models)
    ]
    acc_sum = sum(accuracies)
    accuracies = [a / acc_sum for a in accuracies] if acc_sum > 0 else [0.0] * 2
    accuracies.reverse()  # flip: higher opp accuracy = lower concealing for us

    return dict(
        zip(
            negotiators,
            (
                {"Advantage": adv, "Concealing": acc, "Score": adv + acc}
                for adv, acc in zip(advantages, accuracies)
            ),
        )
    )


def run_negotiation(
    scenario: Scenario, opponent_dotted: str, negotiator_first: bool = True
) -> tuple[SAOMechanism, dict[str, dict[str, float]]]:
    opp_cls = _load_class(opponent_dotted)
    opp_name = opponent_dotted.split(".")[-1]
    neg_name = "MyNegotiator"

    m = SAOMechanism(n_steps=100, outcome_space=scenario.outcome_space)
    if negotiator_first:
        m.add(MyNegotiator(ufun=scenario.ufuns[0], id=neg_name, name=neg_name))
        m.add(opp_cls(ufun=scenario.ufuns[1], id=opp_name, name=opp_name))
    else:
        m.add(opp_cls(ufun=scenario.ufuns[0], id=opp_name, name=opp_name))
        m.add(MyNegotiator(ufun=scenario.ufuns[1], id=neg_name, name=neg_name))

    m.run()
    return m, calc_scores(m)


# ---------------------------------------------------------------------------
# display sections
# ---------------------------------------------------------------------------


def print_welcome() -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]ANL 2026 - Negotiation Agent Showcase[/bold cyan]\n"
            "[dim]Collaborative Artificial Intelligence - Course Portfolio[/dim]\n\n"
            "[white]Agent  :[/white] [yellow bold]Kaleidoscope Negotiator[/yellow bold]  "
            "[dim](MyNegotiator)[/dim]\n"
            "[white]Goal   :[/white] Maximise  [green]Advantage[/green] + [red]Concealing[/red]  "
            "[dim](ANL 2026 scoring)[/dim]\n"
            "[white]Methods:[/white] Deceptive bidding · Frequency opponent model · Boulware acceptance",
            border_style="cyan",
            title="[bold]CAI 2026[/bold]",
            padding=(1, 4),
        )
    )
    console.print()


def print_design() -> None:
    console.rule("[bold cyan]1 - Agent Design[/bold cyan]")
    console.print()

    console.print(
        Panel(
            "[bold]ANL 2026 Score = Advantage + Concealing[/bold]\n\n"
            "  [green]Advantage[/green]  =  ufun(agreement) - reserved_value\n"
            "             [dim]How much better than our walkaway value?[/dim]\n\n"
            "  [red]Concealing[/red] =  1 - normalise(opponent modelling accuracy)\n"
            "             [dim]How well did we hide our preferences?[/dim]",
            title="Scoring Formula",
            border_style="yellow",
            padding=(0, 2),
        )
    )
    console.print()

    bid_panel = Panel(
        "[bold]Kaleidoscope Bidding[/bold]\n\n"
        "Offer from the top-25% of rational\n"
        "outcomes, but pick whichever looks\n"
        "[italic]most different[/italic] from recent bids.\n\n"
        "score(o) = alpha*u_norm + (1-alpha)*div\n"
        "alpha: 0.4 early -> 0.9 near deadline\n\n"
        "[dim]Rotating issue-value patterns prevents\n"
        "the opponent from learning our weights.[/dim]",
        title="[red]Bidding (Concealing)[/red]",
        border_style="red",
        padding=(0, 1),
    )

    model_panel = Panel(
        "[bold]Frequency Opponent Model[/bold]\n\n"
        "Count how often each issue value\n"
        "appears in the opponent's offers.\n"
        "High frequency -> high est. utility.\n\n"
        "u_opp(o) = mean_i[freq_i(o[i])/N_i]\n\n"
        "[dim]Stored in private_info[\"opponent_ufun\"]\n"
        "for ANL concealing evaluation.[/dim]",
        title="[blue]Modelling[/blue]",
        border_style="blue",
        padding=(0, 1),
    )

    accept_panel = Panel(
        "[bold]Boulware Acceptance[/bold]\n\n"
        "Time-adaptive threshold:\n"
        "  t=0  -> accept >= 90% of range\n"
        "  t=1  -> accept >= 10% of range\n\n"
        "thresh = rv + (max-rv)\n"
        "       * max(0.1, 0.9 - 0.8*t)\n\n"
        "[dim]Smooth concession toward deadline\n"
        "maximises expected Advantage.[/dim]",
        title="[green]Acceptance (Advantage)[/green]",
        border_style="green",
        padding=(0, 1),
    )

    console.print(Columns([bid_panel, model_panel, accept_panel]))
    console.print()


def print_demo(scenario_name: str) -> None:
    console.rule(f"[bold cyan]2 - Live Demo: {scenario_name} vs BOA[/bold cyan]")
    console.print()

    s = Scenario.load(SCENARIOS_DIR / scenario_name, ignore_discount=True)
    if s is None:
        console.print(f"[red]Could not load scenario '{scenario_name}'[/red]")
        return

    outcomes = list(s.outcome_space.enumerate_or_sample())
    n_issues = len(list(s.outcome_space.issues))
    console.print(
        f"[white]Scenario[/white]  {scenario_name}  "
        f"[dim]({len(outcomes)} outcomes, {n_issues} issues)[/dim]"
    )
    console.print("[white]Opponent [/white]  BOA Negotiator")

    first = random.choice([True, False])
    order_str = "MyNegotiator first" if first else "BOA first"
    console.print(f"[white]Order    [/white]  [yellow]{order_str}[/yellow]")
    console.print()

    m, scores = run_negotiation(s, "examples.boa.BOANeg", negotiator_first=first)

    if m.agreement:
        console.print(f"[green]Agreement reached[/green]  {m.agreement}")
    else:
        console.print("[red]No agreement reached[/red]")
    console.print(f"[white]Rounds   [/white]  {m.current_step} / {m.n_steps}")
    console.print()

    # Score table
    table = Table(
        title="ANL 2026 Scores",
        show_header=True,
        header_style="bold magenta",
        padding=(0, 1),
    )
    table.add_column("Agent", style="cyan", min_width=18)
    table.add_column("Advantage", justify="right", style="green")
    table.add_column("Concealing", justify="right", style="red")
    table.add_column("Total Score", justify="right", style="bold yellow")

    my_score = None
    for agent, sd in scores.items():
        table.add_row(
            agent,
            f"{sd['Advantage']:.3f}",
            f"{sd['Concealing']:.3f}",
            f"{sd['Score']:.3f}",
        )
        if agent == "MyNegotiator":
            my_score = sd["Score"]

    console.print(table)

    if my_score is not None:
        if my_score > 0.7:
            rating = "[bold green]Excellent[/bold green]"
        elif my_score > 0.4:
            rating = "[bold yellow]Good[/bold yellow]"
        else:
            rating = "[yellow]Developing[/yellow]"
        console.print(f"\n[dim]Performance rating: {rating} (score {my_score:.3f})[/dim]")

    # Trace excerpt (best-effort)
    try:
        trace_df = m.full_trace_with_utils_df()
        if not trace_df.empty:
            console.print()
            console.rule("[dim]Negotiation Trace (first 5 + last 3 rounds)[/dim]", style="dim")
            import pandas as pd
            excerpt = pd.concat([trace_df.head(5), trace_df.tail(3)]) if len(trace_df) > 8 else trace_df
            ttable = Table(show_header=True, header_style="bold dim", padding=(0, 1))
            for col in excerpt.columns:
                ttable.add_column(str(col), style="dim", min_width=6)
            for _, row in excerpt.iterrows():
                ttable.add_row(
                    *[str(round(v, 3) if isinstance(v, float) else v) for v in row]
                )
            console.print(ttable)
    except Exception:
        pass  # trace is supplementary

    console.print()


def print_tournament(scenario_name: str) -> None:
    console.rule("[bold cyan]3 - Mini-Tournament[/bold cyan]")
    console.print()
    console.print(
        "[dim]4 opponents x 2 orderings x 1 scenario = 8 matchups[/dim]"
    )
    console.print()

    s = Scenario.load(SCENARIOS_DIR / scenario_name, ignore_discount=True)
    if s is None:
        console.print(f"[red]Could not load scenario '{scenario_name}'[/red]")
        return

    matchups = [
        (opp_dotted, opp_name, first)
        for opp_dotted, opp_name in OPPONENTS
        for first in [True, False]
    ]

    results: dict[str, list[float]] = {opp_name: [] for _, opp_name in OPPONENTS}
    all_scores: list[float] = []

    for opp_dotted, opp_name, first in track(matchups, description="Running..."):
        try:
            _, scores = run_negotiation(s, opp_dotted, negotiator_first=first)
            my_s = scores.get("MyNegotiator", {}).get("Score", 0.0)
            results[opp_name].append(my_s)
            all_scores.append(my_s)
        except Exception:
            pass

    table = Table(
        title=f"MyNegotiator vs Each Opponent on '{scenario_name}'",
        show_header=True,
        header_style="bold magenta",
        padding=(0, 1),
    )
    table.add_column("Opponent", style="cyan", min_width=12)
    table.add_column("Avg Score", justify="right", style="bold yellow")
    table.add_column("Matchups", justify="right")
    table.add_column("Rating", justify="left")

    for _, opp_name in OPPONENTS:
        lst = results[opp_name]
        if lst:
            avg = sum(lst) / len(lst)
            if avg > 0.6:
                rating = "[green]Strong[/green]"
            elif avg > 0.35:
                rating = "[yellow]Moderate[/yellow]"
            else:
                rating = "[red]Weak[/red]"
            table.add_row(opp_name, f"{avg:.3f}", str(len(lst)), rating)

    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        table.add_section()
        table.add_row(
            "[bold]Overall[/bold]",
            f"[bold]{overall:.3f}[/bold]",
            f"[bold]{len(all_scores)}[/bold]",
            "",
        )

    console.print(table)
    console.print()


def print_objectives() -> None:
    console.rule("[bold cyan]4 - Course Learning Objectives[/bold cyan]")
    console.print()

    objectives = [
        (
            "Centralized vs Collaborative AI",
            "No central coordinator assigns outcomes. Each agent independently "
            "maximises its own utility through bilateral communication -- a pure "
            "collaborative paradigm contrasted with centralised planning where a "
            "single authority computes a joint allocation for all parties.",
        ),
        (
            "Co-active Design",
            "Three clearly separated, human-interpretable modules: bidding, "
            "opponent modelling, and acceptance. Each can be understood, reasoned "
            "about, and tuned independently -- supporting co-active design where "
            "humans remain informed and in control of the agent's behaviour.",
        ),
        (
            "Automated Negotiation Principles",
            "Implements the SAO (Simultaneous Alternating Offers) protocol with "
            "utility functions, reserved values, time-based Boulware concession, "
            "and the ANL 2026 scoring formula (Advantage + Concealing) -- core "
            "principles of automated negotiation from the ANAC competition series.",
        ),
        (
            "Agent Coordination Mechanisms",
            "Agents coordinate via offer/counter-offer sequences with no shared "
            "memory or central broker. The Boulware acceptance curve creates "
            "implicit deadline pressure driving convergence; the deceptive bidding "
            "prevents the opponent from exploiting knowledge of our preferences.",
        ),
        (
            "Agent Interaction Protocols",
            "Full SAO protocol implementation: alternating offers, Accept/ "
            "Reject/End responses, round-limit enforcement, and the ANL 2026 "
            "opponent-modelling evaluation (Kendall-tau correlation between the "
            "opponent's model and our true utility function).",
        ),
    ]

    for i, (title, desc) in enumerate(objectives, 1):
        console.print(
            Panel(
                f"[dim]{desc}[/dim]",
                title=f"[bold green][+] {i}. {title}[/bold green]",
                border_style="green",
                padding=(0, 2),
            )
        )
        console.print()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    scenario: str = typer.Option("Camera", help="Scenario for the demo and tournament."),
    quick: bool = typer.Option(False, "--quick", help="Skip the tournament."),
) -> None:
    """
    ANL 2026 Showcase -- Kaleidoscope Negotiation Agent.

    Runs a live demo negotiation, an optional mini-tournament, and explains
    how the agent satisfies each Collaborative AI course learning objective.
    """
    print_welcome()
    print_design()
    print_demo(scenario)
    if not quick:
        print_tournament(scenario)
    print_objectives()
    console.rule("[bold cyan]End of Showcase[/bold cyan]")
    console.print()


if __name__ == "__main__":
    app()
