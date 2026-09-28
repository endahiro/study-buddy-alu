"""Interactive chat REPL.

Reads one line at a time, streams the agent's reply to the terminal as it arrives, and
exits on ``exit``/``quit``, Ctrl+C, or Ctrl+D.
"""

from __future__ import annotations

from rich.console import Console

from studybuddy.agent import Agent

PROMPT = "[bold cyan]studybuddy>[/bold cyan] "
EXIT_COMMANDS = frozenset({"exit", "quit"})


def print_banner(console: Console, agent: Agent) -> None:
    """Print the startup line showing which provider is active."""
    console.print(f"[dim]using provider: {agent.llm.name}[/dim]")
    console.print("[dim]Type 'exit' or press Ctrl+C to leave.[/dim]\n")


def stream_reply(console: Console, agent: Agent, user_input: str) -> None:
    """Print the agent's reply to ``user_input`` token by token."""
    for chunk in agent.stream_response(user_input):
        # markup/highlight off so model output is printed verbatim.
        console.print(chunk, end="", markup=False, highlight=False, soft_wrap=True)
    console.print("\n")


def run_repl(console: Console, agent: Agent) -> None:
    """Run the read-eval-print loop until the user exits."""
    print_banner(console, agent)
    while True:
        try:
            user_input = console.input(PROMPT).strip()
        except (KeyboardInterrupt, EOFError):
            console.print()
            break
        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            break
        try:
            stream_reply(console, agent, user_input)
        except KeyboardInterrupt:
            # DESIGN: Ctrl+C mid-stream cancels only the current reply, not the session.
            console.print("\n[dim](interrupted)[/dim]\n")
    console.print("[dim]Bye![/dim]")
