"""Interactive chat REPL.

Reads one line at a time, streams the agent's reply to the terminal as it arrives, and
exits on ``exit``/``quit``, Ctrl+C, or Ctrl+D. Lines starting with ``/`` are commands.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table

from studybuddy.agent import Agent
from studybuddy.llm import FreeModel, ProviderError, download_model, get_provider, list_free_models
from studybuddy.workspace import WorkspaceError, propose_fix

PROMPT = "[bold cyan]studybuddy>[/bold cyan] "
EXIT_COMMANDS = frozenset({"exit", "quit"})

HELP_TEXT = """\
[bold]Ask any coding question[/bold], or use a command:
  [cyan]/read <file>[/cyan]                 share a file; your next message is about it
  [cyan]/fix <file> [what's wrong][/cyan]   find and fix bugs, then apply after you approve
  [cyan]/model[/cyan]                       list free models; [cyan]/model 2[/cyan] switches to #2
  [cyan]/reset[/cyan]                       start a fresh conversation
  [cyan]/help[/cyan]                        show this help
  [cyan]exit[/cyan]                         leave"""


def print_banner(console: Console, agent: Agent) -> None:
    """Print the startup lines showing which model is active."""
    console.print("[bold]Study Buddy[/bold] — ask a coding question, or type /help.")
    try:
        console.print(f"[dim]Model: {escape(agent.llm.model)} ({agent.llm.name})[/dim]")
    except ProviderError as exc:
        console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}")
    console.print("[dim]/model to switch models · exit or Ctrl+C to leave[/dim]\n")


def total_ram_gb() -> float | None:
    """Return this computer's RAM in GB, or None where the OS doesn't report it."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (AttributeError, ValueError, OSError):
        return None


def too_big(model: FreeModel, ram: float | None) -> bool:
    """Return True if ``model`` needs more RAM than this computer has."""
    return ram is not None and model.min_ram_gb > ram


def current_model(agent: Agent) -> str | None:
    """Return the active model name, or None if it can't be determined yet."""
    try:
        return agent.llm.model
    except ProviderError:
        return None


def show_models(console: Console, agent: Agent) -> None:
    """Print the numbered list of free models and which one is in use."""
    active = current_model(agent) if agent.llm.name == "ollama" else None
    ram = total_ram_gb()
    table = Table(box=None, padding=(0, 2), show_header=False)
    for number, model in enumerate(list_free_models(), start=1):
        if model.name == active:
            status = "[bold green]● in use[/bold green]"
        elif too_big(model, ram):
            status = f"[yellow]needs {model.min_ram_gb:g} GB RAM[/yellow]"
        elif model.installed:
            status = "[green]ready[/green]"
        else:
            status = "[dim]not downloaded[/dim]"
        table.add_row(f"[cyan]{number}[/cyan]", escape(model.name), status, escape(model.note))
    console.print("[bold]Free models[/bold] (run on your computer, no key needed):")
    console.print(table)
    console.print("[dim]Switch with /model <number or name>.[/dim]\n")


def find_model(models: list[FreeModel], choice: str) -> FreeModel:
    """Return the model picked by number or name; unknown names are treated as new models."""
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        return models[int(choice) - 1]
    for model in models:
        if model.name == choice:
            return model
    return FreeModel(name=choice, note="", installed=False)


def download(console: Console, name: str) -> None:
    """Download ``name`` with a progress bar."""
    columns = (TextColumn("Downloading {task.description}"), BarColumn(), TaskProgressColumn())
    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task(escape(name), total=1.0)
        for fraction in download_model(name):
            progress.update(task, completed=fraction)


def switch_model(console: Console, agent: Agent, choice: str) -> None:
    """Switch the agent to a free model, downloading it first if the student agrees."""
    model = find_model(list_free_models(), choice)
    ram = total_ram_gb()
    if too_big(model, ram) and not confirm(
        console,
        f"{escape(model.name)} needs about {model.min_ram_gb:g} GB of RAM and this computer "
        f"has {ram:.1f} GB, so it may crash or be very slow. Use it anyway?",
    ):
        console.print("[dim]Kept the current model.[/dim]\n")
        return
    if not model.installed:
        if choice.isdigit():
            console.print(f"[yellow]There is no model #{escape(choice)}. Type /model.[/yellow]\n")
            return
        what = f" — {model.note}" if model.note else ""
        if not confirm(
            console, f"{escape(model.name)} isn't downloaded{escape(what)}. Download it?"
        ):
            console.print("[dim]Kept the current model.[/dim]\n")
            return
        download(console, model.name)
    agent.llm = get_provider("ollama", model.name)
    console.print(
        f"[bold green]✓[/bold green] Now using {escape(model.name)}. "
        "[dim](The conversation so far is kept.)[/dim]\n"
    )


def print_stream(console: Console, chunks: Iterator[str]) -> str:
    """Print ``chunks`` as they arrive and return the full text."""
    parts: list[str] = []
    for chunk in chunks:
        parts.append(chunk)
        # markup/highlight off so model output is printed verbatim.
        console.print(chunk, end="", markup=False, highlight=False, soft_wrap=True)
    console.print("\n")
    return "".join(parts)


def stream_reply(console: Console, agent: Agent, user_input: str) -> str:
    """Print the agent's reply to ``user_input`` token by token and return it."""
    return print_stream(console, agent.stream_response(user_input))


def confirm(console: Console, question: str) -> bool:
    """Ask a yes/no question; Ctrl+C or Ctrl+D count as no."""
    try:
        return Confirm.ask(question, console=console, default=False)
    except (KeyboardInterrupt, EOFError):
        console.print()
        return False


def run_fix(console: Console, agent: Agent, relative: str, problem: str) -> None:
    """Ask for a fix to ``relative``, show the diff, and write it if the student agrees."""
    reply = print_stream(console, agent.request_fix(relative, problem))
    proposal = propose_fix(agent.root, relative, reply)
    if proposal is None:
        console.print("[yellow]No code changes were proposed.[/yellow]\n")
        return
    console.print(Syntax(proposal.diff, "diff", theme="ansi_dark", background_color="default"))
    if confirm(console, f"Apply these changes to {escape(relative)}?"):
        proposal.path.write_text(proposal.new_text, encoding="utf-8")
        console.print(f"[bold green]✓[/bold green] Updated {escape(relative)}\n")
    else:
        console.print("[dim]Left the file unchanged.[/dim]\n")


def handle_command(console: Console, agent: Agent, line: str) -> None:
    """Run a ``/command`` line."""
    command, _, rest = line.partition(" ")
    target, _, problem = rest.strip().partition(" ")
    command = command.lower()
    if command == "/help":
        console.print(HELP_TEXT + "\n")
    elif command == "/model":
        if rest.strip():
            switch_model(console, agent, rest.strip())
        else:
            show_models(console, agent)
    elif command == "/reset":
        agent.reset()
        console.print("[dim]Started a fresh conversation.[/dim]\n")
    elif command in ("/read", "/fix") and not target:
        console.print(f"[yellow]Usage: {command} <file>[/yellow]\n")
    elif command == "/read":
        agent.attach(target)
        console.print(f"[dim]Shared {escape(target)}. Now ask your question about it.[/dim]\n")
    elif command == "/fix":
        run_fix(console, agent, target, problem.strip())
    else:
        console.print(f"[yellow]Unknown command {escape(command)}. Type /help.[/yellow]\n")


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
            if user_input.startswith("/"):
                handle_command(console, agent, user_input)
            else:
                stream_reply(console, agent, user_input)
        except KeyboardInterrupt:
            # DESIGN: Ctrl+C mid-stream cancels only the current reply, not the session.
            console.print("\n[dim](interrupted)[/dim]\n")
        except (ProviderError, WorkspaceError) as exc:
            # DESIGN: a failed request or bad file path keeps the session alive.
            console.print(f"\n[bold red]Error:[/bold red] {escape(str(exc))}\n")
    console.print("[dim]Bye![/dim]")
