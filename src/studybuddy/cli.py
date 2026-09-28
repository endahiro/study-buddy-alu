"""Command-line entry point: the ``studybuddy`` typer app and all its commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from studybuddy.agent import Agent
from studybuddy.chat import run_repl
from studybuddy.llm import get_provider
from studybuddy.project import (
    PROJECT_DIR_NAME,
    ProjectAlreadyInitializedError,
    find_project_root,
    init_project,
)

console = Console()

app = typer.Typer(
    name="studybuddy",
    help="Study Buddy — a Socratic AI coding tutor that debugs your code and teaches as it goes.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def init() -> None:
    """Initialize a Study Buddy project in the current directory."""
    cwd = Path.cwd()
    try:
        project_dir = init_project(cwd)
    except ProjectAlreadyInitializedError:
        console.print(
            f"[bold red]Error:[/bold red] {cwd} is already a Study Buddy project "
            f"([cyan]{PROJECT_DIR_NAME}/[/cyan] exists)."
        )
        raise typer.Exit(code=1) from None
    console.print(f"[bold green]✓[/bold green] Initialized Study Buddy project at {project_dir}")
    console.print("Run [bold cyan]studybuddy chat[/bold cyan] to start a session.")


@app.command()
def chat() -> None:
    """Start an interactive tutoring session in the current project."""
    root = find_project_root(Path.cwd())
    if root is None:
        console.print(
            "[bold red]Error:[/bold red] No Study Buddy project found in this directory "
            "or any parent.\nRun [bold cyan]studybuddy init[/bold cyan] first."
        )
        raise typer.Exit(code=1)
    run_repl(console, Agent(get_provider()))
