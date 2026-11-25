"""Command-line entry point: the ``studybuddy`` typer app and all its commands."""

from __future__ import annotations

from pathlib import Path

import typer

from studybuddy import ui
from studybuddy.agent import Agent
from studybuddy.chat import run_repl
from studybuddy.knowledge import Notes
from studybuddy.llm import ProviderError, embed_texts, get_provider
from studybuddy.project import (
    PROJECT_DIR_NAME,
    ProjectAlreadyInitializedError,
    find_project_root,
    init_project,
)

console = ui.make_console()

app = typer.Typer(
    name="studybuddy",
    help="Study Buddy — a Socratic AI coding tutor that debugs your code and teaches as it goes.",
    add_completion=False,
)

PROVIDER_OPTION = typer.Option(
    None,
    "--provider",
    "-p",
    help="LLM to use: ollama (free, local), anthropic, gemini, or mock. Auto-detected if omitted.",
)
MODEL_OPTION = typer.Option(
    None,
    "--model",
    "-m",
    help="Model name for the chosen provider (e.g. llama3.2:1b). You can also switch in the "
    "chat with /model.",
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    provider: str | None = PROVIDER_OPTION,
    model: str | None = MODEL_OPTION,
) -> None:
    """Run with no command to start chatting in the current folder."""
    if ctx.invoked_subcommand is None:
        chat(provider, model)


@app.command()
def init() -> None:
    """Initialize a Study Buddy project in the current directory."""
    cwd = Path.cwd()
    try:
        project_dir = init_project(cwd)
    except ProjectAlreadyInitializedError:
        ui.init_exists(console, cwd, PROJECT_DIR_NAME)
        raise typer.Exit(code=1) from None
    ui.init_done(console, project_dir)


@app.command()
def chat(
    provider: str | None = PROVIDER_OPTION,
    model: str | None = MODEL_OPTION,
) -> None:
    """Start an interactive tutoring session (same as running studybuddy on its own)."""
    # DESIGN: no init needed; files are limited to the nearest Study Buddy project, or else
    # to the folder the chat was started in.
    root = find_project_root(Path.cwd()) or Path.cwd()
    try:
        llm = get_provider(provider, model)
    except (ProviderError, ValueError) as exc:
        ui.error(console, str(exc))
        raise typer.Exit(code=1) from None
    run_repl(console, Agent(llm, root, Notes(root, embed_texts)))
