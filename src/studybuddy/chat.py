"""Interactive chat REPL.

Reads one line at a time, streams the agent's reply to the terminal as it arrives, and
exits on ``exit``/``quit``, Ctrl+C, or Ctrl+D. Lines starting with ``/`` are commands.

All output goes through ``studybuddy.ui``; this module only decides what to show.
"""

from __future__ import annotations

import os

from rich.console import Console
from rich.markup import escape

from studybuddy import ui
from studybuddy.agent import Agent
from studybuddy.knowledge import Hit
from studybuddy.llm import (
    EMBED_MODEL,
    EMBED_MODEL_SIZE,
    FreeModel,
    ProviderError,
    download_model,
    get_provider,
    list_free_models,
    model_installed,
)
from studybuddy.workspace import WorkspaceError, propose_fix

EXIT_COMMANDS = frozenset({"exit", "quit"})


def print_banner(console: Console, agent: Agent) -> None:
    """Print the welcome screen showing which model is active and where files come from."""
    notes = len(agent.notes.files()) if agent.notes is not None else None
    try:
        ui.welcome(console, agent.llm.model, agent.llm.name, None, agent.root, notes)
    except ProviderError as exc:
        ui.welcome(console, None, agent.llm.name, str(exc), agent.root, notes)


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


def show_models(console: Console, agent: Agent) -> list[FreeModel]:
    """Print the numbered list of free models and which one is in use, and return it."""
    active = current_model(agent) if agent.llm.name == "ollama" else None
    ram = total_ram_gb()
    models = list_free_models()
    rows = []
    for model in models:
        if model.name == active:
            status = "in_use"
        elif too_big(model, ram):
            status = "too_big"
        elif model.installed:
            status = "ready"
        else:
            status = "missing"
        rows.append(ui.ModelRow(model.name, model.note, status, model.min_ram_gb))
    ui.model_table(console, rows)
    return models


def pick_model(console: Console, agent: Agent) -> None:
    """List the free models and switch to the one the student picks; Enter keeps the current."""
    models = show_models(console, agent)
    choice = clean_choice(ui.ask_model_number(console))
    if not choice or choice.lower() in EXIT_COMMANDS:
        ui.note(console, "Kept the current model.")
    elif choice.isdigit() or any(model.name == choice for model in models):
        switch_model(console, agent, choice)
    else:
        ui.warn(console, f"There is no model {choice}.")


def clean_choice(choice: str) -> str:
    """Strip spaces and the brackets/quotes students copy from usage hints, e.g. ``<2>``."""
    return choice.strip().strip("<>[]\"'").strip()


def find_model(models: list[FreeModel], choice: str) -> FreeModel:
    """Return the model picked by number or name; unknown names are treated as new models."""
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        return models[int(choice) - 1]
    for model in models:
        if model.name == choice:
            return model
    return FreeModel(name=choice, note="", installed=False)


def switch_model(console: Console, agent: Agent, choice: str) -> None:
    """Switch the agent to a free model, downloading it first if the student agrees."""
    model = find_model(list_free_models(), choice)
    ram = total_ram_gb()
    # ui.confirm questions are Rich markup, so names from the model list are escaped.
    if too_big(model, ram) and not ui.confirm(
        console,
        f"{escape(model.name)} needs about {model.min_ram_gb:g} GB of RAM and this computer "
        f"has {ram:.1f} GB, so it may crash or be very slow. Use it anyway?",
    ):
        ui.note(console, "Kept the current model.")
        return
    if not model.installed:
        if choice.isdigit():
            ui.warn(console, f"There is no model #{choice}. Type /model.")
            return
        what = f" — {model.note}" if model.note else ""
        if not ui.confirm(
            console, f"{escape(model.name)} isn't downloaded{escape(what)}. Download it?"
        ):
            ui.note(console, "Kept the current model.")
            return
        ui.download_progress(console, model.name, download_model(model.name))
    agent.llm = get_provider("ollama", model.name)
    ui.switched_model(console, model.name)


def waiting_for(agent: Agent) -> str:
    """Return the spinner text shown until the model's first word arrives."""
    model = current_model(agent)
    return f"Waiting for {model} to start" if model else "Waiting for the model to start"


def stream_reply(console: Console, agent: Agent, user_input: str) -> str:
    """Render the agent's reply to ``user_input``, using relevant notes, and return it."""
    hits = recall(console, agent, user_input)
    return ui.render_reply(console, agent.stream_response(user_input, hits), waiting_for(agent))


def recall(console: Console, agent: Agent, question: str) -> list[Hit]:
    """Return the notes relevant to ``question`` and show which; [] if search fails."""
    try:
        hits = agent.recall(question)
    except ProviderError as exc:
        # DESIGN: notes are a bonus; if they can't be searched, still answer the question.
        ui.warn(console, f"Couldn't search your notes, so answering without them. {exc}")
        return []
    if hits:
        ui.notes_used(console, [hit.chunk.label for hit in hits])
    return hits


def learn(console: Console, agent: Agent, target: str) -> None:
    """Learn the text files at ``target``; with no target, list what was learned."""
    notes = agent.notes
    if notes is None:
        ui.warn(console, "Notes aren't available in this session.")
        return
    if not target:
        ui.notes_list(console, notes.files())
        return
    files = notes.find_files(target)
    if not model_installed(EMBED_MODEL):
        if not ui.confirm(
            console,
            f"Learning notes needs the free {EMBED_MODEL} model ({EMBED_MODEL_SIZE}), "
            "which runs on your computer. Download it?",
        ):
            ui.note(console, "Didn't learn anything.")
            return
        ui.download_progress(console, EMBED_MODEL, download_model(EMBED_MODEL))
    ui.progress_bar(console, "Learning", f"{len(files)} file(s)", notes.learn(files))
    sources = {file.relative_to(notes.root).as_posix() for file in files}
    ui.learned(console, len(files), sum(chunk.source in sources for chunk in notes.chunks))


def run_fix(console: Console, agent: Agent, relative: str, problem: str) -> None:
    """Ask for a fix to ``relative``, show the diff, and write it if the student agrees."""
    reply = ui.render_reply(
        console, agent.request_fix(relative, problem), f"Looking for bugs in {relative}"
    )
    proposal = propose_fix(agent.root, relative, reply)
    if proposal is None:
        ui.warn(console, "No code changes were proposed.")
        return
    ui.diff_panel(console, relative, proposal.diff)
    if ui.confirm(console, f"Apply these changes to {escape(relative)}?"):
        proposal.path.write_text(proposal.new_text, encoding="utf-8")
        ui.success(console, f"Updated {relative}")
    else:
        ui.note(console, "Left the file unchanged.")


def handle_command(console: Console, agent: Agent, line: str) -> None:
    """Run a ``/command`` line."""
    command, _, rest = line.partition(" ")
    target, _, problem = rest.strip().partition(" ")
    target = clean_choice(target)
    command = command.lower()
    if command == "/help":
        ui.help_text(console)
    elif command == "/model":
        if clean_choice(rest):
            switch_model(console, agent, clean_choice(rest))
        else:
            pick_model(console, agent)
    elif command == "/learn":
        learn(console, agent, clean_choice(rest))
    elif command == "/forget":
        if agent.notes is not None:
            agent.notes.forget()
        ui.note(console, "Forgot all learned notes.")
    elif command == "/reset":
        agent.reset()
        ui.note(console, "Started a fresh conversation.")
    elif command in ("/read", "/fix") and not target:
        ui.warn(console, f"Usage: {command} <file>")
    elif command == "/read":
        ui.shared_file(console, target, agent.attach(target))
    elif command == "/fix":
        run_fix(console, agent, target, problem.strip())
    else:
        ui.warn(console, f"Unknown command {command}. Type /help.")


def run_repl(console: Console, agent: Agent) -> None:
    """Run the read-eval-print loop until the user exits."""
    print_banner(console, agent)
    while True:
        try:
            user_input = console.input(ui.PROMPT).strip()
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
            ui.interrupted(console)
        except (ProviderError, WorkspaceError) as exc:
            # DESIGN: a failed request or bad file path keeps the session alive.
            ui.error(console, str(exc))
    ui.goodbye(console)
