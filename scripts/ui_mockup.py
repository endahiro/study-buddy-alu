"""Render every Study Buddy screen with fake data, to review the look before wiring it in.

Run:  uv run python scripts/ui_mockup.py [--width 80] [--colors 16] [--fast]
"""

from __future__ import annotations

import argparse
import difflib
import time
from collections.abc import Iterator
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

from studybuddy import ui

BUGGY = """\
def average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers) - 1
"""
FIXED = """\
def average(numbers):
    if not numbers:
        return 0
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)
"""

# Bursty chunks like a small Ollama model: fences split mid-marker, code arriving in lumps.
REPLY = [
    "Your `average` function has two problems.\n\n",
    "1. **Operator precedence.** `total / len(numbers) - 1` divides first, then subtracts 1",
    " from the result.\n2. **Empty lists.** `len([])` is 0, so you get a ",
    "`ZeroDivisionError`.\n\nThe concept here is *precedence*: `/` binds tighter than `-`. ",
    "Check this in the REPL:\n\n`",
    "``py",
    "thon\n>>> 10 / 5 - 1\n1.0\n>>> 10 / (5 - 1)\n",
    "2.5\n",
    "``",
    "`\n\nHere is the corrected file:\n\n```python\ndef average(numbers):\n    if not numbers:\n",
    "        return 0\n    total = 0\n    for n in numbers:\n",
    "        total += n\n    return total / len(numbers)\n",
    "```\n",
]


def chunks(parts: list[str], first_delay: float, delay: float) -> Iterator[str]:
    """Yield ``parts`` with a slow first token and a pause between bursts."""
    time.sleep(first_delay)
    for part in parts:
        yield part
        time.sleep(delay)


def fake_download(delay: float) -> Iterator[float]:
    """Yield download progress from 0 to 1."""
    for step in range(41):
        time.sleep(delay)
        yield step / 40


def section(console: Console, title: str) -> None:
    """Print a divider naming the next screen."""
    console.print(Rule(f"[dim]{title}[/dim]", style="dim", align="left"))
    console.print()


def main() -> None:
    """Render every screen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=None, help="force a terminal width")
    parser.add_argument("--colors", choices=["16", "256", "truecolor"], default=None)
    parser.add_argument("--fast", action="store_true", help="skip the animation delays")
    args = parser.parse_args()
    systems = {"16": "standard", "256": "256", "truecolor": "truecolor"}
    kwargs: dict[str, object] = {}
    if args.width:
        kwargs["width"] = args.width
    if args.colors:
        kwargs["color_system"] = systems[args.colors]
    console = ui.make_console(**kwargs)
    first, delay = (0, 0) if args.fast else (2.5, 0.25)

    section(console, "studybuddy init")
    ui.init_done(console, Path("/home/student/cs101/.studybuddy"))
    console.print()
    ui.init_exists(console, Path("/home/student/cs101"), ".studybuddy")
    console.print()
    ui.error(console, "Unknown provider 'gpt'; choose one of: anthropic, gemini, ollama, mock")

    section(console, "banner")
    ui.banner(console, "qwen2.5-coder:1.5b", "ollama", None)
    ui.banner(console, None, "ollama", "No models downloaded yet. Type /model to get one.")

    section(console, "/help")
    ui.help_text(console)

    section(console, "/model")
    ui.model_table(
        console,
        [
            ui.ModelRow("qwen2.5-coder:1.5b", "Default. Best all-round on a 4 GB laptop", "in_use"),
            ui.ModelRow("llama3.2:1b", "Fast, but makes basic factual mistakes", "ready"),
            ui.ModelRow("qwen3:1.7b", "Asks you follow-up questions; slower", "missing"),
            ui.ModelRow("qwen2.5-coder:3b", "A bit more careful, ~3x slower", "too_big", 4.5),
        ],
    )
    console.print("[sb.muted]Pick a number (Enter keeps the current one):[/] 3")
    ui.download_progress(console, "qwen3:1.7b", fake_download(0 if args.fast else 0.05))
    ui.switched_model(console, "qwen3:1.7b")
    ui.note(console, "Kept the current model.")
    ui.warn(console, "There is no model #9. Type /model.")

    section(console, "/read")
    ui.shared_file(console, "stats.py", 5)

    section(console, "streamed reply (watch the fences while it streams)")
    reply = ui.render_reply(
        console, chunks(REPLY, first, delay), "Waiting for qwen2.5-coder:1.5b to start"
    )

    section(console, "/fix: diff, then confirm")
    diff = "".join(
        difflib.unified_diff(
            BUGGY.splitlines(True), FIXED.splitlines(True), "a/stats.py", "b/stats.py"
        )
    )
    ui.diff_panel(console, "stats.py", diff)
    console.print("Apply these changes to stats.py? \\[y/n] [dim](n)[/dim]: y")
    ui.success(console, "Updated stats.py")
    ui.note(console, "Left the file unchanged.")
    ui.warn(console, "No code changes were proposed.")

    section(console, "errors and the rest of the chat loop")
    ui.warn(console, "Usage: /fix <file>")
    ui.warn(console, "Unknown command /wat. Type /help.")
    ui.error(
        console, "../secrets.txt is outside the project; only files under /home/student/cs101 work."
    )
    ui.error(console, "Could not reach Ollama at localhost:11434. Is it running?")
    ui.note(console, "Started a fresh conversation.")
    ui.interrupted(console)
    ui.goodbye(console)
    assert reply == "".join(REPLY)


if __name__ == "__main__":
    main()
