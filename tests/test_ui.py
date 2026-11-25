"""Tests for the terminal rendering in ``studybuddy.ui``."""

from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from studybuddy import ui
from studybuddy.ui.stream import balance, open_fence, split_finished
from studybuddy.ui.theme import MAX_WIDTH
from studybuddy.ui.views import WORDMARK


def plain_console(width: int = 80) -> tuple[Console, StringIO]:
    out = StringIO()
    return ui.make_console(file=out, width=width, color_system=None), out


# --- Streaming Markdown -------------------------------------------------------------------


def test_open_fence_tracks_unclosed_fences() -> None:
    assert open_fence("text\n```python\nx = 1\n") == "```"
    assert open_fence("```python\nx = 1\n```\n") is None
    assert open_fence("no code here") is None


def test_balance_closes_an_open_fence() -> None:
    assert balance("Fix:\n```python\nx = 1") == "Fix:\n```python\nx = 1\n```"


def test_balance_holds_back_a_half_written_fence_line() -> None:
    assert balance("Here:\n``") == "Here:\n"


def test_split_finished_cuts_at_a_blank_line_between_blocks() -> None:
    finished, rest = split_finished("First para.\n\nSecond para")
    assert finished == "First para."
    assert rest == "Second para"


def test_split_finished_never_cuts_inside_a_fence() -> None:
    text = "```python\nx = 1\n\ny = 2\n"
    assert split_finished(text) == ("", text)


def test_split_finished_keeps_lists_together() -> None:
    text = "1. one\n\n2. two\n"
    assert split_finished(text) == ("", text)


def test_plain_reply_is_printed_verbatim() -> None:
    console, out = plain_console()
    reply = ui.render_reply(console, iter(["**bold** ", "and `code`"]), "Waiting")
    assert reply == "**bold** and `code`"
    assert "**bold** and `code`" in out.getvalue()


def test_plain_reply_cut_off_by_an_error_ends_its_line() -> None:
    def failing() -> Iterator[str]:
        yield "half a sent"
        raise RuntimeError("boom")

    console, out = plain_console()
    with pytest.raises(RuntimeError):
        ui.render_reply(console, failing(), "Waiting")
    ui.error(console, "network down")
    assert "half a sent\nError: network down" in out.getvalue()


# --- Views --------------------------------------------------------------------------------


def test_warnings_carry_a_symbol_not_just_colour() -> None:
    console, out = plain_console()
    ui.warn(console, "Unknown command /wat.")
    assert out.getvalue().startswith("! Unknown command /wat.")


def test_markup_in_messages_is_shown_literally() -> None:
    console, out = plain_console()
    ui.error(console, "No such file: [red]x.py")
    assert "No such file: [red]x.py" in out.getvalue()


def test_diff_panel_is_capped_on_wide_terminals() -> None:
    console, out = plain_console(width=200)
    ui.diff_panel(console, "a.py", "--- a.py\n+++ a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n")
    widest = max(len(line) for line in out.getvalue().splitlines())
    assert widest <= MAX_WIDTH
    assert "+x = 2" in out.getvalue()
    assert "+1 -1" in out.getvalue()


def test_model_table_shows_every_status() -> None:
    console, out = plain_console(width=120)
    ui.model_table(
        console,
        [
            ui.ModelRow("a", "", "in_use"),
            ui.ModelRow("b", "", "ready"),
            ui.ModelRow("c", "", "missing"),
            ui.ModelRow("d", "", "too_big", 4.5),
        ],
    )
    text = out.getvalue()
    for label in ("● in use", "ready", "not downloaded", "needs 4.5 GB RAM"):
        assert label in text


def terminal_console(width: int) -> tuple[Console, StringIO]:
    out = StringIO()
    console = ui.make_console(file=out, width=width, force_terminal=True, color_system=None)
    return console, out


def test_welcome_screen_in_a_terminal(tmp_path: Path) -> None:
    console, out = terminal_console(100)
    ui.welcome(console, "qwen2.5-coder:1.5b", "ollama", None, tmp_path)
    text = out.getvalue()
    assert WORDMARK[0] in text
    assert "● qwen2.5-coder:1.5b" in text
    assert "Try asking" in text
    assert tmp_path.name in text


def test_welcome_screen_shows_a_missing_model(tmp_path: Path) -> None:
    console, out = terminal_console(100)
    ui.welcome(console, None, "ollama", "No models downloaded yet.", tmp_path)
    assert "✗ No models downloaded yet." in out.getvalue()


def test_welcome_falls_back_to_plain_banner_when_narrow(tmp_path: Path) -> None:
    console, out = terminal_console(40)
    ui.welcome(console, "m", "ollama", None, tmp_path)
    assert WORDMARK[0] not in out.getvalue()
    assert "Model: m (ollama)" in out.getvalue()


def test_welcome_falls_back_to_plain_banner_when_piped(tmp_path: Path) -> None:
    console, out = plain_console(100)
    ui.welcome(console, "m", "ollama", None, tmp_path)
    assert WORDMARK[0] not in out.getvalue()
    assert "Model: m (ollama)" in out.getvalue()


def test_long_folder_paths_keep_their_end() -> None:
    from studybuddy.ui.views import _short_path

    path = Path("/srv/very/long/directory/structure/for/courses/cs101")
    short = _short_path(path, limit=20)
    assert short.endswith("courses/cs101")
    assert short.startswith("…/")
    assert len(short) <= 20
