"""Tests for learning and searching the student's notes (RAG)."""

from pathlib import Path

import pytest

from studybuddy.knowledge import (
    CHUNK_LINES,
    CHUNK_OVERLAP,
    Notes,
    cosine,
    format_notes,
    split_lines,
)
from studybuddy.llm import ProviderError
from studybuddy.workspace import WorkspaceError
from tests.fakes import keyword_embed

RECURSION = "# Recursion\nA recursive function needs a base case.\nElse the stack overflows.\n"
RAG = "# RAG\nRAG finds the closest embedding vector to the question.\n"


@pytest.fixture
def notes_dir(tmp_path: Path) -> Path:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "recursion.md").write_text(RECURSION)
    (tmp_path / "notes" / "rag.md").write_text(RAG)
    (tmp_path / "notes" / "photo.png").write_bytes(b"\x89PNG")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("recursion recursion")
    return tmp_path


def learned(root: Path, target: str = "notes") -> Notes:
    notes = Notes(root, keyword_embed)
    list(notes.learn(notes.find_files(target)))
    return notes


def test_split_lines_overlaps_and_numbers_lines() -> None:
    text = "\n".join(f"line {n}" for n in range(1, 101))
    pieces = split_lines(text)
    assert pieces[0][:2] == (1, CHUNK_LINES)
    assert pieces[1][0] == CHUNK_LINES - CHUNK_OVERLAP + 1
    assert pieces[-1][1] == 100


def test_split_lines_skips_blank_text() -> None:
    assert split_lines("\n\n  \n") == []


def test_cosine() -> None:
    assert cosine([1, 0], [1, 0]) == pytest.approx(1)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0)
    assert cosine([0, 0], [1, 1]) == 0


def test_find_files_reads_text_and_skips_tool_folders(notes_dir: Path) -> None:
    notes = Notes(notes_dir, keyword_embed)
    names = sorted(path.name for path in notes.find_files("."))
    assert names == ["rag.md", "recursion.md"]


def test_find_files_stays_inside_the_project(notes_dir: Path) -> None:
    notes = Notes(notes_dir / "notes", keyword_embed)
    with pytest.raises(WorkspaceError, match="outside the project"):
        notes.find_files("..")
    with pytest.raises(WorkspaceError, match="No such file"):
        notes.find_files("missing")


def test_find_files_explains_unsupported_files(notes_dir: Path) -> None:
    with pytest.raises(WorkspaceError, match="PDFs"):
        Notes(notes_dir, keyword_embed).find_files("notes/photo.png")


def test_search_returns_the_relevant_note(notes_dir: Path) -> None:
    hits = learned(notes_dir).search("why does recursion need a base case?")
    assert hits
    assert hits[0].chunk.source == "notes/recursion.md"
    assert hits[0].chunk.label == "notes/recursion.md:1-3"


def test_search_ignores_unrelated_notes(notes_dir: Path) -> None:
    hits = learned(notes_dir).search("recursion base case")
    assert all(hit.chunk.source != "notes/rag.md" for hit in hits)


def test_index_is_saved_and_reloaded(notes_dir: Path) -> None:
    learned(notes_dir)
    again = Notes(notes_dir, keyword_embed)
    assert again.files() == ["notes/rag.md", "notes/recursion.md"]


def test_relearning_a_file_replaces_its_old_chunks(notes_dir: Path) -> None:
    notes = learned(notes_dir)
    (notes_dir / "notes" / "rag.md").write_text("RAG uses a vector index.\n")
    list(notes.learn(notes.find_files("notes/rag.md")))
    rag = [chunk for chunk in notes.chunks if chunk.source == "notes/rag.md"]
    assert [chunk.text for chunk in rag] == ["RAG uses a vector index."]


def test_forget_deletes_the_index(notes_dir: Path) -> None:
    notes = learned(notes_dir)
    notes.forget()
    assert notes.chunks == []
    assert not notes.index_path.exists()
    assert Notes(notes_dir, keyword_embed).chunks == []


def test_failed_embedding_saves_nothing(notes_dir: Path) -> None:
    def down(texts: list[str]) -> list[list[float]]:
        raise ProviderError("Ollama is down")

    notes = Notes(notes_dir, down)
    with pytest.raises(ProviderError):
        list(notes.learn(notes.find_files("notes")))
    assert not notes.index_path.exists()


def test_unknown_index_versions_are_ignored(notes_dir: Path) -> None:
    notes = learned(notes_dir)
    notes.index_path.write_text('{"version": 999, "chunks": []}')
    assert Notes(notes_dir, keyword_embed).chunks == []


def test_format_notes_names_each_source(notes_dir: Path) -> None:
    hits = learned(notes_dir).search("recursion base case")
    assert "From `notes/recursion.md:1-3`" in format_notes(hits)
