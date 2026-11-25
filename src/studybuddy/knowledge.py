"""The student's notes: retrieval-augmented generation (RAG) over their own files.

``/learn`` splits text files into overlapping chunks of lines, turns each chunk into a
vector with an embedding model, and saves them in ``.studybuddy/notes.json``. Before each
question, the chunks whose vectors are closest to the question's vector are sent to the
model with it, so answers can use the student's course notes and code.

Pure Python on purpose: a student's notes are a few thousand chunks at most, which a
plain cosine-similarity scan handles in milliseconds, so no vector database or numpy.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from studybuddy.project import PROJECT_DIR_NAME
from studybuddy.workspace import MAX_FILE_BYTES, WorkspaceError

# Embeds a batch of texts, one vector per text (``studybuddy.llm.embed_texts`` in the app).
Embedder = Callable[[list[str]], list[list[float]]]

INDEX_FILE_NAME = "notes.json"
INDEX_VERSION = 1

# Files /learn reads; everything else (images, PDFs, binaries) is skipped.
TEXT_SUFFIXES = frozenset(
    {".md", ".markdown", ".txt", ".rst", ".tex", ".csv",
     ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".kt", ".c", ".h", ".cpp", ".hpp",
     ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".sql", ".sh", ".html", ".css",
     ".json", ".yaml", ".yml", ".toml"}
)  # fmt: skip
# Folders that hold tools or dependencies, never the student's notes.
SKIP_DIRS = frozenset(
    {".git", ".venv", "venv", "node_modules", "__pycache__", PROJECT_DIR_NAME, ".mypy_cache",
     ".pytest_cache", ".ruff_cache", "dist", "build", ".idea", ".vscode"}
)  # fmt: skip

# DESIGN: chunks of ~40 lines (overlapping by 8 so an idea split across a boundary is still
# whole in one chunk) are small enough that 3 of them plus a question fit comfortably in a
# 1.5b model's context, and large enough to hold a full function or a paragraph or two.
CHUNK_LINES = 40
CHUNK_OVERLAP = 8
MAX_CHUNK_CHARS = 2_000
EMBED_BATCH = 16
TOP_K = 3
# DESIGN: measured with nomic-embed-text on three lecture-note files and ten questions:
# the right note scored 0.63-0.80 (and ranked first every time), wrong notes for those
# questions up to 0.60, and notes for unrelated questions ("what is a Python decorator?")
# up to 0.57. So a chunk must score MIN_SCORE, and be within SCORE_MARGIN of the best
# match, so one strong match doesn't drag weak ones along.
MIN_SCORE = 0.62
SCORE_MARGIN = 0.1
# nomic-embed-text is trained with these task prefixes; using them improves retrieval.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


@dataclass(frozen=True)
class Chunk:
    """A piece of a learned file.

    Attributes:
        source: The file's path relative to the project root.
        start: First line number (1-based).
        end: Last line number.
        text: The lines themselves.
        vector: The embedding of ``text``.
    """

    source: str
    start: int
    end: int
    text: str
    vector: list[float]

    @property
    def label(self) -> str:
        """Return e.g. ``notes/rag.md:1-40``, for showing where an answer came from."""
        return f"{self.source}:{self.start}-{self.end}"


@dataclass(frozen=True)
class Hit:
    """A chunk that matched a question, with its cosine similarity score."""

    chunk: Chunk
    score: float


def split_lines(text: str) -> list[tuple[int, int, str]]:
    """Split ``text`` into overlapping ``(start, end, chunk)`` pieces of lines."""
    lines = text.splitlines()
    pieces: list[tuple[int, int, str]] = []
    step = CHUNK_LINES - CHUNK_OVERLAP
    for first in range(0, len(lines), step):
        window = lines[first : first + CHUNK_LINES]
        body = "\n".join(window).strip()
        if body:
            pieces.append((first + 1, first + len(window), body[:MAX_CHUNK_CHARS]))
        if first + CHUNK_LINES >= len(lines):
            break
    return pieces


def cosine(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of two vectors (0 if either is all zeros)."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norms if norms else 0.0


class Notes:
    """The learned files of one project and the search over them.

    Attributes:
        root: The project root; only files inside it can be learned.
        chunks: Every learned chunk, in the order files were learned.
    """

    def __init__(self, root: Path, embed: Embedder) -> None:
        """Load the saved index under ``root`` (if any); ``embed`` computes vectors."""
        self.root = root.resolve()
        self._embed = embed
        self.chunks: list[Chunk] = self._load()

    @property
    def index_path(self) -> Path:
        """Where the index is saved."""
        return self.root / PROJECT_DIR_NAME / INDEX_FILE_NAME

    def files(self) -> list[str]:
        """Return the learned files, sorted."""
        return sorted({chunk.source for chunk in self.chunks})

    def find_files(self, relative: str) -> list[Path]:
        """Return the text files ``/learn relative`` would read: one file, or a folder's files.

        Raises:
            WorkspaceError: If ``relative`` is outside the project, missing, or has no
                text files.
        """
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise WorkspaceError(
                f"{relative} is outside the project; only files under {self.root} work."
            )
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = []
            for folder, dirs, names in os.walk(path):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
                candidates += [Path(folder) / name for name in sorted(names)]
        else:
            raise WorkspaceError(f"No such file or folder: {relative}")
        found = [
            file
            for file in candidates
            if file.suffix.lower() in TEXT_SUFFIXES and 0 < file.stat().st_size <= MAX_FILE_BYTES
        ]
        if not found:
            raise WorkspaceError(
                f"No text files to learn in {relative} (Markdown, text and code files work; "
                "PDFs and Word files don't yet)."
            )
        return found

    def learn(self, files: list[Path]) -> Iterator[float]:
        """Add ``files`` to the index, yielding progress from 0 to 1, then save it.

        Files learned before are replaced, so re-learning picks up edits. Files that aren't
        UTF-8 text are skipped. Nothing is saved if embedding fails part-way.

        Raises:
            ProviderError: If the embedding model can't be reached.
        """
        pieces: list[tuple[str, int, int, str]] = []
        for file in files:
            try:
                text = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            source = file.relative_to(self.root).as_posix()
            pieces += [(source, start, end, body) for start, end, body in split_lines(text)]
        new: list[Chunk] = []
        for first in range(0, len(pieces), EMBED_BATCH):
            batch = pieces[first : first + EMBED_BATCH]
            vectors = self._embed([DOCUMENT_PREFIX + body for *_, body in batch])
            new += [
                Chunk(source, start, end, body, vector)
                for (source, start, end, body), vector in zip(batch, vectors, strict=True)
            ]
            yield min((first + len(batch)) / len(pieces), 1.0)
        replaced = {chunk.source for chunk in new}
        self.chunks = [c for c in self.chunks if c.source not in replaced] + new
        self._save()

    def search(self, question: str, k: int = TOP_K) -> list[Hit]:
        """Return up to ``k`` chunks relevant to ``question``, best first.

        Raises:
            ProviderError: If the embedding model can't be reached.
        """
        if not self.chunks:
            return []
        query = self._embed([QUERY_PREFIX + question])[0]
        hits = [Hit(chunk, cosine(query, chunk.vector)) for chunk in self.chunks]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        if not hits or hits[0].score < MIN_SCORE:
            return []
        floor = max(MIN_SCORE, hits[0].score - SCORE_MARGIN)
        return [hit for hit in hits if hit.score >= floor][:k]

    def forget(self) -> None:
        """Delete every learned chunk and the saved index."""
        self.chunks = []
        self.index_path.unlink(missing_ok=True)

    def _load(self) -> list[Chunk]:
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        # DESIGN: an index from another format version is ignored rather than migrated;
        # running /learn again rebuilds it.
        if not isinstance(data, dict) or data.get("version") != INDEX_VERSION:
            return []
        return [Chunk(**chunk) for chunk in data.get("chunks", [])]

    def _save(self) -> None:
        self.index_path.parent.mkdir(exist_ok=True)
        data = {"version": INDEX_VERSION, "chunks": [asdict(c) for c in self.chunks]}
        self.index_path.write_text(json.dumps(data), encoding="utf-8")


def format_notes(hits: list[Hit]) -> str:
    """Return ``hits`` as a block of labelled excerpts for the model."""
    parts = ["Notes from the student's own files (use them if they help, and name the file):"]
    parts += [f"From `{hit.chunk.label}`:\n```\n{hit.chunk.text}\n```" for hit in hits]
    return "\n\n".join(parts)
