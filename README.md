# Study Buddy

A terminal-native AI coding agent that debugs student code **and teaches the underlying
concepts** as it goes — grounded in your own project and course materials.

Study Buddy is Socratic-first: it asks questions, surfaces concepts, and walks you through
the fix instead of silently applying it. Free by default (local Ollama or Gemini free tier),
bring-your-own-key for paid providers.

> Status: early scaffold. Only a mock LLM provider is wired up.

## Quickstart

```bash
uv sync
uv run studybuddy --help
uv run studybuddy init     # creates .studybuddy/ in the current directory
uv run studybuddy chat     # opens the REPL (type `exit` or press Ctrl+C to leave)
```

## Development

```bash
uv run pytest -v
uv run ruff check .
uv run mypy src/studybuddy/llm
```

## Architecture

All LLM access goes through the `LLMProvider` abstract base class in
`src/studybuddy/llm/base.py` (strategy pattern). Nothing outside `studybuddy.llm` imports a
concrete provider; use `studybuddy.llm.get_provider()` instead.
