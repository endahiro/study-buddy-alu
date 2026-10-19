# Study Buddy

A terminal-native AI coding agent that debugs student code **and teaches the underlying
concepts** as it goes — grounded in your own project and course materials.

Study Buddy is Socratic-first: it asks questions, surfaces concepts, and walks you through
the fix instead of silently applying it. Free by default (local Ollama or Gemini free tier),
bring-your-own-key for paid providers.

> Status: early. Chat works against Anthropic, Gemini, or Ollama; tutoring features come later.

## Quickstart

```bash
uv sync --all-extras
uv run studybuddy --help
uv run studybuddy init     # creates .studybuddy/ in the current directory
uv run studybuddy chat     # opens the REPL (type `exit` or press Ctrl+C to leave)
```

## LLM providers

Each provider's SDK is an optional extra, so install only what you use:

```bash
uv sync --extra anthropic    # Claude (ANTHROPIC_API_KEY or `ant auth login`)
uv sync --extra gemini       # Gemini (GEMINI_API_KEY)
uv sync --extra ollama       # local models via `ollama serve`
uv sync --all-extras         # everything (needed to run the full test suite)
```

Pick one explicitly with `studybuddy chat --provider anthropic|gemini|ollama|mock`.
Override the model with `--model`, e.g. `studybuddy chat -p ollama -m llama3.2:1b`.
Otherwise it is auto-detected in this order: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
an Ollama server on `localhost:11434`, then the offline mock.

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
