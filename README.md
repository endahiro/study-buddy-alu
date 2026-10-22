# Study Buddy

A terminal-native AI coding agent that debugs student code **and teaches the underlying
concepts** as it goes — grounded in your own project and course materials.

Study Buddy is Socratic-first: it asks questions, surfaces concepts, and walks you through
the fix instead of silently applying it. Free by default (local Ollama or Gemini free tier),
bring-your-own-key for paid providers.

> Status: early. A simple tutor agent answers coding questions and fixes files with your
> approval, on Anthropic, Gemini, or Ollama.

## Quickstart

One-time install (free, no API key; needs [Ollama](https://ollama.com) running):

```bash
uv tool install --editable '.[ollama]'   # run from this repo; puts `studybuddy` on your PATH
```

Then, from any folder with code you want help with:

```bash
studybuddy
```

That's it: no `init` needed. Study Buddy starts on the best free model you have, and can
only read and write files inside that folder (or the nearest parent with `.studybuddy/`).

## In the chat

Ask any coding question; follow-ups remember the conversation. Commands:

| Command | What it does |
| --- | --- |
| `/model` | List the free models and pick one by number (downloads it if needed) |
| `/read <file>` | Share a file; your next message is about it |
| `/fix <file> [what's wrong]` | Explain the bug, show a diff, and apply it only if you say yes |
| `/reset` | Start a fresh conversation |
| `/help` | List commands |
| `exit` | Leave (or press Ctrl+C) |

## Free models

These run on your own computer through Ollama, with no key and no internet once
downloaded. `/model` shows which ones you have.

| Model | Notes |
| --- | --- |
| `qwen2.5-coder:1.5b` | Default. Best all-round on a 4 GB RAM laptop; accurate answers |
| `llama3.2:1b` | Fast, but makes basic factual mistakes |
| `qwen3:1.7b` | Asks you follow-up questions; slower |
| `qwen2.5-coder:3b` | A bit more careful, ~3x slower; wants 8 GB+ RAM |

Picked by benchmarking coding questions and bug fixes on a CPU-only laptop. Small local
models still get some fixes wrong, so always read the diff before you accept it.

## Other providers

`studybuddy -p anthropic|gemini|ollama|mock` picks a provider, and `-m <model>` a model.
Without `-p` it is auto-detected in this order: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, an
Ollama server on `localhost:11434`, then the offline mock.

Each provider's SDK is an optional extra, so install only what you use:

```bash
uv sync --extra anthropic    # Claude (ANTHROPIC_API_KEY or `ant auth login`)
uv sync --extra gemini       # Gemini (GEMINI_API_KEY)
uv sync --extra ollama       # local models via `ollama serve`
uv sync --all-extras         # everything (needed to run the full test suite)
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
