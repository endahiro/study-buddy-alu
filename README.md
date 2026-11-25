# Study Buddy

A free AI coding tutor that runs in your terminal. Ask it programming questions, let it
find and fix bugs in your files, and teach it your course notes so its answers use them.

It runs entirely on your own computer: no account, no API key, no cost.

## Requirements

- Windows 10/11, macOS or Linux
- 4 GB of RAM and 3 GB of free disk space
- An internet connection for the install (it works offline afterwards)

Nothing else: the installer sets up everything Study Buddy needs, including Python.

## Install

**macOS / Linux** — in a terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/endahiro/study-buddy-alu/main/install.sh | sh
```

**Windows** — in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/endahiro/study-buddy-alu/main/install.ps1 | iex"
```

The installer checks your computer, then installs [Ollama](https://ollama.com) (runs the
AI model), [uv](https://docs.astral.sh/uv/) (installs Python apps), Study Buddy, and the
AI model. It takes about 10 minutes. On Linux it asks for your password to install Ollama.

## Run

Open a **new** terminal, go to a folder with your code or notes, and run:

```bash
studybuddy
```

Study Buddy can only open files inside the folder you start it in.

## Commands

| Command | What it does |
| --- | --- |
| *(just type)* | Ask a question or have a concept explained |
| `/read <file>` | Share a file, then ask about it |
| `/fix <file>` | Explain and fix a bug; the file only changes if you say yes |
| `/learn <file or folder>` | Learn your notes so answers can use them |
| `/learn` · `/forget` | List or forget what it learned |
| `/model` | Switch to another free model |
| `/reset` | Start a new conversation |
| `/help` · `exit` | Show all commands · quit |

Example:

```
studybuddy> /learn notes
✓ Learned 3 files (3 passages). Your questions will use them from now on.

studybuddy> when is the lab 7 deadline?
▌ From your notes: notes/week7-llms.md:1-12

  The lab 7 deadline is Friday 14 November.
```

`/learn` reads Markdown, text and code files (not PDFs or Word files yet). It finds the
parts of your notes that match each question and gives them to the model; this is called
RAG (retrieval-augmented generation).

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Can't reach Ollama` | Open the Ollama app, or run `ollama serve` |
| `No models downloaded yet` | Type `/model` and pick `1` |
| `studybuddy: command not found` | Open a new terminal, or run `uv tool update-shell` |
| Out of memory, or answers are slow | Close other programs, or pick a smaller model with `/model` |

The first answer takes 10–30 seconds while the model loads. Small models can be wrong, so
read a proposed fix before accepting it.

## Manual install

If the installer doesn't work for you, do its four steps yourself:

1. Install Ollama from <https://ollama.com/download> and make sure it is running.
2. Download the model: `ollama pull qwen2.5-coder:1.5b`
3. Install uv: see <https://docs.astral.sh/uv/getting-started/installation/>, then open a
   new terminal.
4. Install Study Buddy:
   ```bash
   uv tool install https://github.com/endahiro/study-buddy-alu/releases/download/v1.0.0/studybuddy-1.0.0-py3-none-any.whl
   ```

To uninstall: `uv tool uninstall studybuddy`

## Development

```bash
git clone https://github.com/endahiro/study-buddy-alu.git
cd study-buddy-alu
uv sync --all-extras    # install with test tools
uv run studybuddy       # run your local copy
uv run pytest           # run the tests (no Ollama needed)
```

The code is in `src/studybuddy/`: `chat.py` (chat loop and commands), `agent.py` (the
tutor), `knowledge.py` (notes search), `llm/` (AI providers: Ollama, Anthropic, Gemini) and
`ui/` (terminal screens). To use Anthropic or Gemini instead of a local model, set
`ANTHROPIC_API_KEY` or `GEMINI_API_KEY` and run `studybuddy -p anthropic` or `-p gemini`.
