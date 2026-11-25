# Study Buddy

**A free AI coding tutor that runs in your terminal.**

Ask it programming questions, let it explain and fix the bugs in your files, and teach it
your own course notes so its answers use them. Everything runs on your own computer, with
no account, no API key and no cost.

```
╭───────────────────────────────────────────────╮
│                                               │
│   █▀▀ ▀█▀ █ █ █▀▄ █ █   █▀▄ █ █ █▀▄ █▀▄ █ █   │
│   ▀▀█  █  █ █ █ █ ▀▄▀   █▀▄ █ █ █ █ █ █ ▀▄▀   │
│   ▀▀▀  ▀  ▀▀▀ ▀▀   ▀    ▀▀  ▀▀▀ ▀▀  ▀▀   ▀    │
│                                               │
│   Your coding tutor. Ask, read, fix, learn.   │
│                                               │
│   ● qwen2.5-coder:1.5b  ollama                │
│   Files  ~/cs101                              │
│   Notes  3 files                              │
│                                               │
╰───────────────────────────────────────────────╯
```

## What it does

| You want to… | Type |
| --- | --- |
| Ask a coding question or have a concept explained | just type your question |
| Ask about one of your files | `/read main.py`, then your question |
| Find and fix a bug | `/fix main.py` (it shows the change and only edits the file if you say yes) |
| Have answers use your course notes | `/learn notes` (a file or a folder) |
| See or clear what it learned | `/learn` · `/forget` |
| Switch to another free model | `/model` |
| Start a new conversation | `/reset` |
| See all commands / quit | `/help` · `exit` |

## Install

You need a computer with **at least 4 GB of RAM** and **about 3 GB of free disk space**.
Setup takes about 10 minutes, mostly downloading.

### Step 1: Install Ollama (it runs the AI model on your computer)

- **Windows or macOS:** download the installer from <https://ollama.com/download>, run
  it, and keep the Ollama app running.
- **Linux:** run
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

Check that it worked: `ollama --version` prints a version number.

### Step 2: Download the AI model (about 1 GB)

```bash
ollama pull qwen2.5-coder:1.5b
```

### Step 3: Install uv (it installs Study Buddy and the Python it needs)

- **macOS or Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows** (in PowerShell):
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

Then **close your terminal and open a new one**. You don't need to install Python
yourself; uv takes care of it.

### Step 4: Install Study Buddy

```bash
uv tool install https://github.com/endahiro/study-buddy-alu/releases/download/v1.0.0/studybuddy-1.0.0-py3-none-any.whl
```

If you then see a warning that the install folder is not on your `PATH`, run
`uv tool update-shell`, then close and reopen your terminal.

### Step 5: Run it

Go to a folder with your code or notes and start it:

```bash
cd path/to/your/project
studybuddy
```

Study Buddy can only open files inside the folder you start it in.

## Try it

```
studybuddy> explain recursion with a small example
studybuddy> /fix stats.py it crashes on an empty list
studybuddy> /learn notes
✓ Learned 3 files (3 passages). Your questions will use them from now on.
studybuddy> when is the lab 7 deadline?
▌ From your notes: notes/week7-llms.md:1-12

  The lab 7 deadline is Friday 14 November.
```

The first answer can take 10–30 seconds while the model loads; later ones are faster.

## Teaching it your notes

`/learn notes` reads the Markdown, text and code files in the `notes` folder. Each time you
ask a question, it looks for the parts of your notes that match it and gives them to the
model, and a `▌ From your notes:` line shows which ones it used. This technique is called
RAG (retrieval-augmented generation).

- The first `/learn` offers to download a small search model (`nomic-embed-text`, 274 MB).
- What it learned is saved in a `.studybuddy` folder, so it remembers next time.
- Changed a file? `/learn` it again. Want a clean slate? `/forget`.
- PDFs and Word files aren't supported yet; save them as text or Markdown first.

## Free models

Type `/model` in the chat to see these and switch by number. Ones you haven't downloaded
are downloaded for you.

| Model | Good for |
| --- | --- |
| `qwen2.5-coder:1.5b` | The default. Best all-round on a 4 GB laptop |
| `llama3.2:1b` | Fastest, but makes more factual mistakes |
| `qwen3:1.7b` | Asks you follow-up questions; slower |
| `qwen2.5-coder:3b` | A bit more careful, about 3x slower; needs 4.5 GB+ RAM |

Small models are sometimes wrong, so always read a proposed fix before accepting it.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Can't reach Ollama` | Start the Ollama app (Windows/macOS) or run `ollama serve` (Linux) |
| `No models downloaded yet` | Run `ollama pull qwen2.5-coder:1.5b`, or type `/model` and pick `1` |
| `studybuddy: command not found` | Run `uv tool update-shell`, then open a new terminal |
| `The model may have run out of memory` | Close other programs, or pick a smaller model with `/model` |
| Answers are slow | Normal on a laptop without a graphics card; `llama3.2:1b` is the fastest |

To uninstall: `uv tool uninstall studybuddy`.

## Optional: cloud models

If you have an API key, Study Buddy can use Anthropic or Google Gemini models instead of
local ones:

```bash
uv tool install "studybuddy[anthropic,gemini] @ https://github.com/endahiro/study-buddy-alu/releases/download/v1.0.0/studybuddy-1.0.0-py3-none-any.whl"
export GEMINI_API_KEY=your-key      # or ANTHROPIC_API_KEY
studybuddy -p gemini                # or -p anthropic
```

Without `-p`, Study Buddy picks the first available in this order: an Anthropic key, a
Gemini key, Ollama, then an offline demo mode.

## For developers

```bash
git clone https://github.com/endahiro/study-buddy-alu.git
cd study-buddy-alu
uv sync --all-extras          # installs everything, including test tools
uv run studybuddy             # runs your local copy
uv run pytest                 # 127 tests; no Ollama or API keys needed
uv run ruff check . && uv run mypy src
```

How the code is organised (`src/studybuddy/`):

| Part | What it does |
| --- | --- |
| `cli.py`, `chat.py` | The `studybuddy` command and the chat loop with its `/` commands |
| `agent.py` | The tutor: conversation history, system prompt, `/read`, `/fix` and notes |
| `knowledge.py` | `/learn`: splits notes into passages and finds the ones that match a question |
| `llm/` | One interface for every AI provider (Ollama, Anthropic, Gemini, mock) |
| `workspace.py` | Keeps file access inside the project folder and builds `/fix` diffs |
| `ui/` | Everything printed to the terminal: welcome screen, streamed answers, diffs |

Developed and tested on Linux (Ubuntu on WSL) with Python 3.10; needs Python 3.10 or newer (uv installs it for you).
