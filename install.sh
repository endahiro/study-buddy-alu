#!/bin/sh
# Study Buddy installer for macOS and Linux. Run:
#
#   curl -fsSL https://raw.githubusercontent.com/endahiro/study-buddy-alu/main/install.sh | sh
#
# First checks this computer has what Study Buddy needs, then installs whatever is
# missing: Ollama (runs the AI models), uv (installs Python apps), Study Buddy, and the
# default AI model. Safe to run again: it skips what's there and updates Study Buddy.
set -eu

PACKAGE="https://github.com/endahiro/study-buddy-alu/releases/download/v1.0.0/studybuddy-1.0.0-py3-none-any.whl"
MODEL="qwen2.5-coder:1.5b"
MIN_RAM_GB=3.5  # a "4 GB" computer reports a little under 4
MIN_DISK_GB=3   # Ollama + the model + Study Buddy, with some room to spare

if [ -t 1 ]; then
    BOLD="$(printf '\033[1m')"; CYAN="$(printf '\033[36m')"; GREEN="$(printf '\033[32m')"
    YELLOW="$(printf '\033[33m')"; RED="$(printf '\033[31m')"; DIM="$(printf '\033[2m')"
    RESET="$(printf '\033[0m')"
else
    BOLD=""; CYAN=""; GREEN=""; YELLOW=""; RED=""; DIM=""; RESET=""
fi
ok()    { printf '  %s✓%s %s\n' "$GREEN" "$RESET" "$1"; }
todo()  { printf '  %s•%s %s\n' "$CYAN" "$RESET" "$1"; }
warn()  { printf '  %s!%s %s\n' "$YELLOW" "$RESET" "$1"; }
fail()  { printf '\n%sError:%s %s\n' "$RED$BOLD" "$RESET" "$1" >&2; exit 1; }
step()  { printf '\n%s[%s/4]%s %s%s%s\n' "$CYAN$BOLD" "$1" "$RESET" "$BOLD" "$2" "$RESET"; }
has()   { command -v "$1" >/dev/null 2>&1; }

printf '\n%sStudy Buddy installer%s\n' "$BOLD" "$RESET"
printf '%sA free AI coding tutor for your terminal%s\n' "$DIM" "$RESET"

# --- Before installing: check this computer ---------------------------------------------

printf '\n%sChecking your computer%s\n' "$BOLD" "$RESET"
OS="$(uname)"
case "$OS" in
    Linux | Darwin) ok "System: $OS" ;;
    *) fail "This installer is for macOS and Linux. On Windows, use install.ps1 (see the README)." ;;
esac

has curl || fail "curl is needed. Install it (for example: sudo apt install curl), then run this again."

if curl -fsSI --max-time 10 https://github.com >/dev/null 2>&1; then
    ok "Internet connection"
else
    fail "No internet connection. Connect and run this again."
fi

if [ "$OS" = "Darwin" ]; then
    RAM_KB=$(( $(sysctl -n hw.memsize) / 1024 ))
else
    RAM_KB=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
fi
RAM_GB=$(awk "BEGIN {printf \"%.1f\", $RAM_KB / 1048576}")
if awk "BEGIN {exit !($RAM_GB >= $MIN_RAM_GB)}"; then
    ok "Memory: $RAM_GB GB"
else
    warn "Memory: $RAM_GB GB. 4 GB or more is recommended; answers may be slow or fail."
fi

DISK_GB=$(df -Pk "$HOME" | awk 'NR == 2 {printf "%.0f", $4 / 1048576}')
if [ "$DISK_GB" -ge "$MIN_DISK_GB" ]; then
    ok "Free disk space: $DISK_GB GB"
else
    fail "Only $DISK_GB GB of free disk space; Study Buddy needs about $MIN_DISK_GB GB. Free some up and run this again."
fi

if has ollama; then ok "Ollama: already installed"; else todo "Ollama: will be installed"; fi
if has uv; then ok "uv: already installed"; else todo "uv: will be installed"; fi
if has studybuddy; then todo "Study Buddy: will be updated"; else todo "Study Buddy: will be installed"; fi

# --- 1. Ollama ----------------------------------------------------------------------------

step 1 "Ollama (runs the AI model on your computer)"
if has ollama; then
    ok "Already installed"
elif [ "$OS" = "Darwin" ]; then
    has brew || fail "Install Ollama from https://ollama.com/download, open it once, then run this again."
    brew install --quiet ollama
    ok "Installed"
else
    printf '  %sThis may ask for your password.%s\n' "$DIM" "$RESET"
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Installed"
fi

# Ollama must be running to download the model. Start it if it isn't (e.g. WSL without systemd).
if ! ollama list >/dev/null 2>&1; then
    nohup ollama serve >/dev/null 2>&1 &
    tries=0
    until ollama list >/dev/null 2>&1; do
        tries=$((tries + 1))
        [ "$tries" -gt 30 ] && fail "Ollama didn't start. Run 'ollama serve' in another terminal, then run this again."
        sleep 1
    done
    ok "Started"
fi

# --- 2. uv --------------------------------------------------------------------------------

step 2 "uv (installs Study Buddy and the Python it needs)"
if has uv; then
    ok "Already installed"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
    PATH="$HOME/.local/bin:$PATH"
    export PATH
    has uv || fail "uv didn't install. See https://docs.astral.sh/uv/ and run this again."
    ok "Installed"
fi

# --- 3. Study Buddy -------------------------------------------------------------------------

step 3 "Study Buddy"
# --force so running the installer again updates it.
uv tool install --quiet --force "$PACKAGE" || fail "Installing Study Buddy failed (see the message above)."
uv tool update-shell >/dev/null 2>&1 || true
ok "Installed"

# --- 4. The AI model ------------------------------------------------------------------------

step 4 "AI model $MODEL (about 1 GB, only the first time)"
ollama pull "$MODEL" || fail "Downloading the model failed. Check your internet connection and run this again."
ok "Ready"

printf '\n%s✓ Study Buddy is installed.%s\n\n' "$GREEN$BOLD" "$RESET"
printf 'Open a %snew%s terminal, go to a folder with your code or notes, and run:\n\n' "$BOLD" "$RESET"
printf '    %sstudybuddy%s\n\n' "$CYAN$BOLD" "$RESET"
