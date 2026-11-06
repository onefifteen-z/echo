#!/usr/bin/env bash
# Shared helpers for bin/echo-live*: pick the repo venv and restore the terminal on exit.

ECHO_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo_tui_bin() {
  if [[ -x "$ECHO_REPO_ROOT/.venv/bin/echo-tui" ]]; then
    printf '%s\n' "$ECHO_REPO_ROOT/.venv/bin/echo-tui"
  else
    printf '%s\n' "echo-tui"
  fi
}

# A SIGKILLed Textual app cannot turn mouse tracking / alt screen off itself.
echo_restore_terminal() {
  printf '\e[?1000l\e[?1002l\e[?1003l\e[?1006l\e[?1015l\e[?1049l\e[?25h' 2>/dev/null || true
  stty sane 2>/dev/null || true
}
