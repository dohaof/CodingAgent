#!/bin/sh
# Setup for cagent and its desktop client, for macOS and Linux.
#
# Deliberately thin: it finds a Python and hands over to tools/install.py, which
# is where the error handling lives. The Windows equivalent is install.cmd.
set -eu
cd "$(dirname "$0")"

PY=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PY="$candidate"
    break
  fi
done

if [ -z "$PY" ]; then
  echo >&2
  echo "  Python was not found on PATH." >&2
  echo "  Install Python 3.11 or newer, then run this again." >&2
  echo >&2
  exit 1
fi

exec "$PY" tools/install.py "$@"
