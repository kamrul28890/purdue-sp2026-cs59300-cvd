#!/usr/bin/env bash
# Setup script for Unix-like shells to create a Python 3.10 virtualenv
# and install project dependencies.

set -euo pipefail

RECREATE=false
SKIP_REQUIREMENTS=false
USE_LOCKFILE=false

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -r|--recreate) RECREATE=true; shift;;
    --skip-requirements) SKIP_REQUIREMENTS=true; shift;;
    --use-lockfile) USE_LOCKFILE=true; shift;;
    *) shift;;
  esac
done

create_venv() {
  if command -v python3.10 >/dev/null 2>&1; then
    echo "Creating venv using python3.10"
    python3.10 -m venv .venv
    return
  fi

  if command -v python3 >/dev/null 2>&1 && [[ "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.10" ]]; then
    echo "Creating venv using python3"
    python3 -m venv .venv
    return
  fi

  if command -v python >/dev/null 2>&1 && [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.10" ]]; then
    echo "Creating venv using python"
    python -m venv .venv
    return
  fi

  echo "Python 3.10 not found on PATH. Please install Python 3.10 and re-run." >&2
  exit 1
}

if [ "$RECREATE" = true ] && [ -d .venv ]; then
  echo "Removing existing .venv..."
  rm -rf .venv
fi

if [ ! -d .venv ]; then
  create_venv
fi

echo "To activate: source .venv/bin/activate"

if [ "$SKIP_REQUIREMENTS" = false ]; then
  REQUIREMENTS_FILE="requirements-repro.txt"
  if [ "$USE_LOCKFILE" = true ]; then
    REQUIREMENTS_FILE="requirements-lock.txt"
  fi

  . .venv/bin/activate
  python -m pip install --upgrade pip wheel setuptools
  pip install -r "$REQUIREMENTS_FILE"
  pip install -e .
  deactivate
fi

echo "Setup complete. Activate with: source .venv/bin/activate"
