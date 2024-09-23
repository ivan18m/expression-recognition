#!/bin/sh
src="src tests"

# Exit immediately if a pipeline exits with a non-zero status.
set -e

# no buffering of log messages, all goes straight to stdout
export PYTHONUNBUFFERED=1

echo "========================= format =========================="
ruff format $src --config pyproject.toml --preview
echo "========================= linting =========================="
ruff check $src --config pyproject.toml --fix --show-fixes --preview
