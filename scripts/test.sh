#!/bin/sh
dirs="src"

# Exit immediately if a pipeline exits with a non-zero status.
set -e

# No buffering of log messages, all goes straight to stdout
export PYTHONUNBUFFERED=1

echo "========================= format check =========================="
ruff format $src --config pyproject.toml --preview --check

echo "=============================== linting check =================================="
ruff check $src --config pyproject.toml --preview
echo "syntax ok"

# Run pytest with coverage report
coverage run --rcfile=pyproject.toml --source=$dirs -m pytest $1
coverage report -m --rcfile=pyproject.toml
