#!/bin/bash

# Exit immediately if a pipeline exits with a non-zero status.
set -e

# Function to compile a given requirements*.in file using uv
# $(1) - output file to generate
# $(2+) - input file(s) to compile
uv_compile_reqfile() {
    echo "Compiling ${1}"
    if [ -n "${2}" ]; then
        uv pip compile -q --index-strategy unsafe-best-match --extra "${@:2}" --output-file "${1}" pyproject.toml
    else
        uv pip compile -q --index-strategy unsafe-best-match --output-file "${1}" pyproject.toml
    fi
    chown "${EXTERNAL_UID}:${EXTERNAL_GID}" "${1}"
}

uv_compile_reqfile "requirements.txt"
uv_compile_reqfile "requirements_dev.txt" "dev"
