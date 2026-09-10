#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run.sh - Entry point to start the Physical AI Studio server
#
# Runs database migrations (idempotent via Alembic) and starts the backend
# with the bundled UI via the physicalai-studio serve CLI.
#
# Usage:
#   ./run.sh
# -----------------------------------------------------------------------------

export PYTHONUNBUFFERED=1
export LIBRARY_PATH="$PWD/.venv/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

exec uv run --no-sync physicalai-studio serve
