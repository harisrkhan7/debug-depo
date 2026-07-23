#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SWESMITH_MODE=smoke
exec "$SCRIPT_DIR/submit_swesmith.sh" "$@"
