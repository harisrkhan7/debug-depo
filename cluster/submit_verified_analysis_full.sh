#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ANALYSIS_MODE=full
exec "$SCRIPT_DIR/submit_verified_analysis.sh" "$@"
