#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CACHE_BUILD_MODE=full
exec "$SCRIPT_DIR/submit_apptainer_cache.sh" "$@"
