#!/usr/bin/env bash
set -euo pipefail

# This wrapper only reads collection manifests and trajectory files. It does
# not source the cluster environment because that setup creates cache folders.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "$PYTHON_BIN" ]] || ! "$PYTHON_BIN" -c \
  'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  cat >&2 <<'MSG'
SWE-smith progress requires Python 3.10 or newer.

The repository's project environment was not found and the available python3
is too old. On CX3, create the project environment with:

  bash cluster/setup_rollout_env.sh

Then run this command again. You may also set PYTHON to a Python 3.10+ binary.
MSG
  exit 2
fi
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m debug_depo.swesmith_progress "$@"
