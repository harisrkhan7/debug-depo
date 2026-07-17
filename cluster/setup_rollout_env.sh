#!/usr/bin/env bash
set -euo pipefail

# Install the project rollout dependencies into the debug-depo uv environment.
# Run this on the cluster after `cluster/setup_jupyter_env.sh` and after the
# repo has been synced to ~/debug-depo.

ENV_NAME="${ENV_NAME:-debug-depo}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-$HOME/miniforge3}"
LOAD_TOOLS_PROD="${LOAD_TOOLS_PROD:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$LOAD_TOOLS_PROD" == "1" ]]; then
  if command -v module >/dev/null 2>&1; then
    module load tools/prod
    module load miniforge/3
  else
    echo "Warning: module command not found; skipping tools/prod and miniforge/3." >&2
  fi
fi

CONDA_BIN="$MINIFORGE_ROOT/bin/conda"
if [[ -x "$CONDA_BIN" ]]; then
  eval "$("$CONDA_BIN" shell.bash hook)"
  conda activate "$ENV_NAME"
else
  echo "Warning: conda not found at $CONDA_BIN; using current Python environment." >&2
fi

cd "$ROOT_DIR"
source cluster/env/load.sh

python -m pip install -U pip uv

UV_BIN="${UV:-$(command -v uv)}"
if [[ -z "$UV_BIN" ]]; then
  echo "Could not find uv after installation." >&2
  exit 127
fi
export UV="$UV_BIN"

"$UV_BIN" sync --extra dev --extra swebench
scripts/install_mini_swe_agent_plus.sh

"$UV_BIN" run python - <<'PY'
import importlib.util

required = {
    "debug_depo": "debug-depo",
    "minisweagent": "mini-swe-agent-plus",
    "datasets": "datasets",
}

missing = [label for module, label in required.items() if importlib.util.find_spec(module) is None]
if missing:
    raise SystemExit(f"Missing after setup: {', '.join(missing)}")

print("Rollout environment OK: debug_depo, mini-swe-agent-plus, and datasets import.")
PY

cat <<MSG

Rollout environment is ready.

Next in the cluster notebook:
  1. Rerun the setup/environment cells.
  2. Do not restart vLLM if it is already serving.
  3. Rerun the one-instance smoke rollout cell.
MSG
