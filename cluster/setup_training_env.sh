#!/usr/bin/env bash
set -euo pipefail

# Run after cluster/setup_rollout_env.sh to add GPU preference-training tools
# without removing the rollout/evaluation packages used by later PBS jobs.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-debug-depo}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-$HOME/miniforge3}"

if command -v module >/dev/null 2>&1; then
  module load tools/prod
  module load miniforge/3
fi

CONDA_BIN="$MINIFORGE_ROOT/bin/conda"
if [[ -x "$CONDA_BIN" ]]; then
  eval "$("$CONDA_BIN" shell.bash hook)"
  conda activate "$ENV_NAME"
fi

cd "$ROOT_DIR"
source cluster/env/load.sh
UV_BIN="${UV:-$(command -v uv)}"
if [[ -z "$UV_BIN" ]]; then
  echo "uv is unavailable; run cluster/setup_rollout_env.sh first." >&2
  exit 127
fi
"$UV_BIN" sync --inexact --extra dev --extra swebench --extra training
"$UV_BIN" run python -c \
  'import accelerate, minisweagent, peft, swebench, torch, transformers; print("Preference training environment OK")'
