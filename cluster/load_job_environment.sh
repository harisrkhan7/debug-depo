#!/usr/bin/env bash

# Bootstrap every scheduled collection/evaluation job consistently. Conda
# supplies the user-facing Python/uv commands; `uv run` then selects the
# project environment created by cluster/setup_rollout_env.sh.
ENV_NAME="${ENV_NAME:-debug-depo}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-$HOME/miniforge3}"

if command -v module >/dev/null 2>&1; then
  module load tools/prod
  module load miniforge/3
elif [[ -n "${PBS_JOBID:-}" ]]; then
  echo "The module command is unavailable in scheduled job $PBS_JOBID." >&2
  exit 1
fi

CONDA_BIN="$MINIFORGE_ROOT/bin/conda"
if [[ -x "$CONDA_BIN" ]]; then
  eval "$("$CONDA_BIN" shell.bash hook)"
  conda activate "$ENV_NAME"
elif [[ -n "${PBS_JOBID:-}" ]]; then
  cat >&2 <<MSG
Could not find Conda at:
  $CONDA_BIN

Run cluster/setup_jupyter_env.sh first, or set MINIFORGE_ROOT in the qsub
environment to the Miniforge directory containing bin/conda.
MSG
  exit 1
fi

# Optional site-specific additions. This file is deliberately ignored by Git.
if [[ -f cluster/env/modules.sh ]]; then
  source cluster/env/modules.sh
fi

if [[ -n "${PBS_JOBID:-}" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not available after activating Conda environment '$ENV_NAME'." >&2
    echo "Run cluster/setup_rollout_env.sh before submitting jobs." >&2
    exit 1
  fi
  if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is not available after loading tools/prod." >&2
    echo "Check the cluster software setup or add the correct site module to cluster/env/modules.sh." >&2
    exit 1
  fi
  if [[ ! -x "$PWD/.venv/bin/python" ]]; then
    echo "The project environment is missing at $PWD/.venv." >&2
    echo "Run cluster/setup_rollout_env.sh from this checkout before submitting jobs." >&2
    exit 1
  fi
fi

unset CONDA_BIN
