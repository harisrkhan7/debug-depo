#!/usr/bin/env bash
set -euo pipefail

# Login-node setup for using a Conda environment from Imperial JupyterHub / VS Code.
#
# Defaults can be overridden, for example:
#   ENV_NAME=debug-depo PYTHON_VERSION=3.12 bash cluster/setup_jupyter_env.sh

ENV_NAME="${ENV_NAME:-debug-depo}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
KERNEL_NAME="${KERNEL_NAME:-$ENV_NAME}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-debug-depo}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-$HOME/miniforge3}"
LOAD_TOOLS_PROD="${LOAD_TOOLS_PROD:-1}"

case "$PYTHON_VERSION" in
  3.10|3.10.*|3.11|3.11.*|3.12|3.12.*)
    ;;
  *)
    cat >&2 <<MSG
debug-depo requires Python >=3.10,<3.13.

Imperial's default Python may be newer, but this script creates an isolated
Conda environment, so use:
  PYTHON_VERSION=3.11 bash cluster/setup_jupyter_env.sh
or:
  PYTHON_VERSION=3.12 bash cluster/setup_jupyter_env.sh
MSG
    exit 1
    ;;
esac

if [[ "$LOAD_TOOLS_PROD" == "1" ]]; then
  if command -v module >/dev/null 2>&1; then
    module load tools/prod
    module load miniforge/3
  else
    echo "Warning: module command not found; skipping tools/prod and miniforge/3." >&2
  fi
fi

CONDA_BIN="$MINIFORGE_ROOT/bin/conda"
if [[ ! -x "$CONDA_BIN" ]]; then
  cat >&2 <<MSG
Could not find conda at:
  $CONDA_BIN

Set MINIFORGE_ROOT to the directory containing bin/conda, for example:
  MINIFORGE_ROOT=/path/to/miniforge3 bash cluster/setup_jupyter_env.sh
MSG
  exit 1
fi

eval "$("$CONDA_BIN" shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' already exists; reusing it."
else
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" ipykernel jupyter_client -y
fi

conda activate "$ENV_NAME"

python -m pip install -U pip ipykernel jupyter_client uv

python -m ipykernel install --user \
  --name "$KERNEL_NAME" \
  --display-name "$KERNEL_DISPLAY_NAME"

cat <<MSG

Jupyter kernel installed:
  name: $KERNEL_NAME
  display: $KERNEL_DISPLAY_NAME
  env: $ENV_NAME

In VS Code, connect to Imperial JupyterHub and select '$KERNEL_DISPLAY_NAME'.
MSG
