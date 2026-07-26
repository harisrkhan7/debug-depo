#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/preference_defaults.sh"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "Set RUN_ROOT to the preference-training run directory." >&2
  exit 2
fi
source "$ROOT_DIR/scripts/preference_trial_paths.sh"
OBJECTIVE="${PREFERENCE_OBJECTIVE:-}"
case "$OBJECTIVE" in
  dmpo)
    OBJECTIVE_LABEL=DMPO
    BASE_MODEL="${BASE_MODEL:-$PREFERENCE_BASE_MODEL_DEFAULT}"
    ADAPTER_PATH="${ADAPTER_PATH:-$DMPO_TRAIN_OUTPUT_DIR/adapter}"
    PACKAGE_OUTPUT_DIR="$DMPO_MODEL_DIR"
    ;;
  depo)
    OBJECTIVE_LABEL=DEPO
    if [[ "$EXPERIMENT_ARM" == "depo" ]]; then
      default_depo_base="$PREFERENCE_BASE_MODEL_DEFAULT"
    elif [[ "$EXPERIMENT_ARM" == "dmpo-depo" ]]; then
      default_depo_base="$DMPO_MODEL_DIR"
    else
      echo "DEPO packaging requires EXPERIMENT_ARM=depo or dmpo-depo." >&2
      exit 2
    fi
    BASE_MODEL="${BASE_MODEL:-$default_depo_base}"
    ADAPTER_PATH="${ADAPTER_PATH:-$DEPO_TRAIN_OUTPUT_DIR/adapter}"
    PACKAGE_OUTPUT_DIR="$DEPO_MODEL_DIR"
    ;;
  *)
    echo "PREFERENCE_OBJECTIVE must be dmpo or depo." >&2
    exit 2
    ;;
esac

if [[ ! -f "$ADAPTER_PATH/adapter_config.json" ]]; then
  echo "LoRA adapter is missing or incomplete: $ADAPTER_PATH" >&2
  exit 2
fi
if [[ -f "$PACKAGE_OUTPUT_DIR/package_manifest.json" && -f "$PACKAGE_OUTPUT_DIR/config.json" ]]; then
  echo "Reusing complete $OBJECTIVE package: $PACKAGE_OUTPUT_DIR"
  exit 0
fi
if [[ -d "$PACKAGE_OUTPUT_DIR" ]] && find "$PACKAGE_OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "Preserving incomplete package and rebuilding atomically: $PACKAGE_OUTPUT_DIR"
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
args=(
  --base-model "$BASE_MODEL"
  --adapter-path "$ADAPTER_PATH"
  --output-dir "$PACKAGE_OUTPUT_DIR"
  --max-shard-size "${MAX_SHARD_SIZE:-5GB}"
)
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  args+=(--trust-remote-code)
fi
"$UV_BIN" run debug-depo-package-model "${args[@]}"

cat <<MSG
$OBJECTIVE_LABEL packaging complete.
Adapter: $ADAPTER_PATH
Standalone evaluation model: $PACKAGE_OUTPUT_DIR
MSG
