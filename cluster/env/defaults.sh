#!/usr/bin/env bash

# Portable path defaults. cluster/env/load.sh first loads local.sh, then this
# file fills in any values the machine-specific configuration did not set.
export DEBUG_DEPO_ROOT="${DEBUG_DEPO_ROOT:-$PWD}"

if [[ -z "${DEBUG_DEPO_EPHEMERAL:-}" ]]; then
  if [[ -n "${RDS:-}" ]]; then
    export DEBUG_DEPO_EPHEMERAL="$RDS/ephemeral/debug-depo"
  elif [[ -n "${EPHEMERAL:-}" ]]; then
    export DEBUG_DEPO_EPHEMERAL="$EPHEMERAL/debug-depo"
  elif [[ -n "${SCRATCH:-}" ]]; then
    export DEBUG_DEPO_EPHEMERAL="$SCRATCH/debug-depo"
  else
    export DEBUG_DEPO_EPHEMERAL="$DEBUG_DEPO_ROOT/scratch"
  fi
fi

export DEBUG_DEPO_EPHEMERAL
export DEBUG_DEPO_SCRATCH="${DEBUG_DEPO_SCRATCH:-$DEBUG_DEPO_EPHEMERAL}"
export HF_HOME="${HF_HOME:-$DEBUG_DEPO_SCRATCH/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DEBUG_DEPO_SCRATCH/uv-cache}"
export TMPDIR="${TMPDIR:-$DEBUG_DEPO_SCRATCH/tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$DEBUG_DEPO_SCRATCH/apptainer-cache}"
export VLLM_IMAGE="${VLLM_IMAGE:-$DEBUG_DEPO_ROOT/cluster/apptainer/vllm-openai.sif}"
export SWEBENCH_APPTAINER_CACHE_DIR="${SWEBENCH_APPTAINER_CACHE_DIR:-$DEBUG_DEPO_SCRATCH/swebench_epoch_cache/apptainer-cache}"
export SWEBENCH_APPTAINER_SIF_DIR="${SWEBENCH_APPTAINER_SIF_DIR:-$DEBUG_DEPO_SCRATCH/swebench_epoch_cache/sifs}"
export SWESMITH_APPTAINER_CACHE_DIR="${SWESMITH_APPTAINER_CACHE_DIR:-$DEBUG_DEPO_SCRATCH/swesmith_cache/apptainer-cache}"
export SWESMITH_APPTAINER_SIF_DIR="${SWESMITH_APPTAINER_SIF_DIR:-$DEBUG_DEPO_SCRATCH/swesmith_cache/sifs}"
export SWESMITH_DATASET_REVISION="${SWESMITH_DATASET_REVISION:-77cab9055d42ab4a5c25c89a8f937096db13558e}"
