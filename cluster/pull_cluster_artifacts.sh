#!/usr/bin/env bash
set -euo pipefail

# Pull the complete cluster runs tree back to this machine for local inspection.
#
# Examples:
#   bash cluster/pull_cluster_artifacts.sh
#   REMOTE_RUNS_DIR=/rds/.../debug-depo/runs bash cluster/pull_cluster_artifacts.sh
#   LOCAL_DIR=scratch/cluster-runs bash cluster/pull_cluster_artifacts.sh
#   PULL_EVAL=1 RUN_ID=cluster_smoke_astropy_12907 bash cluster/pull_cluster_artifacts.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT_DIR/cluster/env/local.sh" ]]; then
  # Only the machine-specific remote settings are needed here. The full
  # environment loader also creates local model and container cache directories.
  source "$ROOT_DIR/cluster/env/local.sh"
fi

if [[ -z "${REMOTE:-}" && -n "${REMOTE_USER:-}" && -n "${REMOTE_HOST:-}" ]]; then
  REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
fi
REMOTE="${REMOTE:-debug-depo-cluster}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"
SSH_BIN="${SSH_BIN:-ssh}"

if [[ -z "${REMOTE_RDS:-}" || -z "${REMOTE_HOME:-}" ]]; then
  if ! remote_paths="$("$SSH_BIN" "$REMOTE" '
    remote_home=${HOME:-}
    remote_rds=${RDS:-}
    if [ -z "$remote_rds" ] && [ "${remote_home##*/}" = home ]; then
      remote_rds=${remote_home%/home}
    fi
    printf "__DEBUG_DEPO_PATHS__%s|%s\n" "$remote_rds" "$remote_home"
  ')"; then
    echo "SSH path discovery failed for host '$REMOTE'. Check your SSH alias and connection." >&2
    exit 1
  fi

  # Ignore any login banner written to stdout and parse the marked final line.
  remote_paths="${remote_paths##*$'\n'}"
  remote_paths="${remote_paths#__DEBUG_DEPO_PATHS__}"
  IFS='|' read -r discovered_rds discovered_home <<< "$remote_paths"
  REMOTE_RDS="${REMOTE_RDS:-$discovered_rds}"
  REMOTE_HOME="${REMOTE_HOME:-$discovered_home}"
fi

if [[ -z "${REMOTE_RDS:-}" || -z "${REMOTE_HOME:-}" ]]; then
  echo "Could not determine RDS and HOME from SSH host '$REMOTE'." >&2
  echo "Set REMOTE_RDS and REMOTE_HOME explicitly in cluster/env/local.sh if needed." >&2
  exit 1
fi

REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-$REMOTE_HOME/debug-depo}"
REMOTE_RUNS_DIR="${REMOTE_RUNS_DIR:-${REMOTE_OUTPUT_DIR:-$REMOTE_RDS/ephemeral/debug-depo/runs}}"
REMOTE_RUNS_DIR="${REMOTE_RUNS_DIR%/}"
LOCAL_DIR="${LOCAL_DIR:-$ROOT_DIR/scratch/cluster-artifacts/runs}"

mkdir -p "$LOCAL_DIR"

rsync_args=(
  -az
  --human-readable
  --exclude ".ipynb_checkpoints"
  --exclude "__pycache__"
)

if "$RSYNC_BIN" --help 2>&1 | grep -q -- "--protect-args"; then
  rsync_args+=(--protect-args)
fi

if "$RSYNC_BIN" --help 2>&1 | grep -q -- "--info"; then
  rsync_args+=(--info=progress2)
else
  rsync_args+=(--progress)
fi

copy_remote_path() {
  local remote_path="$1"
  local local_name="$2"
  shift 2

  mkdir -p "$(dirname "$LOCAL_DIR/$local_name")"
  "$RSYNC_BIN" "${rsync_args[@]}" "$@" "$REMOTE:$remote_path" "$LOCAL_DIR/$local_name"
}

cat <<MSG
Pulling the complete cluster runs tree from:
  $REMOTE:$REMOTE_RUNS_DIR/

Into:
  $LOCAL_DIR/

Override REMOTE, REMOTE_RUNS_DIR, or LOCAL_DIR if needed. REMOTE_OUTPUT_DIR
remains supported as an alias for REMOTE_RUNS_DIR. REMOTE_USER plus REMOTE_HOST
remain supported together.
MSG

copy_remote_path "$REMOTE_RUNS_DIR/" ""

if [[ "${PULL_EVAL:-0}" == "1" ]]; then
  RUN_ID="${RUN_ID:-cluster_smoke_astropy_12907}"
  REMOTE_REPORT_DIR="${REMOTE_REPORT_DIR:-$REMOTE_REPO_DIR/results/swebench}"
  REMOTE_EVAL_LOG_DIR="${REMOTE_EVAL_LOG_DIR:-$REMOTE_RDS/ephemeral/debug-depo/swebench-eval/logs}"

  copy_remote_path "$REMOTE_REPORT_DIR/" "results/swebench/" \
    --include "/*${RUN_ID}*" \
    --exclude "*"

  copy_remote_path "$REMOTE_EVAL_LOG_DIR/" "swebench-eval/logs/" \
    --include "/$RUN_ID/***" \
    --exclude "*" || true
fi

{
  printf 'remote=%s\n' "$REMOTE"
  printf 'remote_runs_dir=%s\n' "$REMOTE_RUNS_DIR"
  printf 'remote_repo_dir=%s\n' "$REMOTE_REPO_DIR"
  printf 'local_dir=%s\n' "$LOCAL_DIR"
  printf 'pull_eval=%s\n' "${PULL_EVAL:-0}"
  if [[ "${PULL_EVAL:-0}" == "1" ]]; then
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'remote_report_dir=%s\n' "$REMOTE_REPORT_DIR"
    printf 'remote_eval_log_dir=%s\n' "$REMOTE_EVAL_LOG_DIR"
  fi
  printf 'pulled_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} > "$LOCAL_DIR/manifest.txt"

printf '\nArtifact bundle ready:\n  %s\n\n' "$LOCAL_DIR"
printf 'Useful files to inspect:\n'
find "$LOCAL_DIR" -maxdepth 8 -type f | sort
