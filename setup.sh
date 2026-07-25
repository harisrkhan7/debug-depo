#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p \
  "$ROOT_DIR/cluster/apptainer" \
  "$ROOT_DIR/cluster/env" \
  "$ROOT_DIR/cluster/logs" \
  "$ROOT_DIR/cluster/pbs" \
  "$ROOT_DIR/cluster/slurm" \
  "$ROOT_DIR/data/processed/agentforge_swebench_verified" \
  "$ROOT_DIR/data/processed/swesmith_collection" \
  "$ROOT_DIR/data/raw" \
  "$ROOT_DIR/data/splits" \
  "$ROOT_DIR/results/swebench"

touch \
  "$ROOT_DIR/data/splits/.gitkeep" \
  "$ROOT_DIR/cluster/logs/.gitkeep" \
  "$ROOT_DIR/data/raw/.gitkeep" \
  "$ROOT_DIR/results/swebench/.gitkeep"

chmod +x "$ROOT_DIR"/scripts/*.sh
if compgen -G "$ROOT_DIR/cluster/*.sh" >/dev/null; then
  chmod +x "$ROOT_DIR"/cluster/*.sh
fi
if compgen -G "$ROOT_DIR/cluster/apptainer/*.sh" >/dev/null; then
  chmod +x "$ROOT_DIR"/cluster/apptainer/*.sh
fi

cat <<'MSG'
debug-depo is set up for the AgentForge SWE-bench Verified reproduction.

Local smoke:
  MOCK=1 LIMIT=1 scripts/collect_rollouts.sh

Real AgentForge rollout:
  AGENTFORGE_COMMAND='<your harness command using {task_json} and {output_dir}>' \
  LLM_BASE_URL=http://127.0.0.1:8000/v1 \
  LLM_API_KEY=local \
  scripts/collect_rollouts.sh

Official SWE-bench evaluation:
  scripts/evaluate_all.sh

SWE-smith smoke pipeline:
  DRY_RUN=1 cluster/submit_swesmith_smoke.sh
MSG
