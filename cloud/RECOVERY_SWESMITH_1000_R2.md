# Recover `swesmith-train-1000-r2`

This frozen recipe reproduces the resume-sensitive settings of the completed
1,000-task Lambda collection. For generic VM provisioning, use the
[Lambda setup runbook](RUNBOOK.md).

Confirm the persistent run and its manifest:

```bash
source cloud/env.sh
RUN_ROOT="$DEBUG_DEPO_SCRATCH/runs/swesmith-train-1000-r2"

test -f "$RUN_ROOT/collection/shard-0/collection_manifest.json"
jq '{
  num_shards,
  expected_tasks,
  runs_per_temperature,
  temperatures,
  base_seed,
  max_steps,
  context_length,
  timeout_seconds
}' "$RUN_ROOT/collection/shard-0/collection_manifest.json"

echo "Current GPU count: $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
```

The replacement VM must reproduce the manifest's result-affecting settings.
In particular, its `num_shards` must match `NUM_SHARDS`; this normally means
using the same GPU count. A larger VM may select the original GPU count, but a
smaller VM cannot resume the existing shard layout.

Start the compatible four-rollout-per-task pipeline:

```bash
tmux new-session -d -s swesmith-1000-r2 \
  'cd /home/ubuntu/debug-depo &&
   RUN_NAME=swesmith-train-1000-r2 \
   EXPECTED_TASKS=1000 \
   TASK_IDS_FILE=data/splits/swesmith_train_1000_instance_ids.txt \
   RUNS_PER_TEMPERATURE=2 \
   TEMPERATURES=0.6:0.7 \
   BASE_SEED=42 \
   MAX_STEPS=200 \
   CONTEXT_LENGTH=32768 \
   TIMEOUT_SECONDS=21600 \
   ROLLOUT_WORKERS=8 \
   EVAL_MAX_WORKERS=100 \
   LIMIT="" \
     bash cloud/run.sh pipeline swesmith'
```

`pipeline` resumes collection, then evaluates and analyzes it. Replace
`pipeline swesmith` with `collect swesmith` to stop after collection.

Monitor it from another shell:

```bash
tmux attach-session -t swesmith-1000-r2

source cloud/env.sh
scripts/check_swesmith_progress.sh \
  "$DEBUG_DEPO_SCRATCH/runs/swesmith-train-1000-r2" \
  --watch 30
```

Detach from tmux with `Ctrl-b`, then `d`.
