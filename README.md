# debug-depo

Utilities for reproducing the Klear AgentForge 8B SFT result on
SWE-bench Verified while keeping the project structure lightweight enough for
Mac smoke tests and HPC batch runs.

The repository also includes a separate SWE-smith trajectory-data pipeline.
It collects four independent trajectories at each of two temperatures per
training task, evaluates all 8 samples, and produces rollout-level and
task-level analysis.

## Target

The reproduction target is the Klear-AgentForge-8B-SFT SWE-bench Verified
score reported in the Klear-AgentForge paper:

- Dataset: `princeton-nlp/SWE-bench_Verified`
- Dataset revision: `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`
- Split: `test`
- Instances: 500
- Harness: `mini-swe-agent-plus` / AgentForge-style SWE scaffold
- Budget: 200 steps, 64k context
- Reported 8B SFT score: `38.2%` (`191 / 500` resolved)

The code here does not vendor the Kwai/Klear harness. Kwai/Klear publish the
model and report in `Kwai-Klear/Klear-AgentForge`, and the runnable SWE scaffold
is their official `mini-swe-agent-plus` repository. This project wraps that
installed package, records per-instance trajectories, emits the official
SWE-bench prediction JSONL format, and launches the official SWE-bench evaluator.

## Evaluation defaults and future dataset roles

The existing workflow is the evaluation setup. Its defaults remain:

```text
DATASET=princeton-nlp/SWE-bench_Verified
SWEBENCH_DATASET_REVISION=c104f840cc67f8b6eec6f759ebc8b2693d585d4a
SPLIT=test
EXPECTED_COUNT=500
```

The revision is the exact 500-row snapshot matched against the saved task
records from the completed `agentforge-verified-full-20260715` run. The default
is therefore pinned even when the environment variable is omitted.

Collection selects tasks from the configured dataset and split, then passes one
validated local task at a time to mini-swe-agent-plus. The cluster submission
scripts pass the same dataset values to dependent evaluation jobs. The run
directory structure and Verified output paths are unchanged.

Future training and validation runs can use the same pipeline by setting a
different dataset, split, immutable instance-ID file, expected count, and run
name:

```bash
DATASET=org/dataset \
SWEBENCH_DATASET_REVISION=<immutable-dataset-commit> \
SPLIT=train \
TASK_IDS_FILE=data/splits/train_instance_ids.txt \
EXPECTED_COUNT=1000 \
NUM_SHARDS=10 \
RUN_NAME=training-rollouts-v1 \
DRY_RUN=1 \
cluster/submit_verified_full.sh
```

Use a distinct `RUN_NAME` for every dataset role. `RUN_ID` defaults to the run
name with dashes replaced by underscores and can still be set explicitly.
`NUM_SHARDS` cannot exceed `EXPECTED_COUNT`, ensuring that every scheduled
collection shard receives at least one task. A non-default remote dataset is
left unpinned unless `SWEBENCH_DATASET_REVISION` is supplied.

Evaluation summaries compare against the paper's 38.2% (191/500) target only
for the complete default SWE-bench Verified `test` evaluation with the
AgentForge model. Other datasets, splits, models, and partial evaluations retain
their measured results but leave the target-comparison fields empty.

## Install

```bash
uv sync --extra dev
```

For official SWE-bench scoring on a Docker-capable machine:

```bash
uv sync --extra dev --extra swebench
```

Run the idempotent local scaffold step whenever you want the expected folders
and executable bits refreshed:

```bash
./setup.sh
```

## Mac Smoke Run

This does not call a model or Docker. It verifies task selection, trajectory
writing, prediction JSONL writing, and summaries.

```bash
MOCK=1 LIMIT=1 scripts/collect_rollouts.sh
```

A notebook version of this flow lives at
`notebooks/local_agentforge_swebench_smoke.ipynb`.

Use gold patches only when you specifically want to test the SWE-bench evaluator
on a tiny subset:

```bash
MOCK=1 MOCK_PATCH=gold LIMIT=1 scripts/collect_rollouts.sh
scripts/evaluate_all.sh
```

## Official mini-swe-agent-plus Harness

Install the official Kwai/Klear harness into this repo's `.venv`:

```bash
scripts/install_mini_swe_agent_plus.sh
```

Then point it at an OpenAI-compatible model server. For local Mac smoke tests,
start with a public MLX coder model:

```bash
MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit scripts/serve_local_llm.sh
```

The official harness expects a LiteLLM-style model name. For an MLX/OpenAI
server, use the `hosted_vllm/...` provider prefix while preserving the full
served model id:

Before launching the SWE harness, verify that the local server can both list
models and generate a tiny chat completion:

```bash
PYTHONPATH=src python3 -m debug_depo.check_local_llm \
  --base-url http://127.0.0.1:8000/v1 \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
```

A notebook version of this check lives at
`notebooks/local_llm_server_check.ipynb`.

```bash
HARNESS=mini-swe-agent-plus \
AGENTFORGE_MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
MINI_SWE_MODEL=hosted_vllm/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
LLM_BASE_URL=http://127.0.0.1:8000/v1 \
LLM_API_KEY=local \
LIMIT=1 \
scripts/collect_rollouts.sh
```

For the exact paper-target model, run the server with
`MODEL=Kwai-Klear/Klear-AgentForge-8B-SFT` and use
`AGENTFORGE_MODEL=Kwai-Klear/Klear-AgentForge-8B-SFT`
`MINI_SWE_MODEL=hosted_vllm/Kwai-Klear/Klear-AgentForge-8B-SFT`.

The wrapper automatically uses the package's
`swebench_add_edit_tool.yaml` config and writes the one-line vLLM server file
expected by the upstream script.

## Custom AgentForge Rollout

If you need a different harness entrypoint, set `AGENTFORGE_COMMAND`. The
command is rendered once per instance and can use these template fields:

`{instance_id}`, `{repo}`, `{task_json}`, `{output_dir}`, `{model}`,
`{llm_base_url}`, `{llm_api_key}`, `{agentforge_repo}`, `{max_steps}`,
`{context_length}`, `{temperature}`, `{top_p}`, `{seed}`.

The command should write one of the following under `{output_dir}`:

- `prediction.patch`, `prediction.diff`, `patch.patch`, or `patch.diff`
- a JSON file named `prediction.json`, `result.json`, `trajectory.json`,
  `rollout.json`, or `output.json` containing `model_patch`, `patch`, `diff`,
  `output_patch`, or `git_diff`
- a JSON object with one of those keys on stdout

Persisted trajectory metadata redacts `llm_api_key` and any occurrence of that
value in the rendered command. A custom harness should still avoid printing
credentials to stdout or stderr because those streams are retained as rollout
logs.

Example shape for a custom command:

```bash
AGENTFORGE_REPO=/path/to/Klear-AgentForge \
AGENTFORGE_COMMAND='python -m agentforge.swebench.run \
  --task-json {task_json} \
  --output-dir {output_dir} \
  --model {model} \
  --base-url {llm_base_url} \
  --api-key {llm_api_key} \
  --max-steps {max_steps} \
  --context-length {context_length}' \
LLM_BASE_URL=http://127.0.0.1:8000/v1 \
LLM_API_KEY=local \
scripts/collect_rollouts.sh
```

Outputs land in:

```text
data/processed/agentforge_swebench_verified/
  predictions.jsonl
  run_config.json
  summary.json
  trajectories/<instance_id>/
```

For new runs, `run_config.json` is also the resume manifest. It records the
dataset revision, exact selected instance IDs and row-content hash,
mini-swe-agent version/Git state, and result-affecting run settings. Reusing the
directory with an incompatible manifest fails before any trajectory is reused;
choose a new output directory or set `OVERWRITE=1` deliberately. Older run
directories remain readable and analysable, but their missing historical
dependency provenance is not backfilled.

## Official Evaluation

```bash
scripts/evaluate_all.sh
```

On Apple Silicon, the evaluator automatically passes `--namespace ''` so
SWE-bench builds images locally instead of pulling Linux images. For the full
benchmark, use an x86_64 Linux machine or HPC node with Docker or a compatible
container setup.

## HPC Shards

Rollouts are shardable:

```bash
NUM_SHARDS=10 SHARD_INDEX=0 OUTPUT_DIR=$SCRATCH/agentforge/shard-0 scripts/collect_rollouts.sh
```

Merge shard predictions before scoring:

```bash
scripts/merge_predictions.sh $SCRATCH/agentforge/shard-*/predictions.jsonl
```

PBS templates live in `cluster/pbs/`. They are intentionally plain so you can
adapt modules, queues, walltimes, and scratch paths to the cluster you end up
using.

## SWE-smith trajectory collection

Install mini-swe-agent-plus plus the official SWE-smith package and repository
profiles:

```bash
scripts/install_mini_swe_agent_plus.sh
scripts/install_swesmith.sh
```

The tracked modes provide a bounded pilot as well as an explicit full-dataset
submission:

```text
dataset:          SWE-bench/SWE-smith-py (train)
pilot tasks:      30
temperatures:     0.6, 0.7
runs/temperature: 4
total runs/task:  8
base seed:        42
```

Each task/sample pair receives a stable derived seed. Collection writes one
prediction file per sample slot so duplicate task IDs are never collapsed:

The upstream dataset exposes only `train`. Reproducible, repository-disjoint
local train and validation memberships are tracked in
`data/splits/train_instance_ids.txt` (45,809 tasks) and
`data/splits/validation_instance_ids.txt` (5,099 tasks). Their pinned revision,
policy, repository membership, and hashes are recorded in
`data/splits/swesmith_py_split_manifest.json`. Derived, repository-covering
samples provide 5,000 training tasks and 500 validation tasks, with their exact
union available for cache building. The full sampling policy is documented in
`data/splits/README.md`. Regenerate the derived samples from the tracked parent
memberships with:

```bash
python -m debug_depo.prepare_swesmith_splits --subsets-only
```

Use the 500-task validation sample without changing the upstream split name:

```bash
RUN_NAME=swesmith-validation-500 \
SPLIT=train \
TASK_IDS_FILE=data/splits/swesmith_validation_500_instance_ids.txt \
EXPECTED_TASKS=500 \
NUM_SHARDS=100 \
DRY_RUN=1 \
cluster/submit_swesmith_full.sh
```

Collection reuses the existing mini-swe-agent-plus integration. Tasks are
selected once from `SWE-bench/SWE-smith-py`, then each rollout passes its local
task JSON through the same `run_agentforge_instance` path as the SWE-bench
Verified collector. Before the agent starts, the generated mini-swe startup
command checks out that task's SWE-smith branch in the repository-level image.
Use the standard `swebench` runner with Docker or the default `singularity`
runner; mini-swe-agent-plus's `pool_way` runner does not execute the required
startup command.

```text
<run-root>/
  cluster-logs/
  collection/shard-*/
    collection_manifest.json
    samples/sample-0/ ... sample-7/
      predictions.jsonl
      summary.json
      trajectories/<instance-id>/
  merged/sample-0/ ... sample-7/
  evaluation/sample-0/ ... sample-7/
  analysis/
    rollouts.csv
    tasks.csv
    summary.json
```

For a local artifact smoke test with no agent or task container:

```bash
MOCK=1 MOCK_PATCH=gold LIMIT=2 scripts/collect_swesmith.sh
```

SWE-smith stores the bug-producing patch rather than a solution patch. Gold
mock predictions are therefore marked for reverse application by the
SWE-smith evaluator, matching the upstream harness.

Preview the complete cluster dependency chains:

```bash
DRY_RUN=1 cluster/submit_swesmith_smoke.sh
DRY_RUN=1 cluster/submit_swesmith_pilot.sh
DRY_RUN=1 cluster/submit_swesmith_full.sh
```

Then submit the two-task smoke run, the default 30-task pilot, or the tracked
5,000-task training sample:

```bash
cluster/submit_swesmith_smoke.sh
cluster/submit_swesmith_pilot.sh
cluster/submit_swesmith_full.sh
```

All three commands submit `collect → eval → analyse` jobs with `afterok`
dependencies. Override `TASK_LIMIT`, `EXPECTED_TASKS`, `NUM_SHARDS`, or provide
an immutable `TASK_IDS_FILE` for a different subset. In bounded modes,
`EXPECTED_TASKS` defaults to `TASK_LIMIT`; full mode omits the limit and checks
the selected dataset size before starting rollouts. Evaluation also refuses to
start when any sample is incomplete. PBS stdout and stderr are stored in the
run's `cluster-logs/` directory in ephemeral storage, not in the repository.

The default dataset is pinned to Hugging Face revision
`77cab9055d42ab4a5c25c89a8f937096db13558e`. Override
`SWESMITH_DATASET_REVISION` deliberately when moving to another snapshot.
`scripts/install_mini_swe_agent_plus.sh` and `scripts/install_swesmith.sh`
likewise check out fixed Git commits; their revision environment variables are
`MINI_SWE_AGENT_PLUS_REVISION` and `SWESMITH_REVISION`. Collection manifests
record the dataset revision, installed package commits, and a hash of any
deterministic installer patch applied on top.

Agent terminations caused by the configured step/context limits are retained as
model outcomes with empty patches and proceed to evaluation. Infrastructure and
unexpected execution failures fail the shard, but rerunning the same shard
retries only those failed sample slots.

Each trajectory subprocess receives the single task JSON already selected by
the collector. The adapter still uses the pinned official mini-swe runner, but
does not reload and materialise the complete Hugging Face split before applying
its one-instance filter.

The full wrapper defaults to
`data/splits/swesmith_train_5000_instance_ids.txt`, 5,000 expected tasks, 50
shards, and six concurrent trajectories per shard. Set `TASK_IDS_FILE` and
`EXPECTED_TASKS` together to select a different subset.

SWE-smith's upstream evaluator is Docker-oriented. The cluster path here uses
Apptainer for the same repository images while retaining SWE-smith's official
profile-specific test commands and grading logic. Generated SIFs and caches
live under `SWESMITH_APPTAINER_SIF_DIR` and
`SWESMITH_APPTAINER_CACHE_DIR`. Collection and evaluation pull each distinct
image once under a filesystem lock and reuse the persistent SIF. Collection
builds each agent's writable sandbox from that local SIF rather than repeatedly
building from a remote `docker://` URI.

Prebuild the shared task-image caches before collection:

```bash
# One Verified image plus one SWE-smith image.
cluster/submit_apptainer_cache_smoke.sh

# All 500 Verified images plus the unique images required by this SWE-smith split.
cluster/submit_apptainer_cache_full.sh
```

The full job is resumable and deduplicates SWE-smith tasks by repository image.
See `cluster/README.md` for dry-run, validation-split, directory, worker, and
summary options.

Use `notebooks/cluster_agentforge_swesmith.ipynb` for an interactive cluster
workflow covering preflight, dry-run submission, optional smoke collection,
evaluation, and analysis. Use `notebooks/inspect_swesmith_collection.ipynb` to
inspect an existing run's shard coverage, temperatures, raw agent messages and
patches, per-instance evaluation reports, and per-temperature pass@1…4 plus
the explicitly labelled mixed-temperature pool metrics.

## DMPO and DEPO training data

Build post-training data only after collection and evaluation are complete.
Both builders read the raw mini-swe message histories, per-call token usage,
and evaluation outcomes directly from the SWE-smith run. Provider response
payloads are removed, while the multi-turn assistant actions and environment
observations are retained.

DMPO is pairwise. For every task, a resolved trajectory is preferred to an
unresolved trajectory. Between two resolved trajectories, the cheaper one is
preferred only when the configured cost ratio is met. Unresolved-versus-
unresolved cost pairs are excluded by default so that cheap failure is never a
positive training signal.

```bash
RUN_ROOT=scratch/cluster-artifacts/runs/swesmith-pilot-20260719 \
  scripts/build_dmpo_pairs.sh
```

The default cost is accumulated API `total_tokens` (prompt plus completion),
which matches the debugging-cost objective. Set
`TOKEN_METRIC=completion_tokens` for a generated-token-only objective,
`MIN_COST_RATIO=1.25` to require 25% savings, or
`MAX_PAIRS_PER_TASK=8` to cap correlated pairs.

DEPO is not a pairwise objective. It is an efficiency-aware KTO objective over
independently labelled desirable and undesirable trajectories. Resolved
rollouts are desirable and scored non-resolved rollouts are undesirable.
Each record contains total steps, completion and total tokens per step, and
their inverses. A DEPO trainer can therefore use either the paper's generated
token bonus or the billed-token variant:

```text
b(trajectory) =
  alpha1 * inverse_{completion|total}_tokens_per_step
  + alpha2 * inverse_steps
```

```bash
RUN_ROOT=scratch/cluster-artifacts/runs/swesmith-pilot-20260719 \
  scripts/build_depo_data.sh
```

Outputs are written under:

```text
<run-root>/preference-data/
  dmpo/pairs.jsonl
  dmpo/summary.json
  depo/trajectories.jsonl
  depo/desirable.jsonl
  depo/undesirable.jsonl
  depo/summary.json
```

On PBS, preview and submit both CPU-only data jobs with:

```bash
RUN_NAME=swesmith-pilot-20260719 DRY_RUN=1 \
  cluster/submit_preference_data.sh
RUN_NAME=swesmith-pilot-20260719 \
  cluster/submit_preference_data.sh
```

The submission wrapper schedules `DMPO data -> DEPO data` with an `afterok`
dependency. The datasets themselves are independent; the dependency gives a
clear operational order and prevents a partial handoff. For model training,
fine-tune the Klear AgentForge checkpoint with DMPO first, then use the
resulting DMPO checkpoint as both the DEPO initialization and its frozen
reference policy. Keep a held-out repository-disjoint validation set and
select checkpoints on success rate subject to total-token cost; optimizing
token cost without the success constraint can reward premature termination.
