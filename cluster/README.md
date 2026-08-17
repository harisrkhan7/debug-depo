# PBS cluster runner

This directory schedules the project workflows on a PBS cluster. The research
logic is shared with the local and cloud paths; cluster scripts provide job
arrays, resource requests, environment setup, and `afterok` dependencies.
Run all commands below from the repository root.

Use the submission wrappers as the public interface:

| Workflow | Preview command |
| --- | --- |
| SWE-bench Verified smoke | `DRY_RUN=1 cluster/submit_verified_smoke.sh` |
| SWE-bench Verified full | `DRY_RUN=1 cluster/submit_verified_full.sh` |
| SWE-smith smoke | `DRY_RUN=1 cluster/submit_swesmith_smoke.sh` |
| SWE-smith pilot | `DRY_RUN=1 cluster/submit_swesmith_pilot.sh` |
| SWE-smith training split | `DRY_RUN=1 cluster/submit_swesmith_full.sh` |
| Shared task-image cache | `DRY_RUN=1 cluster/submit_apptainer_cache_full.sh` |
| Preference data/training | `DRY_RUN=1 cluster/submit_preference_training.sh` |

The wrappers submit collection, evaluation, and analysis in order. Low-level
`cluster/pbs/*.pbs` and `cluster/run_*_job.sh` files are implementation
details and should not normally be submitted directly.

## Storage

Cluster outputs and rebuildable caches default to `$DEBUG_DEPO_SCRATCH`.
`cluster/env/defaults.sh` chooses, in order:

1. `$RDS/ephemeral/debug-depo`;
2. `$EPHEMERAL/debug-depo`;
3. `$SCRATCH/debug-depo`;
4. `scratch/` inside the checkout.

On Imperial RCS, ephemeral storage is not backed up, has a per-user quota, and
deletes files after 30 days. Pull important results promptly. Current storage
policy is documented in the
[Imperial RCS data-management guide](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/getting-started/data-management-on-hpc/).

Before a cache build or full collection:

```bash
source cluster/env/load.sh
printf 'scratch: %s\n' "$DEBUG_DEPO_SCRATCH"
df -h "$DEBUG_DEPO_SCRATCH"
```

Keep the checkout, ignored machine configuration, Hugging Face token, and vLLM
SIF in persistent home storage. Keep model caches, task SIFs, sandboxes, and run
artifacts in ephemeral storage.

## Initial setup

Configure an SSH alias such as `debug-depo-cluster`, then create the ignored
local configuration:

```bash
cp -n cluster/env/local.sh.example cluster/env/local.sh
DRY_RUN=1 bash cluster/sync_to_cx3.sh
bash cluster/sync_to_cx3.sh
```

On the cluster, the checkout must be visible from login and compute nodes. The
supported scheduled-job environment uses Miniforge and the project `.venv`:

```bash
cd ~/debug-depo
bash cluster/setup_jupyter_env.sh
bash cluster/setup_rollout_env.sh
bash cluster/save_hf_token.sh
```

`setup_jupyter_env.sh` creates the `debug-depo` Conda environment used to
bootstrap jobs. `setup_rollout_env.sh` creates the checkout's `.venv` and
installs the pinned mini-swe-agent-plus and SWE-smith integrations. Run
`bash cluster/setup_training_env.sh` only when preference training is needed.

Scheduled jobs currently load Imperial's `tools/prod` and `miniforge/3`
modules. Set `MINIFORGE_ROOT` and `ENV_NAME` when using different Miniforge
locations or environment names. Site-specific extra modules belong in ignored
`cluster/env/modules.sh`.

## Model runtime

The baseline model and revision are pinned in `cluster/env/defaults.sh`. Build
the matching vLLM SIF once:

```bash
module load tools/prod
source cluster/env/load.sh
apptainer pull "$VLLM_IMAGE" "$VLLM_APPTAINER_SOURCE"
```

The default source is `docker://vllm/vllm-openai:v0.11.0`. Populate the shared
Hugging Face cache before launching an array so jobs do not all start with the
same model download:

```bash
bash cluster/prefetch_model.sh
```

The token helper stores the gated-model credential outside the repository with
mode `0600`. Do not put tokens in PBS variables, notebooks, or tracked files.

## Task-image caches

Collection and evaluation share persistent Apptainer SIF caches. Start with one
image from each task family:

```bash
DRY_RUN=1 cluster/submit_apptainer_cache_smoke.sh
cluster/submit_apptainer_cache_smoke.sh
```

The full helper defaults to all 500 SWE-bench Verified tasks and the active
5,700-task SWE-smith cache membership:

```bash
DRY_RUN=1 cluster/submit_apptainer_cache_full.sh
cluster/submit_apptainer_cache_full.sh
```

The SWE-smith file is
`data/splits/swesmith_cache_5700_instance_ids.txt`, the union of the tracked
training, screening, and confirmatory memberships. Override
`SWESMITH_TASK_IDS_FILE` for another run. Cache builds are resumable and skip
complete SIFs.

After building the cache, verify the confirmatory task refs:

```bash
source cluster/env/load.sh
PYTHONPATH=src .venv/bin/python scripts/preflight_swesmith_branches.py \
  --missing-output /tmp/swesmith-missing-branches.txt
```

## Smoke workflows

Run both end-to-end smoke chains before a full submission:

```bash
DRY_RUN=1 cluster/submit_verified_smoke.sh
cluster/submit_verified_smoke.sh

DRY_RUN=1 cluster/submit_swesmith_smoke.sh
cluster/submit_swesmith_smoke.sh
```

Each chain submits collection, then evaluation, then analysis. PBS logs and
artifacts are kept under:

```text
$DEBUG_DEPO_SCRATCH/runs/<run-name>/
  cluster-logs/
  rollouts/ or collection/
  merged/
  evaluation/
  analysis/
```

Use a unique `RUN_NAME` for an independent run. Resubmitting the same compatible
run name reuses completed trajectories and retries incomplete infrastructure
slots. A manifest mismatch requires a new name or deliberate `OVERWRITE=1`.

Unlike the cloud supervisor, PBS jobs do not automatically restart a stalled
shard. If an array element fails, inspect its shard and vLLM logs, correct the
cause, and resubmit the same wrapper with the same result-affecting settings.
Dependent jobs remain blocked because the wrappers use `afterok`.

## Full workflows

Run all SWE-bench Verified tasks:

```bash
DRY_RUN=1 cluster/submit_verified_full.sh
cluster/submit_verified_full.sh
```

Run the tracked 5,000-task SWE-smith training membership:

```bash
DRY_RUN=1 cluster/submit_swesmith_full.sh
cluster/submit_swesmith_full.sh
```

The SWE-smith wrapper collects eight rollouts per task: four at temperature 0.6
and four at 0.7. For the balanced confirmatory membership:

```bash
RUN_NAME=swesmith-validation-500 \
TASK_IDS_FILE=data/splits/swesmith_validation_confirmatory_balanced_500_instance_ids.txt \
EXPECTED_TASKS=500 \
DRY_RUN=1 \
  cluster/submit_swesmith_full.sh
```

Keep `SPLIT=train`; validation is a tracked local membership within the
upstream SWE-smith `train` split. Set `TASK_IDS_FILE` and `EXPECTED_TASKS`
together for every custom subset.

Use `SUBMIT_EVAL=0` to stop after collection or `SUBMIT_ANALYSIS=0` to omit
the final analysis job. Dataset, model, decoding, and shard settings are
forwarded through the full dependency chain.

## Preference training

Install the optional training dependencies, then build immutable preference
data:

```bash
bash cluster/setup_training_env.sh
RUN_NAME=<evaluated-swesmith-run> DRY_RUN=1 \
  cluster/submit_preference_data.sh
```

Preview DMPO followed by DEPO:

```bash
RUN_NAME=<evaluated-swesmith-run> \
EXPERIMENT_ARM=dmpo-depo \
DMPO_TRIAL_NAME=<dmpo-trial> \
DEPO_TRIAL_NAME=<depo-trial> \
PREFERENCE_DATA_MODE=reuse \
DRY_RUN=1 \
  cluster/submit_preference_training.sh
```

Use `EXPERIMENT_ARM=dmpo` or `depo` for one branch. Trial names isolate
manifests, checkpoints, packages, and evaluations. See
[preference training](../docs/preference-training.md) and
[hyperparameter sweeps](../docs/hyperparameter-sweep.md) for experiment
settings.

## Transfer and configuration

Pull run artifacts and cache summaries to `scratch/cluster-artifacts/`:

```bash
bash cluster/pull_cluster_artifacts.sh
```

Experiment model payloads are excluded by default; set
`PULL_EXPERIMENT_MODELS=1` only when needed. Configure remote aliases and path
overrides in ignored `cluster/env/local.sh`.

Common overrides include:

- `DEBUG_DEPO_SCRATCH`, `VLLM_IMAGE`, and `VLLM_APPTAINER_SOURCE`;
- `RUN_NAME`, `TASK_IDS_FILE`, and the matching expected task count;
- `NUM_SHARDS`, `ROLLOUT_WORKERS`, and `EVAL_MAX_WORKERS`;
- `AGENTFORGE_MODEL`, `MAX_STEPS`, `CONTEXT_LENGTH`, and sampling values;
- `STREAM_OUTPUT=1`, the maintained default for recoverable partial output;
- `VLLM_LOG_REQUESTS=1` and `VLLM_MAX_LOG_LEN=2048` for request diagnostics.

Resource requests and walltimes live only in `cluster/pbs/*.pbs`; review those
templates against current site policy before submitting. For workflow and
dataset details, use the [SWE-bench guide](../docs/swebench.md),
[SWE-smith guide](../docs/swesmith.md), and
[split provenance](../data/splits/README.md).
