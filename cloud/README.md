# Lambda Cloud runner

This directory runs collection, evaluation, analysis, and preference training
on one Lambda Cloud GPU VM. The scripts detect `nvidia-smi` GPUs at startup and
use one collection shard and one private vLLM server per GPU.

Key defaults are defined in `cloud/env.sh`:

| Setting | Default |
| --- | ---: |
| GPUs and collection shards | All detected GPUs; one shard per GPU |
| Trajectory workers | 8 per shard |
| Cache-build workers | 50 (maximum 100) |
| Evaluation workers | 80 |
| Training processes | One per detected GPU |

Run `bash cloud/run.sh help` for the complete command list.

## Storage

Attach a Lambda filesystem when launching the VM; Lambda cannot attach one to
an already-running instance. A filesystem named `Debug-Depo` is mounted at
`/lambda/nfs/Debug-Depo`, which matches the tracked defaults. Change
`CLOUD_PERSISTENT_ROOT` on the VM if your name or mount point differs; update
`CLOUD_REMOTE_PERSISTENT_ROOT` locally so `pull` uses the same location.

The runner separates durable output from disposable VM-local data:

```text
/lambda/nfs/Debug-Depo/debug-depo-persistent/
  scratch/       # runs, logs, summaries, checkpoints, and packaged models
  sifs/          # SIF backup copied from the VM before termination
  tools/         # persistent uv environment

/home/ubuntu/debug-depo-ephemeral/
  cache/         # Hugging Face, uv, Torch, and Apptainer caches
  sifs/          # SWE-bench, SWE-smith, and vLLM images
  run-state/
  tmp/
```

VM-local data is lost when the instance is terminated. Verify that the two
roots are on different filesystems before starting work:

```bash
bash cloud/run.sh storage
```

`setup`, `preflight`, and execution workflows perform the same separation
check. Do not format or manually mount Lambda storage; attach it through the
Lambda console or launch API.

Official references: [filesystems](https://docs.lambda.ai/public-cloud/filesystems/),
[instance overview](https://docs.lambda.ai/public-cloud/on-demand/), and
[SSH access](https://docs.lambda.ai/public-cloud/on-demand/connecting-instance/).

## Setup and recovery

Follow the [VM setup and recovery runbook](RUNBOOK.md) for provisioning,
checkout transfer, Hugging Face and Docker Hub credentials, SIF restoration,
full setup, and replacement-VM resume commands.

For an existing checkout on a configured VM, the normal setup sequence is:

```bash
cd /home/ubuntu/debug-depo
cp -n cloud/local.env.example cloud/local.env
bash cluster/save_hf_token.sh
bash cloud/run.sh setup
bash cloud/run.sh storage
bash cloud/run.sh preflight
```

The top-level `./setup.sh` only creates local directories and fixes script
permissions. `cloud/run.sh setup` installs and validates the cloud runtime.

## Transfer results

Pull the persistent scratch tree after a run:

```bash
DRY_RUN=1 bash cloud/run.sh pull
bash cloud/run.sh pull
```

Results are copied to `scratch/cloud/` by default. Set
`LOCAL_CLOUD_SCRATCH_DIR` to change the destination. SIFs and other ephemeral
caches are not included in this transfer. Large model and checkpoint payloads
under `runs/*/experiments/` are also skipped by default while experiment
metadata is retained. Set `PULL_EXPERIMENT_MODELS=1` to include those payloads.

## Model and SIF caches

Retry only the AgentForge model prefetch after a transient download failure:

```bash
bash cloud/run.sh prefetch-model
```

Build two task images as a smoke check, or build all 500 SWE-bench Verified
images plus the tracked 5,500-task SWE-smith cache set:

```bash
bash cloud/run.sh build-cache smoke
bash cloud/run.sh build-cache full
```

The full build uses 50 concurrent pulls, disables Apptainer's shared
intermediate OCI cache, limits each `mksquashfs` conversion to two processors,
and requires 1,000 GiB free on the local cache filesystem by default. Override
these guards only after checking capacity:

```bash
CACHE_BUILD_MAX_WORKERS=40 MIN_FULL_CACHE_FREE_GIB=1200 \
  bash cloud/run.sh build-cache full
```

Before terminating a VM, copy the complete SIF tree to durable storage. The
setup command installs `rclone`; install it first and restore before full setup
on a replacement VM if you want setup to reuse the restored vLLM image.

```bash
# VM-local sifs/ -> persistent sifs/
bash cloud/run.sh sifs persist

# Persistent sifs/ -> VM-local sifs/
bash cloud/run.sh sifs restore
```

These commands include SWE-bench, SWE-smith, vLLM, and future SIF
subdirectories. They use 20 parallel transfers by default and do not delete or
remove source files. Override concurrency with `SIF_SYNC_TRANSFERS` and the
durable location with `PERSISTENT_SIF_DIR`:

```bash
SIF_SYNC_TRANSFERS=12 bash cloud/run.sh sifs persist
```

The vLLM SIF contains the serving runtime, not the AgentForge model weights.
By default, weights remain in the disposable Hugging Face cache and must be
prefetched on a new VM with `setup` or `prefetch-model`.

## Collection, evaluation, and analysis

Start with a bounded end-to-end smoke run. Each command uses one task per GPU;
the SWE-smith smoke produces one rollout at each of temperatures 0.6 and 0.7.

```bash
bash cloud/run.sh smoke verified
bash cloud/run.sh smoke swesmith
```

Preview any pipeline without starting vLLM or Apptainer:

```bash
DRY_RUN=1 bash cloud/run.sh pipeline verified
DRY_RUN=1 bash cloud/run.sh pipeline swesmith
```

Run collection, evaluation, and analysis together:

```bash
RUN_NAME=agentforge-verified-h100 \
  bash cloud/run.sh pipeline verified

RUN_NAME=swesmith-train-1000 \
  bash cloud/run.sh pipeline swesmith
```

The Verified default covers 500 tasks. The SWE-smith default uses the tracked
1,000-task training split and produces eight rollouts per task: four at 0.6 and
four at 0.7. Each GPU owns one shard, one vLLM server, and eight rollout worker
slots.

Resume or run stages separately with the same `RUN_NAME`:

```bash
RUN_NAME=agentforge-verified-h100 bash cloud/run.sh collect verified
RUN_NAME=agentforge-verified-h100 bash cloud/run.sh evaluate verified
RUN_NAME=agentforge-verified-h100 bash cloud/run.sh analyze verified
```

A compatible collection rerun reuses completed trajectories and retries
infrastructure failures. Use a new run name after changing the dataset, model,
or result-affecting settings.

On a replacement VM, resume only with matching manifest settings, especially
the shard count. The [recovery runbook](RUNBOOK.md#resume-swesmith-train-1000-r2)
contains the compatibility check, exact tmux command, and progress monitor for
`swesmith-train-1000-r2`.

The reduced SFT trajectory suite collects four rollouts for each of 1,000
training tasks—two at 0.6 and two at 0.7—then validates the SFT model once on
each fixed 100-, 200-, and 500-task split:

```bash
TRAIN_RUN_NAME=swesmith-train-1000-r2 \
  bash cloud/run.sh trajectory-suite
```

The suite attempts every stage and exits non-zero if any stage fails. It uses
100 evaluation workers by default.

## Preference training

Build and validate preference data from an evaluated SWE-smith training run:

```bash
RUN_NAME=swesmith-train-1000-r2 bash cloud/run.sh preference-data
RUN_NAME=swesmith-train-1000-r2 bash cloud/run.sh validate-data
```

Run the default data → DMPO → DEPO sequence on all configured GPUs:

```bash
RUN_NAME=swesmith-train-1000-r2 \
EXPERIMENT_ARM=dmpo-depo \
DMPO_TRIAL_NAME=gamma07 \
DEPO_TRIAL_NAME=alpha2 \
  bash cloud/run.sh train
```

Use `dmpo` or `depo` instead of `train` to run one training stage. Trial names
separate checkpoints, manifests, packages, and evaluations. See
[preference training](../docs/preference-training.md) and
[hyperparameter sweeps](../docs/hyperparameter-sweep.md) for model-specific
settings.

## Validation

`validate` runs one deterministic SWE-smith rollout for each ID in an explicit
task file. Supply either a Hugging Face model name or a packaged model path:

```bash
RUN_NAME=sft-validation-100 \
TASK_IDS_FILE=data/splits/swesmith_validation_100_instance_ids.txt \
AGENTFORGE_MODEL=Kwai-Klear/Klear-AgentForge-8B-SFT \
  bash cloud/run.sh validate

RUN_NAME=dmpo-validation-100 \
TASK_IDS_FILE=data/splits/swesmith_validation_100_instance_ids.txt \
MODEL_PATH="$CLOUD_PERSISTENT_ROOT/scratch/runs/<training-run>/experiments/dmpo/<trial>/model" \
  bash cloud/run.sh validate
```

`EXPECTED_TASKS` is inferred from non-empty lines in the task file. Use a
different `RUN_NAME` for every model and budget.

Evaluate a packaged preference model on all 500 SWE-bench Verified tasks:

```bash
TRAIN_RUN_NAME=swesmith-train-1000-r2 \
EXPERIMENT_ARM=dmpo \
DMPO_TRIAL_NAME=gamma07 \
  bash cloud/run.sh validate-model dmpo
```

Use `validate-model depo` with the matching `EXPERIMENT_ARM`,
`DMPO_TRIAL_NAME`, and `DEPO_TRIAL_NAME` for DEPO.

## Overrides

Put stable machine settings in ignored `cloud/local.env`; prefix one-off
experiment settings on the command line. Important overrides include:

- `GPU_IDS`, with matching `NUM_SHARDS` and `NUM_PROCESSES` when selecting a
  subset of GPUs.
- `ROLLOUT_WORKERS`, `EVAL_MAX_WORKERS`, `MAX_STEPS`, and `CONTEXT_LENGTH`.
- `CLOUD_SHARD_MAX_ATTEMPTS` (default `3`) retries a failed shard using its
  existing resumable output.
- `CLOUD_SHARD_STALL_TIMEOUT_SECONDS` (default `600`) restarts a shard when
  its streamed collector/event logs stop advancing; set `0` to disable it.
- `MINI_SWE_MODEL_TIMEOUT_SECONDS` optionally overrides LiteLLM's per-request
  timeout. It is operational rather than a sampling parameter, and must be
  lower than a nonzero shard stall timeout.
- `CLOUD_WATCHDOG_INTERVAL_SECONDS` (default `30`) controls health-check
  frequency, and `CLOUD_SHARD_RETRY_DELAY_SECONDS` (default `15`) controls the
  delay between attempts.
- `RUN_NAME`, `TASK_IDS_FILE`, `EXPECTED_TASKS`, and `AGENTFORGE_MODEL`.
- `VLLM_APPTAINER_SOURCE`, `VLLM_GPU_MEMORY_UTILIZATION`, and
  `APPTAINER_MKSQUASHFS_ARGS`.
- `STREAM_OUTPUT=1` (the cloud default) preserves partial mini-swe-agent output.
- `VLLM_LOG_REQUESTS=1` (the default) logs request metadata and up to
  `VLLM_MAX_LOG_LEN=2048` input characters. `RUST_BACKTRACE=1` preserves Rust
  panic backtraces. Set either diagnostic switch to `0` if needed.

Each attempt keeps a timestamped `vllm.attempt-*.log`; `vllm.log` points to the
latest attempt. The shared `cloud-logs/collect-*-shard-*.log` files contain only
the latest launcher run because historical server and rollout evidence remains
inside each shard.
`rollout_events.jsonl` and `active_rollouts.json` identify the instance, sample,
temperature, and seed active at a failure without copying full prompts into a
second log. A failed attempt snapshots that state as
`active_rollouts.failed-*.json` before retrying. Rollout sandboxes live in a
shard-specific temporary directory, so a failed attempt can remove only its own
orphaned sandboxes before resuming.

### Recovering a tail shard on all GPUs

Stop the normal collection before recovery. When all but one logical SWE-smith
shard have finished, distribute the remaining shard's unfinished rollout slots
round-robin across the configured GPUs:

```bash
RUN_NAME=swesmith-train-1000-r2 \
EXPECTED_TASKS=1000 \
TASK_IDS_FILE=data/splits/swesmith_train_1000_instance_ids.txt \
RUNS_PER_TEMPERATURE=2 \
TEMPERATURES=0.6:0.7 \
BASE_SEED=42 \
MAX_STEPS=200 \
CONTEXT_LENGTH=65536 \
TIMEOUT_SECONDS=21600 \
MINI_SWE_MODEL_TIMEOUT_SECONDS=1200 \
CLOUD_SHARD_STALL_TIMEOUT_SECONDS=1500 \
RECOVERY_WORKERS_PER_GPU=8 \
  bash cloud/run.sh recover-shard swesmith 2
```

The command preserves `NUM_SHARDS=8`, the original manifest, and all canonical
trajectory paths. Each GPU runs one vLLM replica and receives a disjoint portion
of the logical shard's slots; completed slots are reused. Only temporary
sandboxes and diagnostic logs are per replica. Once every replica succeeds, a
normal pass rebuilds the shard's canonical predictions and summaries. Rerunning
the command is safe, and `RECOVERY_GPU_IDS` can select fewer physical GPUs.
`MINI_SWE_MODEL_TIMEOUT_SECONDS` only changes how long LiteLLM waits for a
response; it does not change prompts or sampling. When set, the stall watchdog
must be disabled or set to a larger value so it cannot kill a valid in-flight
request first.

`GPU_IDS` defaults to `nvidia-smi` indices. CPU-only dry runs fall back to an
eight-GPU layout, so set GPU and shard values explicitly when inspecting a run
created with a different layout.
