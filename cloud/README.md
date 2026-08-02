# Lambda Cloud runner

This directory runs the same high-level workflows as `cluster/` directly on a
single Lambda Cloud GPU VM. At startup the scripts query `nvidia-smi`, use every
reported GPU, and create one collection shard and one private vLLM server per
GPU. Four- and eight-GPU hosts use the same commands.

The tracked defaults target:

| Resource | Default |
| --- | ---: |
| GPU model | H100 80 GB SXM5 |
| GPUs | Detected with `nvidia-smi` (typically 4 or 8) |
| Collection shards | One per detected GPU |
| Trajectory workers | 8 per shard |
| Cache-build workers | 50 (maximum: 100) |
| Preference-training processes | One per detected GPU |
| Evaluation workers | 80 |

## Storage

Attach a Lambda filesystem named `debug-depo` while launching the instance.
Lambda mounts it at `/lambda/nfs/debug-depo`; it cannot be attached after the
instance has launched. Durable experiment artifacts live there. Rebuildable
caches and SIFs use the instance's large local root volume, which is destroyed
when the instance is terminated.

Filesystem mount paths are case-sensitive. If the filesystem is named
`Debug-Depo`, override both `CLOUD_PERSISTENT_ROOT` and
`CLOUD_REMOTE_PERSISTENT_ROOT` with paths beginning `/lambda/nfs/Debug-Depo`.

The scripts deliberately split rebuildable working data from durable experiment
artifacts:

```text
/lambda/nfs/debug-depo/debug-depo-persistent/
  scratch/
    cache-builds/
    runs/
  tools/

/home/ubuntu/debug-depo-ephemeral/
  cache/
    apptainer/
    huggingface/
    swebench/
    swesmith/
    torch/
    uv/
  sifs/
    swebench/
    swesmith/
    vllm/
  run-state/
  tmp/
```

The persistent tree holds trajectories, predictions, evaluation reports,
analyses, checkpoints, and packaged models. The local tree holds
rebuildable model/package caches, Apptainer OCI caches, all SIFs, runtime state,
and temporary files. A run directory under
`CLOUD_PERSISTENT_ROOT/scratch/runs/<run-name>/` contains its rollouts or
collection, merged predictions, evaluation, analysis, preference data,
checkpoints, and packaged models.

Verify that durable and rebuildable paths resolve to separate filesystems:

```bash
bash cloud/run.sh storage
```

The check is read-only. Do not format or manually mount Lambda storage; attach
the filesystem in the Lambda console or launch API. `setup`, `preflight`, and
full workflows also reject configurations that put durable runs and local
caches on the same filesystem. Lambda filesystems mounted with the current
`virtiofs` driver are accepted, as are legacy `nfs` and `nfs4` mounts.

Lambda references:

- [On-Demand Cloud overview](https://docs.lambda.ai/public-cloud/on-demand/)
- [Filesystems](https://docs.lambda.ai/public-cloud/filesystems/)
- [Connecting over SSH](https://docs.lambda.ai/public-cloud/on-demand/connecting-instance/)

## Initial setup

Use Lambda Stack 22.04 or GPU Base 22.04, attach the `debug-depo` filesystem,
and connect as `ubuntu`. Lambda supplies the NVIDIA driver and CUDA stack.
Apptainer runs vLLM, task environments, and evaluation environments; the
workflow does not use the Docker daemon.

```bash
cd /home/ubuntu/debug-depo
cp cloud/local.env.example cloud/local.env

bash cluster/save_hf_token.sh
bash cloud/run.sh setup
bash cloud/run.sh storage
bash cloud/run.sh preflight
```

`setup` installs Apptainer from its official Ubuntu PPA, creates a persistent
uv tool environment, installs the rollout/evaluation/training extras, installs
the pinned mini-swe-agent-plus and SWE-smith checkouts, converts the vLLM OCI
image into a local SIF with `apptainer pull`, and serially downloads the gated
AgentForge model into the shared local Hugging Face cache. Save the token before
running setup. The default vLLM image is pinned to `v0.11.0` rather than
following a moving `latest` tag.

The Hugging Face token remains outside the repository at
`$HOME/.config/debug-depo/hf_token`. Runs, model packages, and checkpoints
remain under `CLOUD_PERSISTENT_ROOT`; caches and SIFs stay on local VM storage.

Run long commands inside `tmux` so an SSH disconnect does not terminate the
foreground controller:

```bash
tmux new -s debug-depo
```

## Copying the checkout to and from Lambda Cloud

On the local machine, define an SSH alias for the VM:

```sshconfig
Host debug-depo-cloud
  HostName <lambda-instance-ip>
  User ubuntu
  IdentityFile ~/.ssh/<private-key>
```

Then copy the example configuration and preview the upload:

```bash
cp cloud/local.env.example cloud/local.env
DRY_RUN=1 bash cloud/run.sh push
bash cloud/run.sh push
```

`push` mirrors the behavior of `cluster/sync_to_cx3.sh`. It sends the checkout
to `/home/ubuntu/debug-depo` by default, but excludes Git metadata, virtual
environments, caches, results, scratch data, external checkouts, and the
machine-local `cloud/local.env`. Existing remote-only files are preserved
unless `DELETE=1` is explicitly set.

Because `cloud/local.env` is intentionally excluded, create it separately on
the Lambda VM after the first push. The tracked example already contains the
standard Lambda paths. Legacy `HYPERSTACK_*` overrides remain accepted for old
automation, but new configuration should use `CLOUD_*` names.

After a run, preview and then pull the entire persistent scratch tree:

```bash
DRY_RUN=1 bash cloud/run.sh pull
bash cloud/run.sh pull
```

The local result is `scratch/cloud/`. It includes cache-build summaries,
runs, trajectories, merged predictions, evaluation reports, analyses, logs,
checkpoints, and packaged models. Rebuildable caches, SIFs, runtime state, and
temporary files are not copied. Each successful pull records its source in
`scratch/cloud/_pull_manifest.txt`. Set
`LOCAL_CLOUD_SCRATCH_DIR` to override the local destination.

## Cache

Setup normally prefetches the model once before any GPU shards start. Retry
only that step, for example after a transient Hugging Face failure, with:

```bash
bash cloud/run.sh prefetch-model
```

Start with a two-image smoke build:

```bash
bash cloud/run.sh build-cache smoke
```

Build the complete Verified cache plus the union of the tracked 5,000 training
and 500 validation SWE-smith tasks:

```bash
bash cloud/run.sh build-cache full
```

The full command uses 50 concurrent pulls by default, disables Apptainer's
shared intermediate OCI layer cache, and caps each `mksquashfs` conversion at
two processors through `APPTAINER_MKSQUASHFS_ARGS`. Disabling the intermediate
cache prevents concurrent pulls from corrupting its OCI metadata; completed
SIFs remain cached in the family SIF directories. Override concurrency only
after measuring:

```bash
CACHE_BUILD_MAX_WORKERS=50 \
  bash cloud/run.sh build-cache full
```

Completed SIFs are reused until local VM storage is cleared. The full build
has a conservative 1,000 GiB free-space guard on the local cache
filesystem; set `MIN_FULL_CACHE_FREE_GIB` higher for your flavor sizing.
Collections of 100 or more tasks also check for 500 GiB free by default; the
initial 1,000-task cloud run therefore requires 500 GiB free. A 5,000-task
override checks for 1,000 GiB. These are early safety floors, not estimates of
final disk usage.

## Short smoke jobs

The smoke commands exercise collection, prediction merging, Apptainer
evaluation, and analysis without committing to a full run:

```bash
# One Verified task per detected GPU/shard.
bash cloud/run.sh smoke verified

# One SWE-smith task per GPU/shard, sampled once at 0.6 and once at 0.7.
bash cloud/run.sh smoke swesmith
```

Both use every detected GPU, one shard and one smoke task per GPU, the
configured 8-worker shard pools, and a default ceiling of 20 agent steps. The
evaluation worker count matches the GPU count. The first smoke run may still
need to download task images that were not covered by `build-cache smoke`.

Preview either smoke without starting vLLM or Apptainer:

```bash
DRY_RUN=1 bash cloud/run.sh smoke verified
DRY_RUN=1 bash cloud/run.sh smoke swesmith
```

Override `RUN_NAME`, `EXPECTED_TASKS`, `MAX_STEPS`, or the timeouts when a
larger diagnostic run is useful. Keep `EXPECTED_TASKS` at least as large as
the detected GPU count because every collection shard must receive work.

## Trajectory collection, evaluation, and analysis

Preview commands without starting servers:

```bash
DRY_RUN=1 bash cloud/run.sh pipeline verified
DRY_RUN=1 bash cloud/run.sh pipeline swesmith
```

Run SWE-bench Verified collection, evaluation, and analysis:

```bash
RUN_NAME=agentforge-verified-h100 \
  bash cloud/run.sh pipeline verified
```

Run the initial 1,000-task SWE-smith training collection. This creates eight
trajectories per task (four at each of temperatures 0.6 and 0.7), for 8,000
trajectories in total, evaluates all sample slots, and analyzes the result:

```bash
RUN_NAME=swesmith-train-1000 \
  bash cloud/run.sh pipeline swesmith
```

Each collection uses every detected GPU. Every GPU owns one shard, one private
vLLM server, and 8 trajectory workers. vLLM and mini-swe task environments both
run through Apptainer and reuse the local SIF cache.

Stages may also be run or resumed separately:

```bash
RUN_NAME=agentforge-verified-h100 \
  bash cloud/run.sh collect verified

RUN_NAME=agentforge-verified-h100 \
  bash cloud/run.sh evaluate verified

RUN_NAME=agentforge-verified-h100 \
  bash cloud/run.sh analyze verified
```

Rerunning collection with the same compatible configuration reuses completed
trajectories and retries infrastructure failures. Use a new `RUN_NAME` when
changing data, model, or result-affecting settings.

Run the complete reduced SFT trajectory sequence—1,000 training tasks with two
rollouts at each of temperatures 0.6 and 0.7, followed by deterministic
validation on the fixed 100-, 200-, and 500-task budgets—with one command:

```bash
TRAIN_RUN_NAME=swesmith-train-1000-r2 \
  bash cloud/run.sh trajectory-suite
```

The stages run sequentially. If one fails, the wrapper records the failure and
continues with the remaining validation budgets, then exits non-zero after all
stages have been attempted. Evaluation uses 100 workers by default. Run the
command inside `tmux`; a compatible rerun resumes completed collection work.

## DMPO and DEPO

Build and validate both immutable preference datasets after the SWE-smith
training run has been evaluated:

```bash
RUN_NAME=swesmith-train-1000 \
  bash cloud/run.sh preference-data

RUN_NAME=swesmith-train-1000 \
  bash cloud/run.sh validate-data
```

Train and package DMPO on all detected GPUs:

```bash
RUN_NAME=swesmith-train-1000 \
EXPERIMENT_ARM=dmpo \
DMPO_TRIAL_NAME=gamma07 \
  bash cloud/run.sh dmpo
```

Train DEPO from the selected packaged DMPO model:

```bash
RUN_NAME=swesmith-train-1000 \
EXPERIMENT_ARM=dmpo-depo \
DMPO_TRIAL_NAME=gamma07 \
DEPO_TRIAL_NAME=alpha2 \
  bash cloud/run.sh depo
```

The complete default data → DMPO → DEPO sequence is:

```bash
RUN_NAME=swesmith-train-1000 \
EXPERIMENT_ARM=dmpo-depo \
DMPO_TRIAL_NAME=gamma07 \
DEPO_TRIAL_NAME=alpha2 \
  bash cloud/run.sh train
```

All training commands default `NUM_PROCESSES` to the detected GPU count, so
Accelerate uses every configured H100. Checkpoints and packaged models remain
in the persistent run root.

The combined command accepts separate settings for each stage:

```bash
RUN_NAME=swesmith-train-1000 \
EXPERIMENT_ARM=dmpo-depo \
DMPO_TRIAL_NAME=gamma09-lr5e7 \
DEPO_TRIAL_NAME=alpha2-lr1e5 \
DMPO_LEARNING_RATE=5e-7 \
DMPO_BETA=0.1 \
DMPO_GAMMA=0.9 \
DEPO_LEARNING_RATE=1e-5 \
DEPO_BETA=0.2 \
ALPHA_TOKENS=2 \
ALPHA_STEPS=2 \
DEPO_TOKEN_METRIC=completion_tokens \
  bash cloud/run.sh train
```

Shared overrides include `MAX_LENGTH`, `MAX_TRAIN_ROWS`,
`PER_DEVICE_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `EPOCHS`, and
`SAVE_STEPS`. Use new `DMPO_TRIAL_NAME` and `DEPO_TRIAL_NAME` values for each
configuration. Completed preference data are reused, while each trial keeps
separate checkpoints, manifests, packages, and evaluation paths.

## Validation

`validate` evaluates one model on an explicit SWE-smith task-ID file. It runs
each task exactly once at temperature 0, then scores and analyzes the complete
task matrix. It does not choose a validation membership implicitly. Give every
model and budget a distinct `RUN_NAME`; `EXPECTED_TASKS` is inferred from the
non-empty lines in `TASK_IDS_FILE` and is checked when supplied explicitly.

Evaluate a packaged DMPO model on the 100-task screening budget:

```bash
TRAIN_RUN_NAME=swesmith-train-1000-r2
TRIAL_NAME=g07-lr1e6-b01-ga16

RUN_NAME="validation-100-dmpo-$TRIAL_NAME" \
TASK_IDS_FILE=data/splits/swesmith_validation_100_instance_ids.txt \
MODEL_PATH="$CLOUD_PERSISTENT_ROOT/scratch/runs/$TRAIN_RUN_NAME/experiments/dmpo/$TRIAL_NAME/model" \
  bash cloud/run.sh validate
```

Use `AGENTFORGE_MODEL=Kwai-Klear/Klear-AgentForge-8B-SFT` instead of
`MODEL_PATH` to evaluate the SFT baseline. For 200 or 500 tasks, supply the
corresponding task-ID file and a new run name. `CONTEXT_LENGTH` defaults to
32,768 and `MAX_STEPS` to 200 for validation.

Evaluate a packaged DMPO or DEPO model on the 500-task SWE-bench Verified set:

```bash
TRAIN_RUN_NAME=swesmith-train-1000 \
EXPERIMENT_ARM=dmpo \
DMPO_TRIAL_NAME=gamma07 \
  bash cloud/run.sh validate-model dmpo

TRAIN_RUN_NAME=swesmith-train-1000 \
EXPERIMENT_ARM=dmpo-depo \
DMPO_TRIAL_NAME=gamma07 \
DEPO_TRIAL_NAME=alpha2 \
  bash cloud/run.sh validate-model depo
```

## Useful overrides

Put stable machine-specific overrides in ignored `cloud/local.env`.
One-off experiment settings can be placed before a command:

```bash
RUN_NAME=verified-smoke-longer MAX_STEPS=40 \
  bash cloud/run.sh smoke verified

VLLM_APPTAINER_SOURCE=docker://vllm/vllm-openai:<tested-version> \
VLLM_GPU_MEMORY_UTILIZATION=0.85 \
  bash cloud/run.sh collect verified
```

`GPU_IDS` defaults to the indices reported by `nvidia-smi`; `NUM_SHARDS` and
`NUM_PROCESSES` default to that list's length. `ROLLOUT_WORKERS=8` remains a
per-shard default. To select a GPU subset, set `GPU_IDS` and matching
`NUM_SHARDS`/`NUM_PROCESSES` values explicitly. Evaluation does not itself use
the GPUs, but `NUM_SHARDS` must match the run being evaluated. When
`nvidia-smi` is unavailable, offline dry-runs fall back to the original eight
GPU layout, so set `GPU_IDS` explicitly when inspecting a differently sharded
run on a CPU-only host.

The `docker://` prefix above is an Apptainer OCI transport URI. It tells
`apptainer pull` where to obtain the image and does not invoke or require the
Docker CLI or daemon.
