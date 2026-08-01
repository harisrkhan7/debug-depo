# HyperStack runner

This directory provides the same high-level workflows as `cluster/`, but runs
them directly on one HyperStack H100 80 GB SXM5 x8 VM rather than submitting
PBS jobs. It targets the Canada `n3-H100-SXM5x8` flavor: 192 vCPUs, 1,800 GB
RAM, eight 80 GB H100 GPUs, and 32,000 GB of ephemeral storage. `preflight`
prints the detected inventory. The scripts themselves remain compatible with
other HyperStack hosts that expose at least eight GPUs.

The tracked defaults target:

| Resource | Default |
| --- | ---: |
| GPU model | H100 80 GB SXM5 |
| GPUs | 8 |
| Collection shards | 8 (one vLLM server per GPU) |
| Trajectory workers | 8 per shard (64 total slots) |
| Cache-build workers | 50 (maximum: 100) |
| Preference-training processes | 8 |
| Evaluation workers | 80 |

## Storage and hibernation

The standard HyperStack H100 SXM5 x8 flavor separates a 100 GB persistent root
disk from the 32 TB ephemeral disk. HyperStack clears the ephemeral disk when
the VM is hibernated or deleted. The root disk survives hibernation, but is
deleted with the VM.

The scripts deliberately split rebuildable working data from durable experiment
artifacts:

```text
/root/debug-depo-persistent/
  scratch/
    cache-builds/
    runs/
  tools/

/ephemeral/debug-depo/
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
analyses, checkpoints, and packaged models. The ephemeral tree holds
rebuildable model/package caches, Apptainer OCI caches, all SIFs, runtime state,
and temporary files. A run directory under
`/root/debug-depo-persistent/scratch/runs/<run-name>/` contains its rollouts or
collection, merged predictions, evaluation, analysis, preference data,
checkpoints, and packaged models.

HyperStack mounts the flavor's temporary disk at `/ephemeral` by default. Set
`HYPERSTACK_EPHEMERAL_ROOT` only if the VM was provisioned with a custom
ephemeral mount. Its contents must be rebuilt after hibernation or deletion.
Do not put `DEBUG_DEPO_SCRATCH` there because run artifacts must survive.

For full runs, create a sufficiently large HyperStack Shared Storage Volume
(SSV), attach it, and mount it at `/root/debug-depo-persistent`. This keeps the
run-directory layout on storage that survives VM deletion and reattachment,
while the large rebuildable image cache uses local ephemeral capacity.

After attaching a new empty SSV, identify its stable `/dev/disk/by-id/...`
device. The preparation command refuses to format anything unless the device
has no filesystem and `FORMAT_EMPTY_DEVICE=1` is explicit:

```bash
ls -l /dev/disk/by-id/

FORMAT_EMPTY_DEVICE=1 \
  sudo bash hyperstack/prepare_volume.sh \
  /dev/disk/by-id/<attached-volume>
```

For a volume that already has a filesystem, omit `FORMAT_EMPTY_DEVICE=1`. The
script mounts it and records its UUID in `/etc/fstab`.

HyperStack storage references:

- [VM storage and hibernation](https://docs.hyperstack.cloud/docs/virtual-machines/virtual-machine-features/)
- [Hibernation behavior](https://docs.hyperstack.cloud/docs/virtual-machines/hibernation/)
- [Creating a persistent volume](https://docs.hyperstack.cloud/docs/storage/volumes/creating-a-volume/)

## Initial setup

Deploy an Ubuntu NVIDIA CUDA image, then clone the repository somewhere on the
root disk (for example `/root/debug-depo`). A Docker daemon is not required:
Apptainer runs vLLM, task environments, and evaluation environments.

```bash
cd /root/debug-depo
cp hyperstack/local.env.example hyperstack/local.env

bash hyperstack/run.sh setup
bash cluster/save_hf_token.sh
bash hyperstack/run.sh preflight
```

`setup` installs Apptainer from its official Ubuntu PPA, creates a persistent
uv tool environment, installs the rollout/evaluation/training extras, installs
the pinned mini-swe-agent-plus and SWE-smith checkouts, and converts the vLLM
OCI image into an ephemeral SIF with `apptainer pull`. The NVIDIA driver is
supplied by the selected HyperStack CUDA image.

The Hugging Face token remains outside the repository at
`/root/.config/debug-depo/hf_token`. Runs, model packages, and checkpoints
remain under `HYPERSTACK_PERSISTENT_ROOT`; caches and SIFs are ephemeral.

Run long commands inside `tmux` so an SSH disconnect does not terminate the
foreground controller:

```bash
tmux new -s debug-depo
```

## Copying the checkout to and from HyperStack

On the local machine, define an SSH alias for the VM:

```sshconfig
Host debug-depo-hyperstack
  HostName <hyperstack-public-ip>
  User root
  IdentityFile ~/.ssh/<private-key>
```

Then copy the example configuration and preview the upload:

```bash
cp hyperstack/local.env.example hyperstack/local.env
DRY_RUN=1 bash hyperstack/run.sh push
bash hyperstack/run.sh push
```

`push` mirrors the behavior of `cluster/sync_to_cx3.sh`. It sends the checkout
to `/root/debug-depo` by default, but excludes Git metadata, virtual
environments, caches, results, scratch data, external checkouts, and the
machine-local `hyperstack/local.env`. Existing remote-only files are preserved
unless `DELETE=1` is explicitly set.

After a run, preview and then pull the entire persistent scratch tree:

```bash
DRY_RUN=1 bash hyperstack/run.sh pull
bash hyperstack/run.sh pull
```

The local result is `scratch/hyperstack/`. It includes cache-build summaries,
runs, trajectories, merged predictions, evaluation reports, analyses, logs,
checkpoints, and packaged models. Rebuildable caches, SIFs, runtime state, and
temporary files are not copied. Each successful pull records its source in
`scratch/hyperstack/_pull_manifest.txt`. Set
`LOCAL_HYPERSTACK_SCRATCH_DIR` to override the local destination.

## Cache

Start with a two-image smoke build:

```bash
bash hyperstack/run.sh build-cache smoke
```

Build the complete Verified cache plus the union of the tracked 5,000 training
and 500 validation SWE-smith tasks:

```bash
bash hyperstack/run.sh build-cache full
```

The full command uses 50 concurrent pulls by default, disables Apptainer's
shared intermediate OCI layer cache, and caps each `mksquashfs` conversion at
two processors through `APPTAINER_MKSQUASHFS_ARGS`. Disabling the intermediate
cache prevents concurrent pulls from corrupting its OCI metadata; completed
SIFs remain cached in the family SIF directories. Override concurrency only
after measuring:

```bash
CACHE_BUILD_MAX_WORKERS=50 \
  bash hyperstack/run.sh build-cache full
```

Completed SIFs are reused until ephemeral storage is cleared. The full build
has a conservative 1,000 GiB free-space guard on the ephemeral cache
filesystem; set `MIN_FULL_CACHE_FREE_GIB` higher for your flavor sizing.
Collections of 100 or more tasks also check for 500 GiB free by default; the
initial 1,000-task Hyperstack run therefore requires 500 GiB free. A 5,000-task
override checks for 1,000 GiB. These are early safety floors, not estimates of
final disk usage.

## Short smoke jobs

The smoke commands exercise collection, prediction merging, Apptainer
evaluation, and analysis without committing to a full run:

```bash
# Eight Verified tasks: one task per GPU/shard, eight trajectories total.
bash hyperstack/run.sh smoke verified

# Eight SWE-smith tasks, sampled once at 0.6 and once at 0.7:
# one task per GPU/shard, 16 trajectories total.
bash hyperstack/run.sh smoke swesmith
```

Both use all eight GPUs, eight shards, the configured 8-worker shard pools,
and a default ceiling of 20 agent steps. The evaluation stage uses eight
workers. The first smoke run may still need to download task images that were
not covered by `build-cache smoke`.

Preview either smoke without starting vLLM or Apptainer:

```bash
DRY_RUN=1 bash hyperstack/run.sh smoke verified
DRY_RUN=1 bash hyperstack/run.sh smoke swesmith
```

Override `RUN_NAME`, `EXPECTED_TASKS`, `MAX_STEPS`, or the timeouts when a
larger diagnostic run is useful. Keep `EXPECTED_TASKS` at least 8 because the
HyperStack collection wrapper intentionally requires all eight shards.

## Trajectory collection, evaluation, and analysis

Preview commands without starting servers:

```bash
DRY_RUN=1 bash hyperstack/run.sh pipeline verified
DRY_RUN=1 bash hyperstack/run.sh pipeline swesmith
```

Run SWE-bench Verified collection, evaluation, and analysis:

```bash
RUN_NAME=agentforge-verified-h100 \
  bash hyperstack/run.sh pipeline verified
```

Run the initial 1,000-task SWE-smith training collection. This creates eight
trajectories per task (four at each of temperatures 0.6 and 0.7), for 8,000
trajectories in total, evaluates all sample slots, and analyzes the result:

```bash
RUN_NAME=swesmith-train-1000 \
  bash hyperstack/run.sh pipeline swesmith
```

Each collection uses eight shards and all eight GPUs. Every shard owns one GPU,
one private vLLM server, and 8 trajectory workers. vLLM and mini-swe task
environments both run through Apptainer and reuse the ephemeral SIF cache.

Stages may also be run or resumed separately:

```bash
RUN_NAME=agentforge-verified-h100 \
  bash hyperstack/run.sh collect verified

RUN_NAME=agentforge-verified-h100 \
  bash hyperstack/run.sh evaluate verified

RUN_NAME=agentforge-verified-h100 \
  bash hyperstack/run.sh analyze verified
```

Rerunning collection with the same compatible configuration reuses completed
trajectories and retries infrastructure failures. Use a new `RUN_NAME` when
changing data, model, or result-affecting settings.

## DMPO and DEPO

Build and validate both immutable preference datasets after the SWE-smith
training run has been evaluated:

```bash
RUN_NAME=swesmith-train-1000 \
  bash hyperstack/run.sh preference-data

RUN_NAME=swesmith-train-1000 \
  bash hyperstack/run.sh validate-data
```

Train and package DMPO on all eight GPUs:

```bash
RUN_NAME=swesmith-train-1000 \
EXPERIMENT_ARM=dmpo \
DMPO_TRIAL_NAME=gamma07 \
  bash hyperstack/run.sh dmpo
```

Train DEPO from the selected packaged DMPO model:

```bash
RUN_NAME=swesmith-train-1000 \
EXPERIMENT_ARM=dmpo-depo \
DMPO_TRIAL_NAME=gamma07 \
DEPO_TRIAL_NAME=alpha2 \
  bash hyperstack/run.sh depo
```

The complete default data → DMPO → DEPO sequence is:

```bash
RUN_NAME=swesmith-train-1000 \
EXPERIMENT_ARM=dmpo-depo \
DMPO_TRIAL_NAME=gamma07 \
DEPO_TRIAL_NAME=alpha2 \
  bash hyperstack/run.sh train
```

All training commands default to `NUM_PROCESSES=8`, so Accelerate uses every
H100. Checkpoints and packaged models remain in the persistent run root.

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
  bash hyperstack/run.sh train
```

Shared overrides include `MAX_LENGTH`, `MAX_TRAIN_ROWS`,
`PER_DEVICE_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `EPOCHS`, and
`SAVE_STEPS`. Use new `DMPO_TRIAL_NAME` and `DEPO_TRIAL_NAME` values for each
configuration. Completed preference data are reused, while each trial keeps
separate checkpoints, manifests, packages, and evaluation paths.

## Validation

Run the repository-disjoint 500-task SWE-smith validation pipeline with the
same eight-shard, eight-worker-per-shard layout:

```bash
bash hyperstack/run.sh validate
```

Evaluate a packaged DMPO or DEPO model on the 500-task SWE-bench Verified set:

```bash
TRAIN_RUN_NAME=swesmith-train-1000 \
EXPERIMENT_ARM=dmpo \
DMPO_TRIAL_NAME=gamma07 \
  bash hyperstack/run.sh validate-model dmpo

TRAIN_RUN_NAME=swesmith-train-1000 \
EXPERIMENT_ARM=dmpo-depo \
DMPO_TRIAL_NAME=gamma07 \
DEPO_TRIAL_NAME=alpha2 \
  bash hyperstack/run.sh validate-model depo
```

## Useful overrides

Put stable machine-specific overrides in ignored `hyperstack/local.env`.
One-off experiment settings can be placed before a command:

```bash
RUN_NAME=verified-smoke-longer MAX_STEPS=40 \
  bash hyperstack/run.sh smoke verified

VLLM_APPTAINER_SOURCE=docker://vllm/vllm-openai:<tested-version> \
VLLM_GPU_MEMORY_UTILIZATION=0.85 \
  bash hyperstack/run.sh collect verified
```

`NUM_SHARDS=8`, eight `GPU_IDS`, `ROLLOUT_WORKERS=8`, and
`NUM_PROCESSES=8` are enforced by the relevant H100 workflows. Evaluation does
not require GPUs and uses the requested ephemeral task-image cache.

The `docker://` prefix above is an Apptainer OCI transport URI. It tells
`apptainer pull` where to obtain the image and does not invoke or require the
Docker CLI or daemon.
