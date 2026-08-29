# Lambda Cloud runner

This directory runs collection, evaluation, analysis, and preference training
on Lambda Cloud. Run all commands below from the repository root.
The runner detects `nvidia-smi` GPUs and uses one collection shard and one
private vLLM server per GPU.

Key defaults from `cloud/env.sh` are:

| Setting | Default |
| --- | ---: |
| GPUs and collection shards | All detected GPUs; one shard per GPU |
| Trajectory workers | 8 per shard |
| Cache-build workers | 50 (maximum 100) |
| Evaluation workers | 100 |
| Threads per evaluation task | 1 |
| Training processes | One per detected GPU |

Run `bash cloud/run.sh help` for the complete command list. The baseline SFT
model is pinned to Hugging Face revision
`0da97e45dbbd44278bd55b878170ec369d2934fb`; override
`VLLM_MODEL_REVISION` only when deliberately testing another snapshot.

## Storage and setup

Attach a Lambda filesystem when launching the VM; it cannot be attached later.
The tracked defaults separate durable output from disposable VM-local data:

```text
/lambda/nfs/Debug-Depo/debug-depo-persistent/
  scratch/       # runs, logs, checkpoints, and packaged models
  sifs/          # SIF backup copied here before VM termination
  tools/         # persistent uv environment

/home/ubuntu/debug-depo-ephemeral/
  cache/         # model, uv, Torch, and Apptainer caches
  sifs/          # active SWE-bench, SWE-smith, and vLLM images
  run-state/
  tmp/
```

Set `CLOUD_PERSISTENT_ROOT` or `CLOUD_EPHEMERAL_ROOT` in ignored
`cloud/local.env` if these paths differ. Verify that they are on separate
filesystems:

```bash
bash cloud/run.sh storage
```

For provisioning, credentials, checkout transfer, SIF restoration, and
replacement-VM recovery, follow the [setup and recovery runbook](RUNBOOK.md).
For an existing checkout on a configured VM:

```bash
cd /home/ubuntu/debug-depo
cp -n cloud/local.env.example cloud/local.env
bash cluster/save_hf_token.sh
bash cloud/run.sh setup
bash cloud/run.sh storage
bash cloud/run.sh preflight
```

The top-level `./setup.sh` only creates local directories and fixes script
permissions; `cloud/run.sh setup` installs and validates the cloud runtime.

Lambda references: [filesystems](https://docs.lambda.ai/public-cloud/filesystems/)
and [SSH access](https://docs.lambda.ai/public-cloud/on-demand/connecting-instance/).

## Transfer and caches

Pull persistent results to local `scratch/cloud/`:

```bash
DRY_RUN=1 bash cloud/run.sh pull
bash cloud/run.sh pull
```

The pull excludes SIFs and large model/checkpoint payloads under
`runs/*/experiments/` while retaining experiment metadata. Use
`PULL_MODEL_ADAPTERS=true` for only LoRA/PEFT adapters, or
`PULL_EXPERIMENT_MODELS=1` for all experiment model payloads. Set
`LOCAL_CLOUD_SCRATCH_DIR` to change the local destination.

Build a two-image smoke cache or the full 500-task SWE-bench Verified and
5,700-task SWE-smith cache:

```bash
bash cloud/run.sh build-cache smoke
bash cloud/run.sh build-cache full
```

The full build requires 1,000 GiB free by default. After it completes, verify
the tracked confirmatory SWE-smith refs:

```bash
PYTHONPATH=src .venv/bin/python scripts/preflight_swesmith_branches.py \
  --missing-output /tmp/swesmith-missing-branches.txt
```

Retry only the baseline model download with `bash cloud/run.sh prefetch-model`.
Before terminating a VM, persist the SIF tree; restore it before setup on a
replacement VM:

```bash
bash cloud/run.sh sifs persist
bash cloud/run.sh sifs restore
```

These commands copy without deleting their source. Set `SIF_SYNC_TRANSFERS`
(default `20`) or `PERSISTENT_SIF_DIR` when needed. Model weights are not inside
the vLLM SIF and must be prefetched on each new VM.

## Collection and evaluation

Start with one task per GPU. The SWE-smith smoke collects one rollout at each
of temperatures 0.6 and 0.7:

```bash
bash cloud/run.sh smoke verified
bash cloud/run.sh smoke swesmith
```

Preview or run collection, evaluation, and analysis together:

```bash
DRY_RUN=1 bash cloud/run.sh pipeline verified
DRY_RUN=1 bash cloud/run.sh pipeline swesmith

RUN_NAME=agentforge-verified-cloud bash cloud/run.sh pipeline verified
RUN_NAME=swesmith-train-1000 bash cloud/run.sh pipeline swesmith
```

The Verified default covers all 500 tasks. A bare SWE-smith pipeline invocation
uses the tracked 1,000-task training split and the generic eight-rollout default:
four at 0.6 and four at 0.7. The completed `swesmith-train-1000-r2` preference
collection instead used `cloud/trajectory_suite.sh`, which explicitly selected
two runs at each temperature, or four trajectories per task. Resume or run
stages separately with the same `RUN_NAME`:

```bash
RUN_NAME=agentforge-verified-cloud bash cloud/run.sh collect verified
RUN_NAME=agentforge-verified-cloud bash cloud/run.sh evaluate verified
RUN_NAME=agentforge-verified-cloud bash cloud/run.sh analyze verified
```

A compatible rerun reuses completed trajectories and retries infrastructure
failures. Use a new run name after changing result-affecting settings. A
replacement VM must reproduce the existing manifest, especially its shard
count; see [the recovery runbook](RUNBOOK.md#resume-swesmith-train-1000-r2).

The reduced SFT suite collects four rollouts for each of 1,000 training tasks,
then validates the SFT model on the tracked 100-, 200-, and 500-task
`swesmith_validation_<budget>_instance_ids.txt` memberships:

```bash
TRAIN_RUN_NAME=swesmith-train-1000-r2 bash cloud/run.sh trajectory-suite
```

The 500-task membership used by this suite is the exclusion/reproduction set,
not `swesmith_validation_confirmatory_balanced_500_instance_ids.txt`. Run the
balanced confirmatory set explicitly with `validate` when required.

## Preference training

Build and validate preference data from an evaluated SWE-smith run:

```bash
RUN_NAME=swesmith-train-1000-r2 bash cloud/run.sh preference-data
RUN_NAME=swesmith-train-1000-r2 bash cloud/run.sh validate-data
```

Run data preparation followed by DMPO and DEPO training on all configured GPUs:

```bash
RUN_NAME=swesmith-train-1000-r2 \
EXPERIMENT_ARM=dmpo-depo \
DMPO_TRIAL_NAME=gamma07 \
DEPO_TRIAL_NAME=alpha2 \
  bash cloud/run.sh train
```

Use `dmpo` or `depo` instead of `train` for one stage. Keep distinct trial
names for independent outputs. See [preference training](../docs/preference-training.md)
and [hyperparameter sweeps](../docs/hyperparameter-sweep-light.md).

## Validation and final testing

`validate` runs one deterministic SWE-smith rollout per non-empty task ID. It
infers `EXPECTED_TASKS` from the file and accepts a Hugging Face model name or a
packaged model path:

```bash
RUN_NAME=sft-validation-100 \
TASK_IDS_FILE=data/splits/swesmith_validation_100_instance_ids.txt \
AGENTFORGE_MODEL=Kwai-Klear/Klear-AgentForge-8B-SFT \
  bash cloud/run.sh validate

source cloud/env.sh
MODEL_DIR="$DEBUG_DEPO_SCRATCH/runs/<training-run>/experiments/dmpo/<trial>/model"
RUN_NAME=dmpo-validation-100 \
TASK_IDS_FILE=data/splits/swesmith_validation_100_instance_ids.txt \
MODEL_PATH="$MODEL_DIR" \
  bash cloud/run.sh validate
```

Use a different `RUN_NAME` for every model and budget. To run the project's
pinned SFT, DMPO, and DEPO models sequentially on all 500 SWE-bench Verified
test tasks:

```bash
bash cloud/run.sh test
```

This command requires the packaged DMPO `g07-paper-informed` and DEPO
`total-balanced` models under the `swesmith-train-1000-r2/experiments` tree. It
uses temperature `0.0`, top-p `1.0`, 200 steps, and a 65,536-token context, and
resumes compatible output.

To test another model on Verified, use `pipeline verified`. For a packaged
model, set both model variables to the same path:

```bash
source cloud/env.sh
MODEL_DIR="$DEBUG_DEPO_SCRATCH/runs/<training-run>/experiments/dmpo/<trial>/model"
RUN_NAME=dmpo-evaluation-500 \
AGENTFORGE_MODEL="$MODEL_DIR" \
VLLM_MODEL="$MODEL_DIR" \
  bash cloud/run.sh pipeline verified
```

## Configuration and recovery

Put stable machine settings in ignored `cloud/local.env` and one-off experiment
settings on the command line. Common overrides are:

- `GPU_IDS`, with matching `NUM_SHARDS` and `NUM_PROCESSES` for a GPU subset.
- `ROLLOUT_WORKERS`, `EVAL_MAX_WORKERS`, `EVAL_THREADS_PER_TASK`, `MAX_STEPS`,
  and `CONTEXT_LENGTH`.
- `RUN_NAME`, `TASK_IDS_FILE`, `EXPECTED_TASKS`, and `AGENTFORGE_MODEL`.
- `CLOUD_SHARD_MAX_ATTEMPTS` (default `3`) and
  `CLOUD_SHARD_STALL_TIMEOUT_SECONDS` (default `1500`; `0` disables it).
- `MINI_SWE_MODEL_TIMEOUT_SECONDS` (default `1200`), which must be lower than a
  nonzero shard stall timeout.
- `VLLM_APPTAINER_SOURCE`, `VLLM_GPU_MEMORY_UTILIZATION`, and
  `APPTAINER_MKSQUASHFS_ARGS`.

Each collection attempt retains its vLLM log and active-rollout state inside
the shard directory. Stop normal collection before using
`bash cloud/run.sh recover-shard swesmith INDEX` to distribute one unfinished
logical shard across idle GPUs. Preserve the original manifest and shard count;
completed rollout slots are reused. Use `RECOVERY_GPU_IDS` to restrict physical
GPUs, and consult `bash cloud/run.sh help` plus the recovery runbook before
resuming on a replacement VM.
