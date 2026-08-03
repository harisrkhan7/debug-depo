# Lambda VM setup and recovery runbook

Use this runbook for a new or replacement Lambda VM. See the
[cloud README](README.md) for storage design, workflows, and configuration.

## Provision and transfer

Launch a Lambda Stack or GPU Base image with Python 3.10–3.12 and attach the
Lambda filesystem at creation time. Ensure the local disk can hold the SIF tree
and model caches.

Configure an SSH alias locally:

```sshconfig
Host debug-depo-cloud
  HostName <lambda-instance-ip>
  User ubuntu
  IdentityFile ~/.ssh/<private-key>
```

Push the current working tree:

```bash
cd <local-debug-depo-checkout>
cp -n cloud/local.env.example cloud/local.env
DRY_RUN=1 bash cloud/run.sh push
bash cloud/run.sh push
```

The transfer includes uncommitted files but excludes Git metadata,
environments, outputs, caches, external checkouts, and `cloud/local.env`. Use
`CLOUD_REMOTE=ubuntu@<lambda-instance-ip>` when no SSH alias is configured.

## Configure the VM

On the VM:

```bash
cd /home/ubuntu/debug-depo
cp -n cloud/local.env.example cloud/local.env
bash ./setup.sh

findmnt /lambda/nfs/Debug-Depo
nvidia-smi
bash cluster/save_hf_token.sh
```

`./setup.sh` only creates repository directories and fixes permissions. Edit
`cloud/local.env` if the persistent mount, local root, image, or GPU selection
differs from the defaults. The Hugging Face helper stores a mode-`600` token
file outside the checkout.

Apptainer pulls OCI images; no Docker daemon is required. To restore SIFs or
authenticate to Docker Hub before full setup, bootstrap the required tools:

```bash
sudo apt-get update
sudo apt-get install -y rclone software-properties-common

if ! command -v apptainer >/dev/null 2>&1; then
  sudo add-apt-repository -y ppa:apptainer/ppa
  sudo apt-get update
  sudo apt-get install -y apptainer
fi

read -r -p 'Docker Hub username: ' DOCKERHUB_USERNAME
apptainer registry login --username "$DOCKERHUB_USERNAME" docker://docker.io
unset DOCKERHUB_USERNAME
```

Enter the Docker access token at the password prompt. Never place either token
in the repository, `cloud/local.env`, or a command-line argument.

## Restore and install

Verify storage separation and restore the SIF backup before setup:

```bash
bash cloud/run.sh storage
source cloud/env.sh
test -d "$CLOUD_PERSISTENT_ROOT/sifs"
bash cloud/run.sh sifs restore
```

`restore` copies SIFs to ephemeral storage without deleting the durable
backup. Skip it when no backup exists.

Install and validate the runtime:

```bash
bash cloud/run.sh setup
bash cloud/run.sh storage
bash cloud/run.sh preflight
```

Cloud setup installs dependencies and pinned external checkouts, validates or
builds the vLLM SIF, and prefetches the AgentForge model. It is safe to rerun
after a transient failure.

## Resume `swesmith-train-1000-r2`

Confirm the persistent run and its resume-sensitive settings:

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
In particular, the existing `num_shards` must match `NUM_SHARDS`; this normally
means using the same GPU count. A larger VM may select the original GPU count,
but a smaller VM cannot resume the existing shard layout.

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
   CONTEXT_LENGTH=65536 \
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
