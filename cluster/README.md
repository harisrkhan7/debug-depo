# Cluster Notes

The runner is split into three phases:

1. Rollout shards call the AgentForge harness and write SWE-bench predictions.
2. Evaluation merges predictions and runs the official SWE-bench Docker harness.
3. Analysis joins rollout and evaluation artifacts and writes run summaries.

The tracked defaults define the current evaluation setup: SWE-bench Verified at
revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`, the `test` split, and 500
expected predictions. That revision exactly matches all 500 saved task records
from the completed `agentforge-verified-full-20260715` run.
`cluster/submit_verified_smoke.sh` and `cluster/submit_verified_full.sh` forward
`DATASET`, `SWEBENCH_DATASET_REVISION`, `SPLIT`, `TASK_IDS_FILE`,
`EXPECTED_COUNT`, model settings, and decoding settings to their dependent PBS
jobs. This prevents collection and Apptainer evaluation from silently selecting
different dataset snapshots. Existing run directory segmentation is unchanged.

For a future training or validation collection, use a distinct `RUN_NAME` and
set all four data-selection values together:

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

Inspect the dry run before submission. The expected count must match the unique
instances selected across all shards. Submission fails before queuing jobs when
`NUM_SHARDS` exceeds `EXPECTED_COUNT`, preventing empty collection shards.

Evaluation summaries include the repository's pinned 38.2% (191/500) target
comparison only for the complete default SWE-bench Verified `test` evaluation
with the AgentForge model. Other datasets, splits, models, and partial
evaluations report their measured results with empty target-comparison fields.

Important runtime split:

- vLLM model serving can run under Apptainer.
- Evaluation can run through this repo's Apptainer evaluator.
- `mini-swe-agent-plus` is still the rollout agent. Its default pool runner
  starts SWE-bench task environments with the `docker` CLI, but this repo can
  also call mini-swe-agent-plus's standard SWE-bench runner with
  `--environment-class singularity` for Apptainer-only clusters.

Edit the PBS headers for your site before submitting. In particular, check queue
names, GPU resource syntax, module names, scratch paths, and whether Docker is
available on compute nodes.

## Apptainer vLLM

Imperial RDS home counts against your quota. Large, rebuildable artifacts should
live in RDS ephemeral storage, which is unquotaed but deleted 30 days after
creation. Keep source code, secrets, and the manually built vLLM server SIF in
home; keep model caches, generated SWE-bench SIFs, and rollout scratch output
in ephemeral.

The tracked `cluster/env/defaults.sh` prefers:

```text
$RDS/ephemeral/debug-depo
```

when `RDS` is set. The vLLM server SIF stays in the repo checkout by default:

```bash
module load tools/prod
apptainer pull cluster/apptainer/vllm-openai.sif docker://vllm/vllm-openai:latest
```

The AgentForge model is gated on Hugging Face. After you have been granted
access on the model page, store a read token outside the repo:

```bash
bash cluster/save_hf_token.sh
```

This writes to `~/.config/debug-depo/hf_token` with `0600` permissions.
The vLLM and rollout helpers load that file automatically if `HF_TOKEN` is not
already set. Do not paste tokens into notebooks, PBS files, or tracked repo
files.

Set up the Python environment using the environment system available on your
cluster.

For Imperial JupyterHub / VS Code, the login-node Conda setup is scripted:

```bash
bash cluster/setup_jupyter_env.sh
```

It creates or reuses a `debug-depo` Conda environment and registers a Jupyter
kernel named `debug-depo`. The repo requires Python `>=3.10,<3.13`, so the
script creates a Python 3.11 Conda environment even if the cluster's default
Python module is newer. It installs the project and its `notebooks` optional
dependencies, including pandas, into that kernel. It does not install
`mini-swe-agent-plus`.

To copy this checkout to your CX3 home directory:

```bash
bash cluster/sync_to_cx3.sh
```

Keep the login name and host outside the repository by defining an SSH alias:

```sshconfig
Host debug-depo-cluster
    HostName cluster.example.edu
    User your-user
```

Copy `cluster/env/local.sh.example` to the ignored `cluster/env/local.sh` and
set `REMOTE` to that alias. Preview the transfer first with
`DRY_RUN=1 bash cluster/sync_to_cx3.sh`.

After syncing the repo to CX3, install the rollout dependencies and official
mini-swe-agent-plus harness:

```bash
cd ~/debug-depo
bash cluster/setup_rollout_env.sh
```

The checkout must live on storage visible from both the login and compute
nodes, such as your RDS home. PBS does not copy or load the repository into a
job. `qsub` records the submission directory as `PBS_O_WORKDIR`; each PBS file
changes back to that shared checkout before running anything.

Local `.venv` and `external/` directories are deliberately excluded by the
sync script, so rerun `cluster/setup_rollout_env.sh` on the cluster after
syncing any mini-swe integration change. It creates the checkout's `.venv`,
installs mini-swe-agent-plus there, and verifies the persistent SIF integration.

Every scheduled collection and evaluation job then runs this bootstrap:

1. `module load tools/prod`
2. `module load miniforge/3`
3. initialize Conda with `$HOME/miniforge3/bin/conda shell.bash hook`
4. activate `$HOME/miniforge3/envs/debug-depo`
5. source the ignored `cluster/env/modules.sh` for any extra site-specific additions
6. run project commands through `uv run`, which uses the checkout's `.venv`

The Conda environment therefore supplies `uv` and the login/kernel tooling;
the reproducible project dependencies and executable Python come from the
checkout's `.venv`. A scheduled job fails early with a setup instruction if
Conda, `uv`, the Apptainer executable supplied by `tools/prod`, or the project
`.venv` is missing.

For Imperial RCS-style Python modules plus a virtualenv:

```bash
module load tools/prod
module load Python
mkdir -p ~/venv
virtualenv ~/venv/debug-depo
source ~/venv/debug-depo/bin/activate
python -m pip install -U pip uv
uv sync --extra dev --extra swebench
scripts/install_mini_swe_agent_plus.sh
```

Only load `SciPy-bundle` if your cluster recommends it and it does not conflict
with the packages installed into the virtualenv.

For Miniforge/Conda, the module may be named `anaconda3`, `miniforge3`, or
similar; use `module avail` to find the site-specific name:

```bash
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n debug-depo python=3.12 -y
conda activate debug-depo
python -m pip install -U pip uv
uv sync --extra dev --extra swebench
scripts/install_mini_swe_agent_plus.sh
```

Copy `cluster/env/modules.sh.example` to the ignored `cluster/env/modules.sh`,
then edit it for your site's module names. The rollout
PBS file starts `cluster/apptainer/serve_vllm.sh` by default, waits for
`http://127.0.0.1:8000/v1`, and then runs `scripts/collect_rollouts.sh`.

On clusters without Docker, use the mini-swe-agent-plus Singularity backend.
This is still mini-swe-agent-plus; only the task container backend changes:

```bash
export HARNESS=mini-swe-agent-plus
export MINI_SWE_RUNNER=singularity
export MINI_SWE_ENVIRONMENT_CLASS=singularity
export MSWEA_SINGULARITY_EXECUTABLE=apptainer
export MINI_SWE_MODEL=hosted_vllm/Kwai-Klear/Klear-AgentForge-8B-SFT
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_API_KEY=local
LIMIT=1 MAX_STEPS=20 scripts/collect_rollouts.sh
```

The first Apptainer/Singularity task run may be slow because the collector
pulls the image into `SWEBENCH_APPTAINER_SIF_DIR` under a filesystem lock, then
mini-swe builds a writable sandbox from that local SIF under `TMPDIR`.
Every collection and evaluation mode uses this same image-keyed rule, so any
later run requesting the same image URI reuses the completed SIF. The setup
script verifies that mini-swe has this integration. The portable cluster
defaults put both the SIF and temporary sandbox in ephemeral scratch. The local
installer patch also pre-creates `/rds` inside that writable sandbox so
Imperial's automatic `/rds` bind has a destination. If your site auto-binds a
different top-level filesystem, set
`MSWEA_SINGULARITY_WRITABLE_BIND_DESTINATIONS=/rds:/yourfs` before installing
or running mini-swe. Generated Singularity configs also put
`/opt/miniconda3/envs/testbed/bin` first on `PATH` so commands use the
SWE-bench task environment rather than the base Conda Python.

## Batch smoke test

Submit from the repository root. Preview the three dependent PBS submissions:

```bash
DRY_RUN=1 cluster/submit_verified_smoke.sh
```

Then submit a real five-instance collection, evaluation, and smoke analysis:

```bash
cluster/submit_verified_smoke.sh
```

The collection job reserves one GPU, four CPU cores, and 32 GB of host memory.
It starts its own vLLM server and runs four trajectories concurrently. The
dependent CPU-only job merges the five predictions and evaluates two instances
concurrently, then the smoke analysis runs after evaluation succeeds. Outputs
are kept together under:

```text
$DEBUG_DEPO_SCRATCH/runs/agentforge-verified-smoke/
  cluster-logs/
  rollouts/shard-0/
  merged/
  evaluation/
  analysis-smoke/
```

Use a new name for an independent run, or reuse a name to resume completed
trajectories:

```bash
RUN_NAME=smoke-002 cluster/submit_verified_smoke.sh
```

Resumed Verified collection reuses completed and model-terminated trajectories
but retries infrastructure-error slots. Collection and evaluation write their
diagnostic summaries and fail the job when any infrastructure outcome remains,
so dependent `afterok` jobs cannot proceed on an incomplete run. Successful
Apptainer evaluation reports are reused only when their prediction, task,
generated test script, image configuration, and evaluator provenance still
match. New collection manifests also compare the dataset revision, selected
task IDs and row-content hash, mini-swe-agent version/Git state, and
result-affecting settings before reusing a trajectory. A mismatch requires a new
run directory or `OVERWRITE=1`. Legacy runs remain available for analysis, but
unknown historical package provenance is not invented after the fact.

## Full collection and evaluation

Preview the default ten-shard submission:

```bash
DRY_RUN=1 cluster/submit_verified_full.sh
```

Submit all 500 SWE-bench Verified instances followed by evaluation and full analysis:

```bash
cluster/submit_verified_full.sh
```

The collection is a ten-element PBS array. Round-robin task selection gives
each shard 50 instances, with six trajectories active per shard. Every array
element reserves `ncpus=12`, `mem=64gb`, and one GPU, and starts a private vLLM
server on a job-specific localhost port. `MINI_SWE_WORKERS=1` is intentional:
the six-way concurrency is provided by `ROLLOUT_WORKERS=6`, and each mini-swe
command is already filtered to one instance. The six remaining CPU cores and
additional host memory provide headroom for vLLM and container overhead.
The per-trajectory timeout is 21,600 seconds, matching the completed full run.

The dependent evaluation job first requires ten shard prediction files and
exactly 500 merged prediction records. It then reserves 32 CPU cores and
256 GB of memory and runs 20 Apptainer evaluations concurrently, with a
60-minute timeout per evaluation and a 12-hour PBS walltime. No GPU is
requested for evaluation.

Outputs are written under:

```text
$DEBUG_DEPO_SCRATCH/runs/agentforge-verified-full/
  cluster-logs/
  rollouts/shard-0/ ... shard-9/
  merged/predictions.jsonl
  evaluation/reports/
  evaluation/logs/
  analysis/
```

Ten shards are a reasonable first full layout if the five-instance smoke test
shows a typical trajectory finishes within roughly five hours. Approximate
shard time is `ceil(tasks_per_shard / 6) * typical_task_time`, plus vLLM startup.
If that approaches the queue walltime, use 20 shards of 25 tasks instead:

```bash
NUM_SHARDS=20 RUN_NAME=agentforge-verified-full-20 cluster/submit_verified_full.sh
```

Twelve cores and 64 GB support six writable task containers while retaining
CPU and host-memory headroom for vLLM tokenization. The single GPU is expected
to become the limiting resource before the host allocation; do not increase
`ROLLOUT_WORKERS` beyond 12 without checking vLLM queueing, CPU saturation,
swapping, and out-of-memory failures.

## Analyze a completed run

The analysis is deterministic and CPU-only; it does not need the rollout LLM or a GPU. Submit a
smoke analysis from the repository root with the same run name used for collection:

```bash
RUN_NAME=agentforge-verified-full-20260715 cluster/submit_verified_analysis_smoke.sh
```

The smoke job still requires all 500 unique source predictions. It selects two representative rows
from every detected rollout shard, preferring one patched and one empty-patch row when both exist,
and writes them under `analysis-smoke/`. With the default ten shards this produces at most 20 data
rows and checks that artifacts can be joined across the whole array rather than only shard 0.

After the smoke job succeeds, submit the complete 500-row analysis:

```bash
RUN_NAME=agentforge-verified-full-20260715 cluster/submit_verified_analysis_full.sh
```

`cluster/submit_verified_analysis.sh` is the shared mode-aware implementation.
Preview any submission with `DRY_RUN=1`. The two jobs write:

```text
$DEBUG_DEPO_SCRATCH/runs/$RUN_NAME/analysis/
  instances.csv
  summary.json
$DEBUG_DEPO_SCRATCH/runs/$RUN_NAME/analysis-smoke/
  instances.csv
  summary.json
```

`instances.csv` has one row per merged prediction. It includes rollout/mini-swe status, patch
size and diff statistics, total trajectory steps (`trajectory_steps`), total stored messages
(`trajectory_messages`), model API calls (`model_api_calls`), duration, SWE-bench resolution and
test counts, and a failure category with short log evidence. Steps and API calls are calculated
independently: mini-swe steps use the same message grouping as its trajectory inspector, while API
calls come only from `info.model_stats.api_calls`. For historical runs, duration is estimated from
mini-swe's timestamped exit-status artifact (falling back to `task.json`) through `trajectory.json`
completion and is marked with `duration_is_estimate=true`. Missing values remain blank rather than
being guessed or substituted from another metric.

Additional scalar columns summarize prompt/completion tokens, peak and final context usage, step
and context-limit utilization, finish reasons, agent action count, commands with successful/failed
or missing return codes, repeated commands, test/edit command steps, format errors, command
timeouts, truncated observations, submission step, and seconds per action. Test/edit and repeated
command counts are syntactic heuristics; token usage, return codes, limits, finish reasons, and
message counts come directly from trajectory records. Raw prompts, responses, and command output
are not copied into the CSV, so these columns add little file size compared with evidence and path
fields already present.

The same analysis can be run without PBS on a node with access to RDS:

```bash
RUN_ROOT="$RDS/ephemeral/debug-depo/runs/agentforge-verified-full-20260715" \
  scripts/analyze_run.sh
```

For the same cross-shard smoke selection without PBS, add
`ANALYSIS_SAMPLE_PER_SHARD=2` and set `ANALYSIS_OUTPUT_DIR` to a separate directory.

Rows marked `needs_llm_review=true` are the useful subset for a later semantic review. In
particular, unresolved patches need code/test reasoning, while infrastructure failures, timeouts,
mini-swe exit statuses, patch-application failures, and discarded error patches can be explained
directly from artifacts without serving a model.

To submit collection without automatically queuing evaluation:

```bash
SUBMIT_EVAL=0 cluster/submit_verified_full.sh
```

To queue collection and evaluation without the default dependent analysis job, set
`SUBMIT_ANALYSIS=0`.

After the array completes, submit evaluation manually with the matching run
name and shard count:

```bash
RUN_NAME=agentforge-verified-full
source cluster/resolve_run_paths.sh
mkdir -p "$CLUSTER_LOG_DIR"
qsub -o "$CLUSTER_LOG_DIR/" -e "$CLUSTER_LOG_DIR/" \
  -v RUN_NAME="$RUN_NAME",RUN_ROOT="$RUN_ROOT",NUM_SHARDS=10 \
  cluster/pbs/evaluate_verified_full.pbs
```

The automatic scripts use `afterok` PBS dependencies, so evaluation begins
only after the complete collection job or array succeeds, and analysis begins
only after evaluation succeeds. Submission wrappers create
`$RUN_ROOT/cluster-logs/` and send PBS stdout and stderr there; the PBS
templates intentionally leave output paths to those wrappers.

## Prebuild the task-image caches

Collection and evaluation can share a prebuilt persistent SIF cache. The
prebuilder resolves:

- one Epoch image for every selected SWE-bench Verified instance;
- one repository image for every distinct SWE-smith profile represented by the
  selected task-ID file.

SWE-smith task IDs are resolved against the pinned dataset and then deduplicated
by image URI. The tracked `train_instance_ids.txt` contains 45,809 tasks across
117 repository snapshots, so it requires far fewer SIFs than tasks. The build
summary records the definitive selected-task and unique-image counts.

First preview and submit a two-image smoke test (one task from each family):

```bash
DRY_RUN=1 cluster/submit_apptainer_cache_smoke.sh
cluster/submit_apptainer_cache_smoke.sh
```

Then build all 500 Verified SIFs plus the SWE-smith SIFs required by the
tracked 5,000-task training and 500-task validation samples:

```bash
SWESMITH_TASK_IDS_FILE=data/splits/swesmith_cache_5500_instance_ids.txt \
  DRY_RUN=1 cluster/submit_apptainer_cache_full.sh
SWESMITH_TASK_IDS_FILE=data/splits/swesmith_cache_5500_instance_ids.txt \
  cluster/submit_apptainer_cache_full.sh
```

Use the validation membership instead when preparing a validation-only run:

```bash
SWESMITH_TASK_IDS_FILE=data/splits/swesmith_validation_500_instance_ids.txt \
  cluster/submit_apptainer_cache_full.sh
```

The job uses the same `SWEBENCH_APPTAINER_*` and
`SWESMITH_APPTAINER_*` directories as collection and evaluation. Pulls use
temporary files plus per-SIF filesystem locks, and rerunning the job skips
completed SIFs. It attempts all images, records failures, and exits nonzero if
any pull failed. Summaries are written under
`$DEBUG_DEPO_SCRATCH/cache-builds/`. `cluster/pull_cluster_artifacts.sh`
includes these summaries by default under
`scratch/cluster-artifacts/cache-builds/`, alongside the complete
`scratch/cluster-artifacts/runs/` tree. Set `PULL_CACHE_BUILDS=0` to skip them.
Standalone cache-build PBS logs are stored under
`$DEBUG_DEPO_SCRATCH/runs/apptainer-cache-{smoke,full}/cluster-logs/`. The
cache-first pilot helper instead puts its cache log in the pilot run's
`cluster-logs/` directory with the downstream jobs.

The smoke template reserves 8 CPUs and 64 GB for 24 hours and runs two pulls
concurrently. The full template reserves 32 CPUs and 128 GB for 48 hours and
runs 20 pulls concurrently. Twenty is an intentionally moderately aggressive
single-node starting point: registry bandwidth, shared-cache I/O, and
temporary-storage capacity are more likely limits than CPU. Reduce
`CACHE_BUILD_MAX_WORKERS` if the smoke logs or site monitoring show registry
throttling, I/O saturation, or memory pressure.

Useful overrides are:

```bash
# Preview image resolution without pulling or creating cache directories.
DRY_RUN=1 CACHE_BUILD_MODE=full scripts/build_apptainer_cache.sh

# Build only one family or tune concurrent pulls.
CACHE_BUILD_DATASETS=swesmith CACHE_BUILD_MAX_WORKERS=12 \
  cluster/submit_apptainer_cache_full.sh
```

Verified collection injects the same
`SWEBENCH_APPTAINER_IMAGE_TEMPLATE` used by evaluation into mini-swe's local
task row. Consequently, the 500 prebuilt Verified SIFs serve both phases rather
than producing separate Docker Hub and GHCR cache entries.

## Evaluation storage

The Apptainer evaluator reconstructs missing per-instance SIF files from its
cache, then runs the SWE-bench evaluation scripts with `apptainer exec
--writable-tmpfs`.

To override the cache layout, put these exports in the ignored
`cluster/env/local.sh` on the cluster. Use `$RDS` rather than a user-specific
path:

```bash
export SWEBENCH_APPTAINER_CACHE_DIR="$RDS/ephemeral/debug-depo/swebench_epoch_cache/apptainer-cache"
export SWEBENCH_APPTAINER_SIF_DIR="$RDS/ephemeral/debug-depo/swebench_epoch_cache/sifs"
```

If you deleted temporary SIF files but kept `apptainer-cache`, the evaluator
will run `apptainer pull <instance>.sif docker://...`; Apptainer should reuse
the cached blobs and rebuild the SIF faster than the initial network pull.

Useful evaluation knobs:

```bash
EVAL_MAX_WORKERS=4 cluster/submit_verified_full.sh
EVAL_TIMEOUT=3600 cluster/submit_verified_full.sh
```

The Apptainer evaluator reuses SWE-bench task specs and grading, but it is a
runtime port of the Docker harness. For strict official comparison, note that
the evaluation was run with Apptainer-converted images.

## SWE-smith collection pipeline

The SWE-smith path is deliberately separate from the Verified evaluation run.
It preserves four predictions at each of two temperatures for every training
task through all three phases:

1. Collection expands each selected task into 8 sample slots: four each at
   `0.6` and `0.7`.
2. Evaluation merges shards within each slot and evaluates all 8 slots.
3. Analysis joins outcomes per task and writes per-temperature pass@1…4 plus
   explicitly labelled mixed-temperature pool views.

Run the cluster environment setup once after syncing. It now installs both
mini-swe-agent-plus and the official SWE-smith package:

```bash
bash cluster/setup_rollout_env.sh
```

Preview the two-task smoke run:

```bash
DRY_RUN=1 cluster/submit_swesmith_smoke.sh
```

Submit it:

```bash
cluster/submit_swesmith_smoke.sh
```

The bounded pilot defaults to 30 tasks over three collection shards:

```bash
DRY_RUN=1 cluster/submit_swesmith_pilot.sh
cluster/submit_swesmith_pilot.sh
```

To prebuild only the cache images needed by the pilot and then run the
collection, evaluation, and analysis jobs as one PBS dependency chain:

```bash
DRY_RUN=1 cluster/submit_swesmith_pilot_with_cache.sh
cluster/submit_swesmith_pilot_with_cache.sh
```

This helper pins the pilot to the first 30 IDs in
`data/splits/swesmith_train_5000_instance_ids.txt`. Set `RUN_NAME`,
`TASK_IDS_FILE`, `TASK_LIMIT`, or `NUM_SHARDS` before the command to override
those defaults. The cache stage uses the full cache-builder logic with only
the selected SWE-smith tasks and eight concurrent image pulls.

The smoke jobs match the Verified smoke allocation: collection uses four CPU
cores, one GPU, and 32 GB of memory, while evaluation uses four CPU cores and
32 GB. Smoke defaults to two rollout and two evaluation workers. Each pilot
collection shard uses eight CPU cores, one GPU, and 48 GB with five rollout
workers; pilot evaluation uses 16 CPU cores, 128 GB, and 12 workers. Each full
collection shard uses 12 CPU cores, one GPU, and 64 GB with six rollout
workers. Full evaluation uses 32 CPU cores, 256 GB, and 25 workers. Full
analysis uses four CPU cores, 32 GB, and a three-hour walltime; smoke and pilot
analysis retain the smaller two-CPU, 8 GB, one-hour template.

Each temperature receives four independent runs. Every task/sample pair also
receives a stable seed derived from `BASE_SEED=42`.
The full submission wrapper defaults to the tracked 5,000-task training sample,
50 shards, and 5,000 expected predictions. Each collection shard has a
24-hour walltime:

```bash
DRY_RUN=1 cluster/submit_swesmith_full.sh
cluster/submit_swesmith_full.sh
```

The corresponding 500-task validation command and complete deterministic
sampling policy are documented in `data/splits/README.md`.

To select the complete pinned 50,908-task Python split explicitly:

```bash
TASK_IDS_FILE= EXPECTED_TASKS=50908 DRY_RUN=1 \
  cluster/submit_swesmith_full.sh
TASK_IDS_FILE= EXPECTED_TASKS=50908 \
  cluster/submit_swesmith_full.sh
```

The complete split produces 407,264 trajectories across eight samples per
task, so inspect the dry run and cluster array limits before submitting it.
For any other subset, set `TASK_IDS_FILE` and `EXPECTED_TASKS` together.

`EXPECTED_TASKS` is checked against the selected dataset before collection and
independently for all 8 merged sample files. Collection and evaluation jobs
also fail on infrastructure outcomes instead of turning them into unresolved
model attempts. Limit/context terminations remain valid model outcomes with
empty patches, and resumed collection jobs retry only failed sample slots. All four
temperature replicates for a task are assigned to the same collection shard,
while the `ROLLOUT_WORKERS` pool can run sample slots concurrently.

The collector passes each subprocess its already-selected task JSON through the
tracked `debug_depo.miniswe_task` adapter. This preserves the pinned official
mini-swe runners and per-trajectory process isolation without reloading the
entire Hugging Face split for every trajectory. Because SWE-smith images are
repository-level, the generated startup command checks out the selected task
branch before the agent starts. The cluster default `singularity` runner
executes this command; mini-swe-agent-plus's `pool_way` runner does not and is
rejected for non-mock SWE-smith collection.

The tracked default pins `SWE-bench/SWE-smith-py` to revision
`77cab9055d42ab4a5c25c89a8f937096db13558e`. The setup scripts also pin the
mini-swe-agent-plus and SWE-smith repositories. Override
`SWESMITH_DATASET_REVISION`, `MINI_SWE_AGENT_PLUS_REVISION`, or
`SWESMITH_REVISION` only when intentionally starting a run on a new snapshot;
the collection manifests record the actual dependency revisions plus hashes of
any deterministic installer patches, and reject an incompatible resume.

The output layout is:

```text
$DEBUG_DEPO_SCRATCH/runs/$RUN_NAME/
  cluster-logs/
  collection/shard-*/
    collection_manifest.json
    samples/sample-0/ ... sample-7/
  merged/sample-0/ ... sample-7/
  evaluation/sample-0/ ... sample-7/
  analysis/
    rollouts.csv
    tasks.csv
    summary.json
```

SWE-smith currently builds and evaluates repository environments through
Docker upstream. This cluster workflow ports the execution step to Apptainer
and imports SWE-smith's repository profiles, test selection, timeouts, and
grading. Keep its converted images and cache in ephemeral storage:

```bash
export SWESMITH_APPTAINER_CACHE_DIR="$RDS/ephemeral/debug-depo/swesmith_cache/apptainer-cache"
export SWESMITH_APPTAINER_SIF_DIR="$RDS/ephemeral/debug-depo/swesmith_cache/sifs"
```

Collection and evaluation share these persistent SIFs. The first process that
needs an image pulls it into a temporary file under a filesystem lock and then
atomically installs the completed SIF. Other PBS shards wait for that lock and
reuse the same file. Collection still creates a separate writable sandbox for
each trajectory, but builds it from the local SIF under the configured `TMPDIR`
instead of fetching `docker://...` for every rollout. Evaluation executes the
same cached SIF read-only with writable temporary state.

The mini-swe-agent-plus checkout remains pinned to its tracked upstream commit.
The deterministic installer patch changes only cluster integration details:
local-vLLM compatibility, the working-directory prompt, writable bind
destinations, and persistent SIF selection. The task rows still come from the
pinned `SWE-bench/SWE-smith-py` `train` split and are passed to the official
mini-swe `swebench` runner by `debug_depo.miniswe_task`.

For an interactive cluster workflow, open
`notebooks/cluster_agentforge_swesmith.ipynb` with the `debug-depo` kernel.
It keeps model launch, collection, evaluation, analysis, and PBS submission
behind explicit switches, while previewing the submission chain by default.
To inspect artifacts from an existing run, use
`notebooks/inspect_swesmith_collection.ipynb` and set `RUN_NAME` or `RUN_ROOT`
if the run is not named `swesmith-pilot`.

## Preference training jobs

After `cluster/setup_rollout_env.sh`, install the optional training stack into
the project environment with:

```bash
bash cluster/setup_training_env.sh
```

`cluster/submit_preference_data.sh` first creates the two immutable training
datasets exactly once per evaluated trajectory collection:

```text
build_dmpo_pairs.pbs ─┐
build_depo_data.pbs  ─┴─> hash-validated immutable preference-data/
```

The jobs are independent and may overlap. Once both finish, the intended
training sequence is:

```text
train DMPO -> package DMPO -> evaluate DMPO
                         |
inspect/select DMPO -----┴-> train DEPO -> package DEPO -> evaluate DEPO
```

Set `EXPERIMENT_ARM=dmpo` or `depo` for either single-method branch. Preview
the default sequential arm before submission:

```bash
RUN_NAME=swesmith-pilot-20260719 EXPERIMENT_ARM=dmpo-depo DRY_RUN=1 \
  cluster/submit_preference_training.sh
```

The default chain requests:

| Stage | CPUs | GPUs | Host memory | Walltime |
| --- | ---: | ---: | ---: | ---: |
| Build DMPO pairs | 2 | 0 | 64 GB | 1 h |
| Build DEPO data | 2 | 0 | 32 GB | 1 h |
| Train DMPO | 8 | 1 | 64 GB | 48 h |
| Package DMPO | 4 | 0 | 64 GB | 4 h |
| Train DEPO | 8 | 1 | 64 GB | 48 h |
| Package DEPO | 4 | 0 | 64 GB | 4 h |

The overlapping builders use four CPUs and 96 GB aggregate host
memory; each training job uses one GPU, eight CPUs, and 64 GB host memory. The
reservation ceiling is 96 GPU-hours; actual use ends when each job finishes. A
high-memory GPU is recommended for the 8B model at 32K. If a pilot runs out of
device memory, reduce `MAX_LENGTH` before requesting more GPUs.

Packaging is CPU-only, so each GPU allocation ends as soon as training
finishes. By default, each completed package is evaluated on the existing
500-task SWE-bench Verified evaluation split by reusing the proven
`cluster/submit_verified_full.sh` workflow. It uses the existing ten-element
rollout array, one CPU-only evaluation job, and one analysis job; no 50-element
array is introduced. Each model receives one temperature-0 attempt per task at
a 32K context. Results are isolated under:

```text
$DEBUG_DEPO_SCRATCH/runs/<training-run>-dmpo-<dmpo-trial>-evaluation-500/
$DEBUG_DEPO_SCRATCH/runs/<training-run>-depo-<depo-trial>-evaluation-500/
$DEBUG_DEPO_SCRATCH/runs/<training-run>-dmpo-<dmpo-trial>-depo-<depo-trial>-evaluation-500/
```

Override `EVAL_NUM_SHARDS` to use a smaller rollout array; it need not be ten
as long as it is positive and no larger than the expected task count.

The DMPO evaluation branches from DMPO packaging and may run alongside DEPO
training; DEPO evaluation starts after DEPO packaging. Set
`SUBMIT_MODEL_EVALUATIONS=0` to submit training and packaging only. Preview one
evaluation independently with:

```bash
PREFERENCE_OBJECTIVE=dmpo \
TRAIN_RUN_NAME=swesmith-train-5000 \
DRY_RUN=1 \
cluster/submit_preference_evaluation.sh
```

The pair builders select four rollouts per task by default. Selection is
temperature-balanced and deterministic: with two temperatures it takes two
rollouts from each (sample slots `0,1,4,5` for the current layout); with four
temperatures it takes one from each (`0,4,8,12`). Set
`PREFERENCE_SAMPLE_INDICES=2:3:6:7` for an explicit same-temperature or mixed
choice, or `PREFERENCE_MAX_ROLLOUTS=0` to use every collected rollout. DMPO and
DEPO receive the same selection. The submission wrapper also accepts commas
and converts them to the colon form required inside `qsub -v`.

Collection keeps the target run's 65,536-token context default. Preference
training and packaged-model evaluation default to 32,768 tokens, so training a
64K collection can truncate long trajectories. Training defaults to bf16,
PyTorch SDPA, gradient checkpointing, one trajectory per device, and gradient
accumulation of 32 on the one-GPU template. Shared preference defaults live in
`scripts/preference_defaults.sh`. Checkpoints are written to temporary
directories and atomically promoted, later-epoch shuffling is deterministic,
and every epoch ends with a checkpoint. Incomplete checkpoint directories are
ignored. Completed training and packages are reused; interrupted packages are
preserved and rebuilt atomically. The final DMPO
package lives under `$RUN_ROOT/experiments/dmpo/<dmpo-trial>/model`. Direct
baseline-DEPO packages live under
`$RUN_ROOT/experiments/depo/<depo-trial>/model`; sequential packages live under
`$RUN_ROOT/experiments/dmpo-depo/<dmpo-trial>/depo/<depo-trial>/model`.
These are standalone Hugging Face packages, not adapter-only directories, and
can be passed directly as `AGENTFORGE_MODEL` to the existing vLLM-backed
evaluation jobs.

Use `DMPO_TRIAL_NAME` and `DEPO_TRIAL_NAME` to run multiple configurations over
the same collection. Set `EXPERIMENT_ARM=dmpo`, `depo`, or `dmpo-depo`;
`dmpo-depo` is the default. Training defaults to
`PREFERENCE_DATA_MODE=reuse`; run `cluster/submit_preference_data.sh` before
the first trial. For another sequential DEPO configuration on an existing
DMPO model, set `DMPO_MODE=reuse`. Each training directory
contains `trial_config.json`; a changed arm, parent model, hyperparameter, or
data hash cannot resume into an existing trial name.

The intended comparison is:

```text
existing baseline result
baseline SFT -> DMPO
baseline SFT -> DEPO
baseline SFT -> DMPO -> DEPO
```

Train and assess DMPO first:

```bash
RUN_NAME=swesmith-train-5000 \
EXPERIMENT_ARM=dmpo \
DMPO_TRIAL_NAME=gamma07 \
cluster/submit_preference_training.sh
```

Then train DEPO from that selected DMPO package:

```bash
RUN_NAME=swesmith-train-5000 \
EXPERIMENT_ARM=dmpo-depo \
DMPO_MODE=reuse \
DMPO_TRIAL_NAME=gamma07 \
DEPO_TRIAL_NAME=alpha2 \
cluster/submit_preference_training.sh
```

The notebook `notebooks/cluster_preference_training.ipynb` validates the data,
provides a 64-row/8K/one-epoch pilot on the current collection, and exposes
separate guarded switches for the one-time data jobs, DMPO, and DEPO.
