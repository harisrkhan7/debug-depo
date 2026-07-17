# Cluster Notes

The runner is split into three phases:

1. Rollout shards call the AgentForge harness and write SWE-bench predictions.
2. Evaluation merges predictions and runs the official SWE-bench Docker harness.
3. Analysis joins rollout and evaluation artifacts and writes run summaries.

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
Python module is newer. It does not install project dependencies or
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
sync script, so run `cluster/setup_rollout_env.sh` on the cluster at least once.
It creates the checkout's `.venv` and installs mini-swe-agent-plus there.

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

The first Apptainer/Singularity task run may be slow because mini-swe builds a
writable sandbox for the SWE-bench image under `TMPDIR`. The portable cluster
defaults point `TMPDIR` at ephemeral scratch by default. The local
`scripts/install_mini_swe_agent_plus.sh` patch also pre-creates `/rds` inside
that writable sandbox so Imperial's automatic `/rds` bind has a destination. If
your site auto-binds a different top-level filesystem, set
`MSWEA_SINGULARITY_WRITABLE_BIND_DESTINATIONS=/rds:/yourfs` before installing
or running mini-swe. Generated Singularity configs also put
`/opt/miniconda3/envs/testbed/bin` first on `PATH` so commands use the
SWE-bench task environment rather than the base Conda Python.

## Batch smoke test

Submit from the repository root. Preview the three dependent PBS submissions:

```bash
DRY_RUN=1 cluster/submit_smoke.sh
```

Then submit a real five-instance collection, evaluation, and smoke analysis:

```bash
cluster/submit_smoke.sh
```

The collection job reserves one GPU, four CPU cores, and 32 GB of host memory.
It starts its own vLLM server and runs four trajectories concurrently. The
dependent CPU-only job merges the five predictions and evaluates two instances
concurrently, then the smoke analysis runs after evaluation succeeds. Outputs
are kept together under:

```text
$DEBUG_DEPO_SCRATCH/runs/agentforge-verified-smoke/
  rollouts/shard-0/
  merged/
  evaluation/
  analysis-smoke/
```

Use a new name for an independent run, or reuse a name to resume completed
trajectories:

```bash
RUN_NAME=smoke-002 cluster/submit_smoke.sh
```

## Full collection and evaluation

Preview the default ten-shard submission:

```bash
DRY_RUN=1 cluster/submit_full.sh
```

Submit all 500 SWE-bench Verified instances followed by evaluation and full analysis:

```bash
cluster/submit_full.sh
```

The collection is a ten-element PBS array. Round-robin task selection gives
each shard 50 instances, with eight trajectories active per shard. Every array
element reserves `ncpus=12`, `mem=64gb`, and one GPU, and starts a private vLLM
server on a job-specific localhost port. `MINI_SWE_WORKERS=1` is intentional:
the eight-way concurrency is provided by `ROLLOUT_WORKERS=8`, and each mini-swe
command is already filtered to one instance. The four remaining CPU cores and
additional host memory provide headroom for vLLM and container overhead.

The dependent evaluation job first requires ten shard prediction files and
exactly 500 merged prediction records. It then reserves 32 CPU cores and
256 GB of memory and runs 20 Apptainer evaluations concurrently, with a
60-minute timeout per evaluation and a 12-hour PBS walltime. No GPU is
requested for evaluation.

Outputs are written under:

```text
$DEBUG_DEPO_SCRATCH/runs/agentforge-verified-full/
  rollouts/shard-0/ ... shard-9/
  merged/predictions.jsonl
  evaluation/reports/
  evaluation/logs/
  analysis/
```

Ten shards are a reasonable first full layout if the five-instance smoke test
shows a typical trajectory finishes within roughly five hours. Approximate
shard time is `ceil(tasks_per_shard / 8) * typical_task_time`, plus vLLM startup.
If that approaches the queue walltime, use 20 shards of 25 tasks instead:

```bash
NUM_SHARDS=20 RUN_NAME=agentforge-verified-full-20 cluster/submit_full.sh
```

Twelve cores and 64 GB support eight writable task containers while retaining
CPU and host-memory headroom for vLLM tokenization. The single GPU is expected
to become the limiting resource before the host allocation; do not increase
`ROLLOUT_WORKERS` beyond 12 without checking vLLM queueing, CPU saturation,
swapping, and out-of-memory failures.

## Analyze a completed run

The analysis is deterministic and CPU-only; it does not need the rollout LLM or a GPU. Submit a
smoke analysis from the repository root with the same run name used for collection:

```bash
RUN_NAME=agentforge-verified-full-20260715 cluster/submit_analysis_smoke.sh
```

The smoke job still requires all 500 unique source predictions. It selects two representative rows
from every detected rollout shard, preferring one patched and one empty-patch row when both exist,
and writes them under `analysis-smoke/`. With the default ten shards this produces at most 20 data
rows and checks that artifacts can be joined across the whole array rather than only shard 0.

After the smoke job succeeds, submit the complete 500-row analysis:

```bash
RUN_NAME=agentforge-verified-full-20260715 cluster/submit_analysis_full.sh
```

`cluster/submit_analysis.sh` remains a backward-compatible alias for the full analysis. Preview any
submission with `DRY_RUN=1`. The two jobs write:

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
SUBMIT_EVAL=0 cluster/submit_full.sh
```

To queue collection and evaluation without the default dependent analysis job, set
`SUBMIT_ANALYSIS=0`.

After the array completes, submit evaluation manually with the matching run
name and shard count:

```bash
qsub -v RUN_NAME=agentforge-verified-full,NUM_SHARDS=10 cluster/pbs/evaluate_all.pbs
```

The automatic scripts use `afterok` PBS dependencies, so evaluation begins
only after the complete collection job or array succeeds, and analysis begins
only after evaluation succeeds.

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
EVAL_MAX_WORKERS=4 cluster/submit_full.sh
EVAL_TIMEOUT=3600 cluster/submit_full.sh
```

The Apptainer evaluator reuses SWE-bench task specs and grading, but it is a
runtime port of the Docker harness. For strict official comparison, note that
the evaluation was run with Apptainer-converted images.
