# debug-depo

Utilities for reproducing this project's pinned Klear AgentForge 8B SFT result
on SWE-bench Verified, collecting and evaluating SWE-smith trajectories, and
training DMPO and DEPO efficiency-preference models. The code can be run on a
local MacBook, a Lambda Cloud VM, or a PBS cluster.

## Workflows

| Workflow | Purpose | Guide |
| --- | --- | --- |
| Completed results | Compare SFT, DMPO, and DMPO-to-DEPO on SWE-smith and SWE-bench Verified | [Results and discussion](docs/hyperparameter-sweep-results.md) |
| SWE-bench Verified | Reproduce the pinned 38.2% (191/500) AgentForge target | [SWE-bench workflow](docs/swebench.md) |
| SWE-smith | Collect and evaluate reproducible repeated trajectories per task | [SWE-smith workflow](docs/swesmith.md) |
| DMPO and DEPO | Build immutable preference data, train models, and compare validation efficiency | [Preference-training workflow](docs/preference-training.md) |
| Training method | Understand the DMPO and DEPO data construction and losses | [DMPO/DEPO explanation](docs/depo-dmpo-model-training.md) |
| Dataset design | Reproduce the repository-disjoint training, screening, and confirmation sets | [Dataset splits](docs/dataset-splits.md) |
| Hyperparameter sweep | Screen DMPO/DEPO trials, then run disjoint confirmatory validation | [Sweep protocol](docs/hyperparameter-sweep-light.md) |
| Historical design notes | Review the objective, pilot findings, and pre-experiment rationale | [Preference-optimization notes](docs/preference-optimization.md) |

The published Verified reproduction target is 38.2% (191/500). The completed
experiment's SFT baseline resolved 196/500 (39.2%); see the
[results](docs/hyperparameter-sweep-results.md#swe-bench-verified-test-results).
Both use the dataset pinned to
`princeton-nlp/SWE-bench_Verified@c104f840cc67f8b6eec6f759ebc8b2693d585d4a`,
split `test`, with 500 instances, a 200-step budget, and a 64K context. The
wrapper installs the official `mini-swe-agent-plus` scaffold; it does not
vendor the Kwai/Klear harness.

## Where to run

### 1. Local MacBook

Use the local path for development and lightweight smoke tests. Install the
development environment and run a one-task test without a model or Docker:

```bash
uv sync --extra dev
./setup.sh
MOCK=1 LIMIT=1 scripts/collect_rollouts.sh
```

For a model-backed smoke test, install the official harness and start a local
MLX server:

```bash
scripts/install_mini_swe_agent_plus.sh

MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  scripts/serve_local_llm.sh
```

In another shell:

```bash
HARNESS=mini-swe-agent-plus \
AGENTFORGE_MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
MINI_SWE_MODEL=hosted_vllm/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
LLM_BASE_URL=http://127.0.0.1:8000/v1 \
LLM_API_KEY=local \
LIMIT=1 \
scripts/collect_rollouts.sh
```

Official SWE-bench scoring also requires Docker:

```bash
uv sync --extra dev --extra swebench
scripts/evaluate_all.sh
```

The [SWE-bench guide](docs/swebench.md) covers the local server check, exact
target model, custom harness contract, outputs, resume validation, sharding, and
dataset overrides.

### 2. Lambda Cloud VM

Lambda Cloud—not PBS—was used for the completed 1,000-task main experiment:
trajectory collection, evaluation, preference training, and final validation.
On a configured VM, start with a smoke test before launching a full pipeline:

```bash
bash cloud/run.sh smoke swesmith
RUN_NAME=your-run-name bash cloud/run.sh pipeline swesmith
```

See the [Lambda Cloud guide](cloud/README.md) for VM setup, storage, caching,
training, recovery, and the exact experiment commands.

### 3. PBS cluster

The PBS path was used mainly for early code development and smoke testing; it
was not used for the completed 1,000-task experiment. It runs the same
workflows through scheduled jobs, arrays, and dependency chains:

```bash
DRY_RUN=1 cluster/submit_swesmith_smoke.sh
cluster/submit_swesmith_smoke.sh
```

See the [PBS cluster guide](cluster/README.md) for environment setup, storage,
Apptainer images, resource requests, and submission commands.

## Reproducibility and recovery

- Dataset revisions, selected task IDs, row hashes, dependency revisions, and
  result-affecting settings are recorded in run manifests.
- A run refuses incompatible artifact reuse. Choose a new run/output name or
  set `OVERWRITE=1` deliberately where supported.
- Step/context-limit terminations remain model outcomes; infrastructure
  failures fail the job and are retried on resume.
- Preference datasets record row counts and SHA-256 hashes. Complete data,
  checkpoints, and model packages are reused and published atomically.
- Keep separate `RUN_NAME`, `DMPO_TRIAL_NAME`, and `DEPO_TRIAL_NAME` values for
  independent datasets and experiments.

## Repository map

```text
src/debug_depo/   Python implementation and command-line tools
scripts/          Local collection, evaluation, data, and training entrypoints
cluster/          PBS submission, environment, and Apptainer support
cloud/           Direct multi-GPU setup, dynamic sharding, cache, training, and validation
notebooks/        Guarded local and cluster workflows
data/splits/      Immutable task memberships and split provenance
docs/             Workflow and research notes
tests/            Unit and submission-contract tests
```

Use the [notebook index](notebooks/README.md) to distinguish current local
helpers, maintained PBS references, and historical/proposed notebooks. Before
any cluster submission, review `DRY_RUN=1` output and adapt the site-specific
queues, modules, walltimes, and scratch paths described in the
[cluster guide](cluster/README.md).

## AI Usage Statement

- Generative AI was used to improve code documentation, grammar, and tables.
- Tests were written by the author, with generative AI used for advice on test
  coverage and limited syntax assistance.
- Coding assistance was used to improve code syntax.
- The results-generation script contains some AI-generated code for producing
  publication-quality images.
