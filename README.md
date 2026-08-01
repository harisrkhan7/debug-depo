# debug-depo

Utilities for reproducing the Klear AgentForge 8B SFT result on SWE-bench
Verified, collecting and evaluating SWE-smith trajectories, and training DMPO
and DEPO efficiency-preference models. The same code supports lightweight Mac
smoke tests and full HPC runs.

## Workflows

| Workflow | Purpose | Guide |
| --- | --- | --- |
| SWE-bench Verified | Reproduce the reported 38.2% (191/500) AgentForge result | [SWE-bench workflow](docs/swebench.md) |
| SWE-smith | Collect eight reproducible trajectories per task and evaluate them | [SWE-smith workflow](docs/swesmith.md) |
| DMPO and DEPO | Build immutable preference data, train models, and compare held-out efficiency | [Preference-training workflow](docs/preference-training.md) |
| Hyperparameter sweep | Screen DMPO/DEPO trials on nested 100/200/500 validation budgets | [Sweep protocol](docs/hyperparameter-sweep.md) |
| Research plan | Track the objective, pilot findings, implementation coverage, and references | [Preference-optimization notes](docs/preference-optimization.md) |
| HPC | Configure storage, Apptainer, vLLM, PBS resources, and submission chains | [Cluster guide](cluster/README.md) |
| HyperStack | Run the same workflows on one persistent H200 x8 VM | [HyperStack guide](hyperstack/README.md) |

The Verified target is pinned to
`princeton-nlp/SWE-bench_Verified@c104f840cc67f8b6eec6f759ebc8b2693d585d4a`,
split `test`, with 500 instances, a 200-step budget, and a 64K context. The
wrapper installs the official `mini-swe-agent-plus` scaffold; it does not
vendor the Kwai/Klear harness.

## Quick start

Install the development environment and refresh the expected local structure:

```bash
uv sync --extra dev
./setup.sh
```

Run a one-task artifact smoke test without a model or Docker:

```bash
MOCK=1 LIMIT=1 scripts/collect_rollouts.sh
```

To run the official harness:

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

For official SWE-bench scoring on a Docker-capable machine:

```bash
uv sync --extra dev --extra swebench
scripts/evaluate_all.sh
```

The [SWE-bench guide](docs/swebench.md) covers the local server check, exact
paper model, custom harness contract, outputs, resume validation, sharding, and
dataset overrides.

## SWE-smith and preference training

Install SWE-smith and run the two-task artifact smoke test:

```bash
scripts/install_swesmith.sh
MOCK=1 MOCK_PATCH=gold LIMIT=2 scripts/collect_swesmith.sh
```

Preview the tracked 30-task pilot:

```bash
DRY_RUN=1 cluster/submit_swesmith_pilot.sh
```

After collection and evaluation, build preference data and preview training:

```bash
RUN_NAME=swesmith-pilot-20260719 \
  cluster/submit_preference_data.sh

RUN_NAME=swesmith-pilot-20260719 \
EXPERIMENT_ARM=dmpo \
DMPO_TRIAL_NAME=gamma07 \
DRY_RUN=1 \
cluster/submit_preference_training.sh
```

The tracked full SWE-smith sample contains 5,000 repository-disjoint training
tasks; the held-out sample contains 500 validation tasks. Exact memberships,
hashes, and generation policy are documented in
[`data/splits/README.md`](data/splits/README.md).

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
hyperstack/       Direct H200 x8 setup, sharding, cache, training, and validation
notebooks/        Guarded local and cluster workflows
data/splits/      Immutable task memberships and split provenance
docs/             Workflow and research notes
tests/            Unit and submission-contract tests
```

Start with the workflow-specific guide above. Before any cluster submission,
review `DRY_RUN=1` output and adapt the site-specific queues, modules,
walltimes, and scratch paths described in the [cluster guide](cluster/README.md).
