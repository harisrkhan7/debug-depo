# SWE-bench Verified workflow

See the [project README](../README.md) for an overview and workflow index.

## Reproduction target

| Setting | Value |
| --- | --- |
| Dataset | `princeton-nlp/SWE-bench_Verified` |
| Revision | `c104f840cc67f8b6eec6f759ebc8b2693d585d4a` |
| Split / instances | `test` / 500 |
| Harness | `mini-swe-agent-plus` with the AgentForge SWE scaffold |
| Budget | 200 steps, 64K context |
| Repository comparison target | 38.2% (191/500 resolved) |

Kwai/Klear publish the model project in
[`Kwai-Klear/Klear-AgentForge`](https://github.com/Kwai-Klear/Klear-AgentForge)
and the runnable scaffold in
[`mini-swe-agent-plus`](https://github.com/Kwai-Klear/mini-swe-agent-plus).
This repository installs and wraps that scaffold; it does not vendor it. The
wrapper records per-instance trajectories, writes official SWE-bench prediction
JSONL, and launches the official evaluator.

The evaluation defaults are:

```text
DATASET=princeton-nlp/SWE-bench_Verified
SWEBENCH_DATASET_REVISION=c104f840cc67f8b6eec6f759ebc8b2693d585d4a
SPLIT=test
EXPECTED_COUNT=500
```

That revision is the exact snapshot used by the completed
`agentforge-verified-full-20260715` run. Only a complete default evaluation
with the AgentForge model is compared with the repository's pinned 38.2%
target; partial
runs and other datasets, splits, or models report measurements without target
comparison fields.

The official Docker evaluator does not accept a Hugging Face revision itself.
Before launching it, this repository therefore loads the requested pinned rows
and the upstream evaluator's current rows and requires their complete canonical
row hashes to match. The evaluation summary records the revision and row hash,
and target-comparison fields additionally require the exact pinned revision. A
snapshot mismatch fails before Docker starts; the Apptainer evaluator loads the
pinned revision directly.

To reuse the pipeline for another immutable dataset role:

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

Use a distinct `RUN_NAME` for each role. `RUN_ID` defaults to that name with
dashes changed to underscores. `NUM_SHARDS` cannot exceed `EXPECTED_COUNT`.
A non-default remote dataset is unpinned unless
`SWEBENCH_DATASET_REVISION` is supplied.

## Install and smoke test

```bash
uv sync --extra dev
./setup.sh
```

Add the official evaluator dependencies on a Docker-capable machine:

```bash
uv sync --extra dev --extra swebench
```

The local smoke test calls neither a model nor Docker. It checks task
selection, trajectory storage, prediction JSONL, and summaries:

```bash
MOCK=1 LIMIT=1 scripts/collect_rollouts.sh
```

To smoke-test the evaluator with a gold patch:

```bash
MOCK=1 MOCK_PATCH=gold LIMIT=1 scripts/collect_rollouts.sh
scripts/evaluate_all.sh
```

See `notebooks/local_agentforge_swebench_smoke.ipynb` for the notebook
equivalent.

## Run the official harness

Install `mini-swe-agent-plus` into this repository's `.venv`:

```bash
scripts/install_mini_swe_agent_plus.sh
```

For a local Mac run, start an OpenAI-compatible MLX server:

```bash
MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  scripts/serve_local_llm.sh
```

Verify both model discovery and a small chat completion:

```bash
PYTHONPATH=src python3 -m debug_depo.check_local_llm \
  --base-url http://127.0.0.1:8000/v1 \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
```

Then collect one trajectory:

```bash
HARNESS=mini-swe-agent-plus \
AGENTFORGE_MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
MINI_SWE_MODEL=hosted_vllm/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
LLM_BASE_URL=http://127.0.0.1:8000/v1 \
LLM_API_KEY=local \
LIMIT=1 \
scripts/collect_rollouts.sh
```

The official harness expects a LiteLLM-style name, hence the
`hosted_vllm/...` prefix in `MINI_SWE_MODEL`. For the target model, serve and
set both model variables to `Kwai-Klear/Klear-AgentForge-8B-SFT`, retaining
that prefix for `MINI_SWE_MODEL`.

The wrapper selects the upstream `swebench_add_edit_tool.yaml` config and
writes its required one-line vLLM server file. The server check is also
available in `notebooks/local_llm_server_check.ipynb`.

## Custom harness

Set `AGENTFORGE_COMMAND` to use another entrypoint. It is rendered once per
instance and supports:

```text
{instance_id} {repo} {task_json} {output_dir} {model}
{llm_base_url} {llm_api_key} {agentforge_repo}
{max_steps} {context_length} {temperature} {top_p} {seed}
```

The command must write one of:

- `prediction.patch`, `prediction.diff`, `patch.patch`, or `patch.diff`;
- `prediction.json`, `result.json`, `trajectory.json`, `rollout.json`, or
  `output.json`, with a `model_patch`, `patch`, `diff`, `output_patch`, or
  `git_diff` field; or
- a JSON object with one of those fields on stdout.

Example:

```bash
AGENTFORGE_REPO=/path/to/Klear-AgentForge \
AGENTFORGE_COMMAND='python -m agentforge.swebench.run \
  --task-json {task_json} \
  --output-dir {output_dir} \
  --model {model} \
  --base-url {llm_base_url} \
  --api-key {llm_api_key} \
  --max-steps {max_steps} \
  --context-length {context_length}' \
LLM_BASE_URL=http://127.0.0.1:8000/v1 \
LLM_API_KEY=local \
scripts/collect_rollouts.sh
```

Persisted metadata redacts `llm_api_key` and its value in the rendered command.
A custom harness must still avoid printing credentials because stdout and
stderr are retained.

Outputs use this layout:

```text
data/processed/agentforge_swebench_verified/
  predictions.jsonl
  run_config.json
  summary.json
  trajectories/<instance_id>/
```

For new runs, `run_config.json` is also the resume manifest. It records the
dataset revision, selected instance IDs and row hash, mini-swe-agent
version/Git state, and result-affecting settings. Incompatible reuse fails
before trajectories are loaded; choose another output directory or set
`OVERWRITE=1` deliberately. Older runs remain readable, but missing historical
dependency provenance is not backfilled.

## Evaluation and HPC

Run official SWE-bench scoring with:

```bash
scripts/evaluate_all.sh
```

Apple Silicon automatically uses `--namespace ''` to build images locally.
Use x86_64 Linux with Docker or a compatible container setup for the full
benchmark.

Rollouts can be sharded and merged:

```bash
NUM_SHARDS=10 SHARD_INDEX=0 \
OUTPUT_DIR=$SCRATCH/agentforge/shard-0 \
scripts/collect_rollouts.sh

scripts/merge_predictions.sh \
  $SCRATCH/agentforge/shard-*/predictions.jsonl
```

PBS templates live in `cluster/pbs/`. See the
[cluster guide](../cluster/README.md) for environment setup, storage, resource
sizing, dry runs, Apptainer/vLLM, cache construction, and complete Verified
submission workflows. Adapt queues, modules, walltimes, and scratch paths to
the target cluster.
