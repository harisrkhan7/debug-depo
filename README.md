# debug-depo

Utilities for reproducing the Klear AgentForge 8B SFT result on
SWE-bench Verified while keeping the project structure lightweight enough for
Mac smoke tests and HPC batch runs.

## Target

The reproduction target is the Klear-AgentForge-8B-SFT SWE-bench Verified
score reported in the Klear-AgentForge paper:

- Dataset: `princeton-nlp/SWE-bench_Verified`
- Split: `test`
- Instances: 500
- Harness: `mini-swe-agent-plus` / AgentForge-style SWE scaffold
- Budget: 200 steps, 64k context
- Reported 8B SFT score: `38.2%` (`191 / 500` resolved)

The code here does not vendor the Kwai/Klear harness. Kwai/Klear publish the
model and report in `Kwai-Klear/Klear-AgentForge`, and the runnable SWE scaffold
is their official `mini-swe-agent-plus` repository. This project wraps that
installed package, records per-instance trajectories, emits the official
SWE-bench prediction JSONL format, and launches the official SWE-bench evaluator.

## Install

```bash
uv sync --extra dev
```

For official SWE-bench scoring on a Docker-capable machine:

```bash
uv sync --extra dev --extra swebench
```

Run the idempotent local scaffold step whenever you want the expected folders
and executable bits refreshed:

```bash
./setup.sh
```

## Mac Smoke Run

This does not call a model or Docker. It verifies task selection, trajectory
writing, prediction JSONL writing, and summaries.

```bash
MOCK=1 LIMIT=1 scripts/collect_rollouts.sh
```

A notebook version of this flow lives at
`notebooks/local_agentforge_swebench_smoke.ipynb`.

Use gold patches only when you specifically want to test the SWE-bench evaluator
on a tiny subset:

```bash
MOCK=1 MOCK_PATCH=gold LIMIT=1 scripts/collect_rollouts.sh
scripts/evaluate_all.sh
```

## Official mini-swe-agent-plus Harness

Install the official Kwai/Klear harness into this repo's `.venv`:

```bash
scripts/install_mini_swe_agent_plus.sh
```

Then point it at an OpenAI-compatible model server. For local Mac smoke tests,
start with a public MLX coder model:

```bash
MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit scripts/serve_local_llm.sh
```

The official harness expects a LiteLLM-style model name. For an MLX/OpenAI
server, use the `hosted_vllm/...` provider prefix while preserving the full
served model id:

Before launching the SWE harness, verify that the local server can both list
models and generate a tiny chat completion:

```bash
python3 scripts/check_local_llm.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
```

A notebook version of this check lives at
`notebooks/local_llm_server_check.ipynb`.

```bash
HARNESS=mini-swe-agent-plus \
AGENTFORGE_MODEL=mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
MINI_SWE_MODEL=hosted_vllm/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
LLM_BASE_URL=http://127.0.0.1:8000/v1 \
LLM_API_KEY=local \
LIMIT=1 \
scripts/collect_rollouts.sh
```

For the exact paper-target model, run the server with
`MODEL=Kwai-Klear/Klear-AgentForge-8B-SFT` and use
`AGENTFORGE_MODEL=Kwai-Klear/Klear-AgentForge-8B-SFT`
`MINI_SWE_MODEL=hosted_vllm/Kwai-Klear/Klear-AgentForge-8B-SFT`.

The wrapper automatically uses the package's
`swebench_add_edit_tool.yaml` config and writes the one-line vLLM server file
expected by the upstream script.

## Custom AgentForge Rollout

If you need a different harness entrypoint, set `AGENTFORGE_COMMAND`. The
command is rendered once per instance and can use these template fields:

`{instance_id}`, `{repo}`, `{task_json}`, `{output_dir}`, `{model}`,
`{llm_base_url}`, `{llm_api_key}`, `{agentforge_repo}`, `{max_steps}`,
`{context_length}`, `{temperature}`, `{top_p}`.

The command should write one of the following under `{output_dir}`:

- `prediction.patch`, `prediction.diff`, `patch.patch`, or `patch.diff`
- a JSON file named `prediction.json`, `result.json`, `trajectory.json`,
  `rollout.json`, or `output.json` containing `model_patch`, `patch`, `diff`,
  `output_patch`, or `git_diff`
- a JSON object with one of those keys on stdout

Example shape for a custom command:

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

Outputs land in:

```text
data/processed/agentforge_swebench_verified/
  predictions.jsonl
  run_config.json
  summary.json
  trajectories/<instance_id>/
```

## Official Evaluation

```bash
scripts/evaluate_all.sh
```

On Apple Silicon, the evaluator automatically passes `--namespace ''` so
SWE-bench builds images locally instead of pulling Linux images. For the full
benchmark, use an x86_64 Linux machine or HPC node with Docker or a compatible
container setup.

## HPC Shards

Rollouts are shardable:

```bash
NUM_SHARDS=10 SHARD_INDEX=0 OUTPUT_DIR=$SCRATCH/agentforge/shard-0 scripts/collect_rollouts.sh
```

Merge shard predictions before scoring:

```bash
scripts/merge_predictions.sh $SCRATCH/agentforge/shard-*/predictions.jsonl
```

PBS templates live in `cluster/pbs/`. They are intentionally plain so you can
adapt modules, queues, walltimes, and scratch paths to the cluster you end up
using.
