# DMPO and DEPO workflow

See the [project README](../README.md) for an overview and workflow index, and
the [optimization notes](preference-optimization.md) for the research
rationale, pilot findings, implementation status, and references.

## Build preference data

Build preference data only after SWE-smith collection and evaluation finish.
Both builders consume raw mini-swe messages, per-call token usage, and outcomes.
They remove provider response payloads but retain multi-turn assistant actions
and environment observations.

DMPO creates pairs: resolved beats unresolved; among two resolved
trajectories, the cheaper one wins only when the configured cost ratio is met.
Failure-versus-failure cost pairs are excluded by default.

DEPO creates independent KTO labels: resolved trajectories are desirable and
scored non-resolved trajectories are undesirable. Each record includes steps,
completion and total tokens per step, and their inverses:

```text
b(trajectory) =
  alpha1 * inverse_{completion|total}_tokens_per_step
  + alpha2 * inverse_steps
```

Build both datasets:

```bash
RUN_ROOT=scratch/cluster-artifacts/runs/swesmith-pilot-20260719 \
  scripts/build_dmpo_pairs.sh

RUN_ROOT=scratch/cluster-artifacts/runs/swesmith-pilot-20260719 \
  scripts/build_depo_data.sh
```

DMPO defaults to API `total_tokens`. Set `TOKEN_METRIC=completion_tokens` for
generated tokens only, `MIN_COST_RATIO=1.25` to require 25% savings, or
`MAX_PAIRS_PER_TASK=8` to cap correlated pairs.

Outputs:

```text
<run-root>/preference-data/
  dmpo/{pairs.jsonl,summary.json}
  depo/{trajectories.jsonl,desirable.jsonl,undesirable.jsonl,summary.json}
```

On PBS:

```bash
RUN_NAME=swesmith-pilot-20260719 DRY_RUN=1 \
  cluster/submit_preference_data.sh
RUN_NAME=swesmith-pilot-20260719 \
  cluster/submit_preference_data.sh
```

The two independent CPU jobs may overlap. Outputs are published atomically
with row counts and SHA-256 hashes. Complete artifacts are reused; set
`REBUILD_PREFERENCE_DATA=1` only for intentional replacement. Build once per
trajectory collection, including the eventual 5K run, then train in reuse
mode.

## Train and package models

Install the optional GPU stack after `cluster/setup_rollout_env.sh`:

```bash
bash cluster/setup_training_env.sh
```

Both entrypoints use LoRA, allowing the frozen initialization to serve as the
reference policy without a second 8B model. DMPO applies
`phi(t,T) = gamma^t * (1 - gamma^(T-t)) / (1 - gamma^T)` to assistant-action
tokens. DEPO uses unpaired KTO and adds
`alpha_tokens / tokens_per_step + alpha_steps / steps` only to desirable
trajectories. System prompts and observations provide context but are excluded
from the likelihood loss.

The direct scripts start from baseline SFT, resume their latest complete
`checkpoint-*`, and merge the final adapter into a standalone Hugging Face
model:

```bash
RUN_ROOT=scratch/cluster-artifacts/runs/swesmith-pilot-20260719 \
  scripts/train_dmpo.sh

RUN_ROOT=scratch/cluster-artifacts/runs/swesmith-pilot-20260719 \
  scripts/train_depo.sh
```

Cluster submissions support four comparison arms:

```text
baseline SFT                         existing result
EXPERIMENT_ARM=dmpo                 SFT -> DMPO
EXPERIMENT_ARM=depo                 SFT -> DEPO
EXPERIMENT_ARM=dmpo-depo (default)  SFT -> DMPO -> DEPO
```

Trained models live under:

```text
<run-root>/experiments/
  dmpo/<dmpo-trial>/{training,model}
  depo/<depo-trial>/{training,model}
  dmpo-depo/<dmpo-trial>/depo/<depo-trial>/{training,model}
```

`DMPO_TRIAL_NAME` and `DEPO_TRIAL_NAME` default to `default`. Trial names
isolate checkpoints, packages, and evaluations while sharing
`preference-data/`. `trial_config.json` records the arm, settings, parent,
and data hash; incompatible resume attempts fail. Checkpoint promotion,
packaging, and completed-run reuse are atomic.

Collection retains the paper's 65,536-token context. Training and packaged
evaluation default to `MAX_LENGTH=32768`; tune this for GPU memory. The builders
select four temperature-balanced slots per task: `0,1,4,5` for the current
two-temperature layout or `0,4,8,12` for four temperatures. Override with
`PREFERENCE_SAMPLE_INDICES`, or set `PREFERENCE_MAX_ROLLOUTS=0` for all
rollouts.

Common training overrides include `MAX_TRAIN_ROWS`, `NUM_PROCESSES`,
`PER_DEVICE_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `EPOCHS`,
`LEARNING_RATE`, `BETA`, `GAMMA`, `ALPHA_TOKENS`, `ALPHA_STEPS`, and
`DEPO_TOKEN_METRIC=completion_tokens|total_tokens`. PBS uses stage-specific
`DMPO_LEARNING_RATE`, `DMPO_BETA`, `DMPO_GAMMA`, `DEPO_LEARNING_RATE`, and
`DEPO_BETA`; shared batch, epoch, save, context, and alpha settings go to both
jobs.

## Cluster workflow

Build data once, then train and inspect DMPO:

```bash
RUN_NAME=swesmith-pilot-20260719 \
  cluster/submit_preference_data.sh

RUN_NAME=swesmith-pilot-20260719 \
EXPERIMENT_ARM=dmpo \
DMPO_TRIAL_NAME=gamma07 \
cluster/submit_preference_training.sh
```

After selecting its package, train DEPO from that DMPO model:

```bash
RUN_NAME=swesmith-pilot-20260719 \
EXPERIMENT_ARM=dmpo-depo \
DMPO_MODE=reuse \
DMPO_TRIAL_NAME=gamma07 \
DEPO_TRIAL_NAME=alpha2 \
cluster/submit_preference_training.sh
```

For a direct baseline-to-DEPO arm, set `EXPERIMENT_ARM=depo`. To try another
DEPO configuration on the same DMPO package, keep `DMPO_MODE=reuse` and choose
a new `DEPO_TRIAL_NAME`. Add `DRY_RUN=1` to preview any submission.

Training defaults to validated data reuse. A DMPO submission schedules GPU
training, CPU packaging, and evaluation; the sequential submission can reuse
that package and schedule only DEPO training, packaging, and evaluation. Set
`SUBMIT_MODEL_EVALUATIONS=0` to omit all evaluation, or
`SUBMIT_DMPO_EVALUATION=1` to repeat intermediate DMPO evaluation in reuse
mode.

## Evaluate and compare

Evaluation roots encode the run, arm, and trial:

```text
<run>-dmpo-<dmpo-trial>-evaluation-500
<run>-depo-<depo-trial>-evaluation-500
<run>-dmpo-<dmpo-trial>-depo-<depo-trial>-evaluation-500
```

Preview independent held-out evaluation:

```bash
PREFERENCE_OBJECTIVE=dmpo \
EXPERIMENT_ARM=dmpo \
TRAIN_RUN_NAME=swesmith-train-5000 \
DRY_RUN=1 \
cluster/submit_preference_evaluation.sh

PREFERENCE_OBJECTIVE=depo \
EXPERIMENT_ARM=dmpo-depo \
TRAIN_RUN_NAME=swesmith-train-5000 \
DRY_RUN=1 \
cluster/submit_preference_evaluation.sh
```

Analyze each held-out arm, then compare the exact per-instance matrices:

```bash
debug-depo-compare-preference-arms \
  --baseline sft=results/sft-evaluation-500/analysis/instances.csv \
  --arm dmpo=results/dmpo-evaluation-500/analysis/instances.csv \
  --arm dmpo-depo=results/dmpo-depo-evaluation-500/analysis/instances.csv \
  --expected-tasks 500 \
  --success-tolerance 0.01 \
  --output results/preference-arm-comparison.json
```

Here `0.01` allows at most a one-percentage-point absolute resolution-rate
drop. The command ranks the baseline and arms by total tokens per resolved
task, reports paired step and token deltas, and refuses selection when task
IDs differ, an evaluation is unscored, or token telemetry is incomplete.

The guarded workflow in `notebooks/cluster_preference_training.ipynb` uses 64
rows, an 8K context, one epoch, and a tracked five-task evaluation split for
the 30-task pilot. It is also linked from
`notebooks/cluster_agentforge_swesmith_train.ipynb`.

See the [cluster guide](../cluster/README.md) for resource sizing and complete
PBS dependency behavior.
