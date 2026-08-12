# DMPO and DEPO hyperparameter sweep

This document defines the bounded hyperparameter-search protocol for the
1,000-task SWE-smith preference-training experiment. It is designed for the
case where a 500-task model evaluation takes close to one day.

The protocol is a manual, successive-halving-like funnel:

```text
all candidates on 100 tasks
        |
        v
best two candidates on 200 tasks
        |
        v
winner, baseline, and at most one runner-up on 500 tasks
```

The smaller evaluations are screening budgets, not final estimates. Hyperband
formalizes this general idea by treating data samples as an allocatable
hyperparameter-optimization resource and stopping weak configurations early
([Li et al., 2018](#references)). This project uses a simpler fixed funnel
rather than claiming to implement Hyperband.

## Fixed validation memberships

Use these immutable, repository-covering samples:

| Budget | Task-ID file | Repositories | Ordered-file SHA-256 |
| ---: | --- | ---: | --- |
| 100 | `data/splits/swesmith_validation_100_instance_ids.txt` | 14 | `99e09b02b651e89282f76b994bac6e82be1776928d5c09d1ce8721806493fcf9` |
| 200 | `data/splits/swesmith_validation_200_instance_ids.txt` | 14 | `a27dcfa4bafb154f3b70baf51ab50696f6b7e3f978bcc845b2f2d72091c437fb` |
| 500 | `data/splits/swesmith_validation_500_instance_ids.txt` | 14 | `dd6342f7f4cb79ea206f523517fef7f3367f4a779cf0c088abf5b0e91088b154` |

All three use the split policy's seed 42, repository grouping, proportional
quota allocation, and SHA-256 selection and ordering. They are nested:

```text
validation-100 subset of validation-200 subset of validation-500
```

Do not redraw a subset after observing results. Use the same task membership,
evaluation temperature, rollout count, context length, step limit, and scoring
pipeline for every arm at a given budget.

Because the 100 and 200 tasks are contained in the 500, the 500-task result is
a higher-budget validation result, not an untouched test estimate. Report it
as validation performance. A final generalization claim needs a separate test
set that was not used for model or hyperparameter selection. Optimizing a
finite, noisy validation criterion can itself overfit and bias the reported
performance ([Cawley and Talbot, 2010](#references)).

## Selection objective

Treat success as a constraint and efficiency as the ranking objective:

```text
minimize total_tokens_per_resolved_task
subject to resolution_rate >= baseline_resolution_rate - 0.03
```

The three-percentage-point tolerance corresponds to three tasks at budget 100,
six tasks at budget 200, and fifteen tasks at budget 500. It is an operational
threshold, not a confidence interval.

For every comparison:

1. Require the exact same instance-ID matrix and complete scoring.
2. Reject configurations outside the success constraint.
3. Rank eligible configurations by `total_tokens_per_resolved_task`.
4. Inspect paired prompt-token, completion-token, total-token, and action-step
   deltas rather than relying only on aggregate means.
5. Inspect the numbers of baseline successes gained and lost.

If very few tasks are resolved, cost per resolved task will be unstable. Treat
the 100-task result as elimination evidence and move plausible candidates to
200 rather than declaring a winner from a small cost difference.

## Settings held fixed

The first sweep changes only the parameters named in the trial tables below.
Hold these settings fixed:

| Setting | Sweep value | Reason |
| --- | ---: | --- |
| Training tasks | 1,000 | Current reduced experiment |
| Base model | `Kwai-Klear/Klear-AgentForge-8B-SFT` | Current reference policy |
| Collection context length | 32,768 | Match the current rollout collection |
| Maximum training length | 8,192 | Current preference-training budget |
| Evaluation context length | 32,768 | Match packaged-model evaluation |
| Maximum evaluation steps | 200 | Current full-evaluation budget |
| Per-device batch size | 1 | Long-context memory constraint |
| Gradient accumulation | 16 | Effective batch 128 on eight GPUs |
| Epochs | 3 | Match the DEPO paper and current project default |
| LoRA rank/alpha | 64/128 | Current implementation default |
| Seed | 42 | Current implementation default |
| Evaluation samples | 1 per task | Cheap deterministic screening |
| Evaluation temperature | 0.0 | Remove decoding-temperature variation |

With eight processes, per-device batch size 1, and accumulation 16, the
effective global batch is 128. The repository default of accumulation 32 has
an effective batch of 256 and therefore produces relatively few optimizer
updates on a small preference dataset. Accumulation 16 is a project-specific
budget choice, not a value taken from the DMPO or DEPO papers. Record the
actual DMPO pair count and DEPO trajectory count before training and compute:

```text
approximate updates per epoch =
    ceil(number_of_training_rows / effective_global_batch)
```

If training is unstable, compare accumulation 16 against 32 once; do not vary
it independently in every objective trial.

## Stage 0: controls

Before tuning:

1. Build and validate one immutable set of DMPO and DEPO preference artifacts.
2. Evaluate the unmodified SFT model on the 100-task split.
3. Train and evaluate the current-default DMPO trial.
4. Give every configuration a unique `DMPO_TRIAL_NAME` or `DEPO_TRIAL_NAME`.
5. Do not rebuild preference data between trainer-only trials.

Retain the SFT evaluation at all three budgets. DMPO and DEPO are acceptable
only when their efficiency gains do not come from a material success-rate
loss. This follows the project's success-constrained deployment objective and
DEPO's principle that efficiency is optimized among successful trajectories
([Chen et al., 2026](#references)).

## Stage 1: DMPO

DMPO reweights turns through the discount factor `gamma`. Its paper searches
`beta` from 0.1 through 0.9 and `gamma` from 0.1 through 0.99. It also reports
that smaller `gamma` can reduce the influence of later actions in noisy losing
trajectories, whereas larger `gamma` can better use later actions in clean,
high-quality losing trajectories ([Shi et al., 2024](#references)).

The local sweep deliberately narrows that research range.

### Stage 1A: discount factor

Evaluate all three trials on 100 tasks:

| Trial | `DMPO_LEARNING_RATE` | `DMPO_BETA` | `DMPO_GAMMA` |
| --- | ---: | ---: | ---: |
| `g07-lr1e6-b01` | `1e-6` | `0.1` | `0.7` |
| `g09-lr1e6-b01` | `1e-6` | `0.1` | `0.9` |
| `g099-lr1e6-b01` | `1e-6` | `0.1` | `0.99` |

Select the best eligible `gamma`, retaining two when the 100-task evidence is
close.

### Stage 1B: learning rate

Using the leading `gamma`, compare:

| Trial suffix | `DMPO_LEARNING_RATE` |
| --- | ---: |
| `lr5e7` | `5e-7` |
| `lr1e6` | `1e-6` |
| `lr2e6` | `2e-6` |

The `1e-6` run already exists from Stage 1A and must be reused rather than
retrained. Promote the best two overall DMPO configurations to 200 tasks.

### Conditional DMPO tests

Only if the leading configurations remain tied or show weak preference
separation:

- compare `DMPO_BETA=0.05` and `0.2` around the winner;
- compare two versus three epochs;
- rebuild a separately named preference dataset with
  `MIN_COST_RATIO=1.25` to remove small cost-gap pairs.

The last item changes the data, not just the trainer. It must have separate
artifact provenance and must not overwrite the shared sweep data.

The original DPO formulation makes `beta` part of the policy/reference
log-ratio and hence the strength of the implicit reward/KL trade-off
([Rafailov et al., 2023](#references)). That motivates a conditional beta
check, but the expensive first pass prioritizes DMPO's objective-specific
`gamma`.

## Stage 2: DEPO

Initialize every core DEPO trial from the single selected DMPO package. Keep
`DEPO_BETA=0.2`, `DEPO_LEARNING_RATE=2e-5`, and three epochs during the bonus
sweep.

The DEPO paper extends KTO's unpaired desirable/undesirable objective with:

```text
bonus =
    alpha_tokens / tokens_per_step
    + alpha_steps / steps
```

The bonus is applied only to desirable trajectories. The paper reports
`beta=0.2`, learning rate `2e-5`, three epochs, equal desirable/undesirable
weights, and joint coefficients `2,2` for its Qwen2.5-7B experiment. Its
ablation compares the joint setting against token-only and step-only settings
and finds that the joint objective gives the best overall trade-off in that
environment ([Chen et al., 2026](#references)). The unpaired binary-label
foundation comes from KTO ([Ethayarajh et al., 2024](#references)).

### Stage 2A: efficiency bonus

Evaluate these trials on 100 tasks:

| Trial | `DEPO_TOKEN_METRIC` | `ALPHA_TOKENS` | `ALPHA_STEPS` | Status |
| --- | --- | ---: | ---: | --- |
| `paper-a2-a2` | `completion_tokens` | `2` | `2` | DEPO paper hyperparameters |
| `completion-balanced` | `completion_tokens` | `12` | `2` | Local scale calibration |
| `completion-step-only` | `completion_tokens` | `0` | `2` | Paper-style ablation |
| `total-balanced` | `total_tokens` | `600` | `2` | Deployment-cost extension |

`completion_tokens` is the closest repository metric to the generated-token
quantity used by the DEPO paper. `total_tokens` instead represents this
project's local-inference/deployment cost objective, where accumulated prompt
context is also costly.

The values 12 and 600 are not DEPO paper settings. They are rounded local
calibrations from all 111 successful trajectories in the tracked eight-rollout
pilot artifact:

| Pilot median | Value |
| --- | ---: |
| `inverse_completion_tokens_per_step` | `0.0071399` |
| `inverse_total_tokens_per_step` | `0.00015098` |
| `inverse_steps` | `0.0454545` |

With `ALPHA_STEPS=2`, equal median contribution implies approximately:

```text
completion alpha = 2 * 0.0454545 / 0.0071399  = 12.7
total alpha      = 2 * 0.0454545 / 0.00015098 = 602
```

This explains why raw `total_tokens, alpha=(2,2)` is not a meaningful balanced
setting for this dataset: at the pilot medians, its token component is about
300 times smaller than its step component. Recompute this calibration from
the completed 1,000-task preference data before treating 12 or 600 as more
than diagnostic starting points.

### Stage 2B: learning rate

For the leading bonus configuration, compare:

| Trial suffix | `DEPO_LEARNING_RATE` |
| --- | ---: |
| `lr1e5` | `1e-5` |
| `lr2e5` | `2e-5` |

Reuse the existing `2e-5` trial from Stage 2A. Promote the best two overall
DEPO configurations to 200 tasks.

Only test `DEPO_BETA=0.1` or `0.4` if the learning-rate comparison remains
ambiguous or training diagnostics show sigmoid saturation. Do not construct a
full beta-by-learning-rate-by-alpha grid.

## Promotion and stopping rules

At 100 tasks:

- eliminate incomplete, unscored, or telemetry-incomplete runs;
- eliminate runs more than three resolved tasks below the baseline;
- promote at most two configurations per stage;
- when results differ by only one or two tasks, prefer promotion to 200 over
  a confident ranking claim.

At 200 tasks:

- apply the same three-percentage-point success constraint;
- choose the lowest-token eligible configuration after inspecting paired
  token and step deltas;
- promote the winner and at most one genuine runner-up to 500.

At 500 tasks:

- evaluate SFT, the selected DMPO model, and the selected DMPO-to-DEPO model;
- optionally include one runner-up only when the 200-task result was
  genuinely ambiguous;
- use exactly the same single-rollout deterministic policy used at 100 and
  200 for the primary comparison;
- run a separate multi-sample stochastic evaluation only as a robustness
  analysis, never mix it into the deterministic ranking table.

Stop expanding the search when a candidate is success-noninferior and clearly
better on total tokens per resolved task at 200, or when added trials repeatedly
change the winner only by small, noisy validation differences. Repeatedly
optimizing a small validation set increases selection bias
([Cawley and Talbot, 2010](#references)).

## Cloud command templates

The examples assume the completed 1,000-task artifacts use
`RUN_NAME=swesmith-train-1000-r2`.

In each new Cloud shell, first load the tracked defaults and any ignored
machine-local overrides. This defines `DEBUG_DEPO_SCRATCH` and the persistent
`UV` executable used below:

```bash
source cloud/env.sh
```

Train a DMPO trial:

```bash
RUN_NAME=swesmith-train-1000-r2 \
EXPERIMENT_ARM=dmpo \
DMPO_TRIAL_NAME=g07-lr1e6-b01-ga16 \
DMPO_GAMMA=0.7 \
DMPO_LEARNING_RATE=1e-6 \
DMPO_BETA=0.1 \
MAX_LENGTH=8192 \
PER_DEVICE_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=16 \
EPOCHS=3 \
  bash cloud/run.sh dmpo
```

Train a DEPO paper-hyperparameter trial from the selected DMPO package:

```bash
RUN_NAME=swesmith-train-1000-r2 \
EXPERIMENT_ARM=dmpo-depo \
DMPO_TRIAL_NAME=<selected-dmpo-trial> \
DEPO_TRIAL_NAME=paper-a2-a2-lr2e5-ga16 \
DEPO_TOKEN_METRIC=completion_tokens \
ALPHA_TOKENS=2 \
ALPHA_STEPS=2 \
DEPO_LEARNING_RATE=2e-5 \
DEPO_BETA=0.2 \
MAX_LENGTH=8192 \
PER_DEVICE_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=16 \
EPOCHS=3 \
  bash cloud/run.sh depo
```

Evaluate one packaged model on the 100-task budget:

```bash
TRAIN_RUN_NAME=swesmith-train-1000-r2
TRIAL_NAME=g07-lr1e6-b01-ga16
MODEL_PATH="$DEBUG_DEPO_SCRATCH/runs/$TRAIN_RUN_NAME/experiments/dmpo/$TRIAL_NAME/model"

RUN_NAME="validation-100-dmpo-$TRIAL_NAME" \
TASK_IDS_FILE=data/splits/swesmith_validation_100_instance_ids.txt \
MODEL_PATH="$MODEL_PATH" \
  bash cloud/run.sh validate
```

`validate` infers the expected count from the task-ID file, runs each task once
at temperature 0, and defaults to a 32,768-token context and 200 steps. For 200
or 500 tasks, change the run name and task-ID file together. Never reuse one
`RUN_NAME` for a different model or task budget.

For single-rollout SWE-smith analyses, compare the generated
`analysis/rollouts.csv` files:

```bash
"$UV" run debug-depo-compare-preference-arms \
  --baseline sft=<sft-run>/analysis/rollouts.csv \
  --arm dmpo=<dmpo-run>/analysis/rollouts.csv \
  --arm dmpo-depo=<depo-run>/analysis/rollouts.csv \
  --expected-tasks 100 \
  --success-tolerance 0.03 \
  --output results/preference-sweep-100.json
```

The comparison command rejects duplicate or mismatched task matrices,
unscored outcomes, inconsistent resolution labels, and missing step or token
telemetry. Use `--allow-incomplete-telemetry` only for explicitly exploratory
summaries; incomplete total-token arms remain ineligible for selection.

## Trial record

Generate the experiment ledger after each comparison:

```bash
TRAIN_RUN_ROOT="$DEBUG_DEPO_SCRATCH/runs/swesmith-train-1000-r2"

"$UV" run debug-depo-build-trial-record \
  --run-root "$TRAIN_RUN_ROOT" \
  --comparison 100=results/preference-sweep-100.json \
  --comparison 200=results/preference-sweep-200.json \
  --comparison 500=results/preference-sweep-500.json
```

The default output is:

```text
<training-run-root>/experiments/trial-record.csv
```

The command scans every `training/trial_config.json` under the run's
`experiments/` tree and adds completion, lineage, data hash, optimizer,
objective, effective-batch, global-step, and package fields. Each supplied
comparison adds separate resolution, cost, paired-delta, eligibility, rank,
selection, and gained/lost-success columns for its 100, 200, or 500 budget.
The CSV also records each trial's latest evaluated budget and whether its
latest result was `selected`, `eligible_not_selected`, or `ineligible`.

More than one comparison may be supplied for the same budget when DMPO and
DEPO were compared in separate stages. A trial reused as the baseline in a
later comparison retains its earlier candidate result; baseline arms are
otherwise treated as comparison context rather than separate trial results.
The command normally joins an evaluation to a trial through the packaged
model path saved in its collection manifest. It falls back to an exact
canonical trial ID such as
`dmpo/g07-lr1e6-b01-ga16` or
`dmpo-depo/g07-lr1e6-b01-ga16/paper-a2-a2-lr2e5-ga16`, then to an unambiguous
leaf trial name.

For pulled artifacts whose original evaluation paths are unavailable and
whose arm labels do not match trial names, provide an explicit mapping:

```bash
"$UV" run debug-depo-build-trial-record \
  --run-root "$TRAIN_RUN_ROOT" \
  --comparison 100=results/preference-sweep-100.json \
  --arm-trial dmpo=dmpo/g07-lr1e6-b01-ga16
```

Regenerating the command atomically replaces only `trial-record.csv`. The
training, package, evaluation, and comparison artifacts remain the underlying
sources of truth.

## What is research-backed versus local

| Choice | Evidence status |
| --- | --- |
| DMPO loss, turn weighting, and tuning `beta`/`gamma` | DMPO paper and authors' official implementation |
| Smaller gamma for noisier losing trajectories | DMPO paper finding |
| Unpaired desirable/undesirable objective | KTO paper |
| Desirable-only inverse token/step DEPO bonus | DEPO paper |
| DEPO `beta=0.2`, LR `2e-5`, three epochs, coefficients `2,2` | DEPO paper experiment |
| Joint versus single-component alpha ablation | DEPO paper experiment |
| 100→200→500 resource funnel | Local manual adaptation of multi-fidelity racing/Hyperband |
| Gamma values `0.7,0.9,0.99` | Narrow local subset of the DMPO paper's search range |
| DMPO LR values `5e-7,1e-6,2e-6` | Local LoRA sweep around the repository default |
| Accumulation 16 | Local update-budget heuristic |
| Completion-balanced `12,2` | Local pilot-median calibration |
| Total-balanced `600,2` | Local deployment-cost extension |
| Three-percentage-point success tolerance | Local operational constraint |

## References

1. Wentao Shi, Mengqi Yuan, Junkang Wu, Qifan Wang, and Fuli Feng. 2024.
   [Direct Multi-Turn Preference Optimization for Language
   Agents](https://aclanthology.org/2024.emnlp-main.138/). EMNLP 2024. See also
   the authors' [official implementation](https://github.com/swt-user/DMPO/blob/main/fastchat/train/dmpo_trainer_efficient.py),
   which includes the leading `gamma^t` used by this repository.
2. Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning,
   Stefano Ermon, and Chelsea Finn. 2023.
   [Direct Preference Optimization: Your Language Model is Secretly a Reward
   Model](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html).
   NeurIPS 2023.
3. Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe
   Kiela. 2024.
   [Model Alignment as Prospect Theoretic
   Optimization](https://proceedings.mlr.press/v235/ethayarajh24a.html).
   ICML 2024.
4. Sirui Chen, Mengshi Zhao, Lei Xu, Yuying Zhao, Beier Zhu, Hanwang Zhang,
   Shengjie Zhao, and Chaochao Lu. 2026.
   [DEPO: Dual-Efficiency Preference Optimization for LLM
   Agents](https://ojs.aaai.org/index.php/AAAI/article/view/40279). AAAI 2026,
   40(36):30279-30287.
5. Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet
   Talwalkar. 2018.
   [Hyperband: A Novel Bandit-Based Approach to Hyperparameter
   Optimization](https://jmlr.org/papers/v18/16-558.html). JMLR 18.
6. Gavin C. Cawley and Nicola L. C. Talbot. 2010.
   [On Over-fitting in Model Selection and Subsequent Selection Bias in
   Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html).
   JMLR 11.
