# Compute-light DMPO and DEPO hyperparameter sweep

This document defines the final bounded hyperparameter-search protocol for the
1,000-task SWE-smith preference-training experiment. It retains the
paper-aligned configurations and tests only a small number of changes with a
clear rationale. There is no initial 100-task round and no broad learning-rate,
beta or epoch grid. The immutable task memberships and sampling policies are
documented in [Dataset splits](dataset-splits.md).

## Compute budget

```text
Screen every candidate once on 200 tasks
                  |
                  v
Select one DMPO and one DMPO -> DEPO model
                  |
                  v
Validate those two models on 500 tasks
```

The plan requires at most six training runs:

- three DMPO configurations;
- three DEPO configurations, all initialized from the selected DMPO model.

At 500 tasks, evaluate only the selected DMPO and selected DMPO-to-DEPO models.
Evaluate SFT on the same confirmatory 500-task membership so every comparison
uses an identical task matrix.

## Fixed settings

Do not vary these settings:

| Setting | Value |
| --- | ---: |
| Training tasks | 1,000 |
| Base model | `Kwai-Klear/Klear-AgentForge-8B-SFT` |
| Collection context length | 32,768 for every rollout |
| Training sequence length | 8,192 for DMPO and DEPO |
| Screening evaluation context length | 32,768 for every arm |
| Confirmatory evaluation context length | 65,536 for every arm |
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation | 32 |
| LoRA rank/alpha | 64/128 |
| Seed | 42 |
| Evaluation rollouts | 1 per task |
| Evaluation temperature | 0.0 |
| Maximum evaluation steps | 200 |

Use the fixed repository split files:

- screening: `data/splits/swesmith_validation_200_instance_ids.txt`;
- final validation:
  `data/splits/swesmith_validation_confirmatory_balanced_500_instance_ids.txt`.

Build the preference data once and reuse it for every trial.

With eight training processes, per-device batch size 1 and gradient
accumulation 32, the nominal global batch is 256 sequences. Record the actual
number of DMPO pairs and DEPO trajectories and calculate the approximate update
count as:

```text
updates per epoch = ceil(number of training rows / 256)
```

This makes the limited optimiser-step budget explicit rather than treating
epochs alone as a measure of training compute.

## Controls and provenance

Before tuning:

1. Build and validate one immutable set of DMPO and DEPO preference artifacts.
2. Evaluate the unmodified SFT model on the 200-task screening membership.
3. Give every configuration a unique `DMPO_TRIAL_NAME` or `DEPO_TRIAL_NAME`.
4. Reuse the preference artifacts for trainer-only comparisons; any change to
   pair filtering must produce a separately named artifact with its own summary
   and hash.

At each evaluation stage, keep task membership, temperature, rollout count,
context length, step limit and scoring pipeline identical across arms.

## Stage 1: three DMPO trials

Keep the learning rate and beta fixed. Test only `gamma`, the DMPO-specific
parameter controlling how strongly later agent turns are discounted. The DMPO
study searches a wider `beta`/`gamma` range and reports that smaller `gamma` can
reduce the influence of noisy later actions, whereas larger `gamma` preserves
more influence from later actions in cleaner trajectories
([Shi et al., 2024](#references)). The three values below are a narrow local
comparison rather than a reproduction of that full search.

| Trial | Learning rate | `beta` | `gamma` | Reason |
| --- | ---: | ---: | ---: | --- |
| `g07-paper-informed` | `1e-6` | `0.1` | `0.7` | Current paper-informed default |
| `g09-late-turns` | `1e-6` | `0.1` | `0.9` | Gives later actions more influence |
| `g05-early-turns` | `1e-6` | `0.1` | `0.5` | Follow-up after `0.7` dominated `0.9`; tests stronger suppression of noisy later actions |

Evaluate all three on the 200-task split and select one. Do not test other
learning rates, beta values, epoch counts, or cost-ratio thresholds unless all
three runs are clearly broken.

## Stage 2: three DEPO trials

Initialize all three trials from the selected DMPO package. Keep
`DEPO_BETA=0.2`, `DEPO_LEARNING_RATE=2e-5`, and three epochs fixed.

DEPO augments its unpaired desirable/undesirable objective with a
desirable-only efficiency bonus:

```text
bonus = alpha_tokens / tokens_per_step + alpha_steps / steps
```

The paper reports `beta=0.2`, learning rate `2e-5`, three epochs and joint
coefficients `(2, 2)` for its Qwen2.5-7B experiment
([Chen et al., 2026](#references)). Its unpaired binary-label foundation follows
KTO ([Ethayarajh et al., 2024](#references)).

| Trial | Token metric | `alpha_tokens` | `alpha_steps` | Reason |
| --- | --- | ---: | ---: | --- |
| `paper-a2-a2` | `completion_tokens` | `2` | `2` | Closest paper-hyperparameter replication |
| `completion-balanced` | `completion_tokens` | `12` | `2` | Gives token and step bonuses similar scale on the pilot |
| `total-balanced` | `total_tokens` | `600` | `2` | Tests the local total deployment-cost objective |

The first trial is the main paper comparison. The other two answer useful local
questions without creating a large grid. They are engineering variants, not
paper settings.

The local coefficients 12 and 600 are rounded scale calibrations from the 111
successful trajectories in the eight-rollout pilot. With median inverse
completion-tokens-per-step `0.0071399`, inverse total-tokens-per-step
`0.00015098`, inverse steps `0.0454545`, and `alpha_steps=2`, equal median bonus
contributions imply coefficients of approximately 12.7 and 602, respectively.
This calibration explains the tested scales but does not make them
paper-derived hyperparameters.

Evaluate all three on the same 200 tasks and select one DMPO-to-DEPO model. Do
not separately sweep DEPO learning rate or beta.

## Selection rule at 200 tasks

Use SFT as the common baseline:

```text
minimize total_tokens_per_resolved_task
subject to resolution_rate >= SFT resolution_rate - 0.03
```

On 200 tasks, the success tolerance allows at most six fewer resolved tasks than
SFT. This is an operational threshold, not a confidence interval. For every
comparison:

1. Require the exact same instance-ID matrix and complete scoring.
2. Reject configurations outside the success constraint or with incomplete
   token or interaction-step telemetry.
3. Rank eligible configurations by total tokens per resolved task.
4. Inspect paired prompt-token, completion-token, total-token and
   interaction-step differences rather than relying only on aggregate means.
5. Inspect the numbers of SFT successes gained and lost.

When few tasks are resolved, cost per resolution is unstable. Treat small
differences as screening evidence rather than a precise ranking.

If two runs are effectively tied, choose the simpler paper/default setting
instead of adding more trials:

- prefer DMPO `gamma=0.7`;
- prefer DEPO `paper-a2-a2`.

## Final validation on 500 tasks

Validate exactly these two preference-trained candidates:

1. the selected DMPO model;
2. the selected DMPO-to-DEPO model.

Compare both with the SFT result on the identical 500-task matrix. Apply
the same three-percentage-point constraint, which permits at most fifteen fewer
resolved tasks than SFT. Report resolution rate, total tokens per resolved task,
steps, and gained/lost SFT successes.

The 500-task result is the confirmatory validation result. Do not use it to
launch a new hyperparameter round. Repeated optimisation against a fixed
validation set increases selection bias ([Cawley and Talbot, 2010](#references)).

## Command templates

In each new Cloud shell, first load the tracked defaults and local path
overrides:

```bash
source cloud/env.sh
```

DMPO example:

```bash
RUN_NAME=swesmith-train-1000-r2 \
EXPERIMENT_ARM=dmpo \
DMPO_TRIAL_NAME=g07-paper-informed \
DMPO_GAMMA=0.7 \
DMPO_LEARNING_RATE=1e-6 \
DMPO_BETA=0.1 \
MAX_LENGTH=8192 \
PER_DEVICE_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=32 \
EPOCHS=3 \
  bash cloud/run.sh dmpo
```

DEPO paper-setting example:

```bash
RUN_NAME=swesmith-train-1000-r2 \
EXPERIMENT_ARM=dmpo-depo \
DMPO_TRIAL_NAME=<selected-dmpo-trial> \
DEPO_TRIAL_NAME=paper-a2-a2 \
DEPO_TOKEN_METRIC=completion_tokens \
ALPHA_TOKENS=2 \
ALPHA_STEPS=2 \
DEPO_LEARNING_RATE=2e-5 \
DEPO_BETA=0.2 \
MAX_LENGTH=8192 \
PER_DEVICE_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=32 \
EPOCHS=3 \
  bash cloud/run.sh depo
```

For screening, validate with the 200-task file and a 32K context:

```bash
RUN_NAME=<unique-validation-run-name> \
TASK_IDS_FILE=data/splits/swesmith_validation_200_instance_ids.txt \
MODEL_PATH=<packaged-model-path> \
CONTEXT_LENGTH=32768 \
  bash cloud/run.sh validate
```

For final validation, change the unique run name and task file to
`swesmith_validation_confirmatory_balanced_500_instance_ids.txt`, and set
`CONTEXT_LENGTH=65536`. Keep the temperature, rollout count, and step limit
unchanged.

Compare the completed analysis matrices with the repository command, adapting
the arm list and expected count to the stage:

```bash
"$UV" run debug-depo-compare-preference-arms \
  --baseline sft=<sft-run>/analysis/rollouts.csv \
  --arm dmpo=<dmpo-run>/analysis/rollouts.csv \
  --arm dmpo-depo=<depo-run>/analysis/rollouts.csv \
  --expected-tasks 200 \
  --success-tolerance 0.03 \
  --output results/preference-sweep-200.json
```

The command rejects duplicate or mismatched task matrices, unscored outcomes,
inconsistent resolution labels, and missing token or step telemetry. Retain the
comparison output alongside the trial configurations and immutable data hashes.

## Summary

This reduced plan tests the most defensible choices while avoiding combinatorial
search:

- three values for DMPO's defining `gamma` parameter;
- one DEPO paper setting;
- one locally balanced generated-token setting;
- one locally balanced total-cost setting;
- one 200-task screening round;
- two preference-trained models on the final 500-task validation.

## Remaining-budget follow-up after confirmatory validation

The confirmatory 500-task evaluation did not show an aggregate improvement over
SFT: SFT resolved 70 tasks, DMPO resolved 61, and DMPO-to-DEPO resolved 58.
The only promising exploratory signal was lower mean and upper-tail token cost
on the small subset of tasks solved by both models. The following three trials
are therefore ordered to prioritize preserving SFT capability while testing
the cheapest plausible explanations for the regression.

This section supersedes the fixed accumulation, epoch, and learning-rate values
above only for these follow-up trials. The existing 500-task result has now been
inspected and must not be treated as untouched confirmation data for the new
trials.

### Follow-up order

Run the trials in this order:

| Priority | Trial | Initialization and data | Settings | Purpose |
| ---: | --- | --- | --- | --- |
| 1 | `g07-lr5e7-ga16-e2` | SFT; reuse the existing DMPO pairs | `gamma=0.7`, `beta=0.1`, LR `5e-7`, accumulation `16`, 2 epochs, length `8192` | Apply a gentler update while retaining about 40 optimizer steps instead of the original 30 |
| 2 | `g07-lr5e7-ga16-e2-min125` | SFT; rebuild DMPO pairs into a separately named artifact with `MIN_COST_RATIO=1.25` | Same trainer settings as trial 1 | Remove weak 10--25% cost-gap preferences and test whether cleaner efficiency supervision preserves success |
| 3 | `direct-total-balanced-lr1e5-ga16-e2` | Direct SFT-to-DEPO; reuse the existing DEPO trajectories | `beta=0.2`, `DEPO_TOKEN_METRIC=total_tokens`, `alpha_tokens=600`, `alpha_steps=2`, LR `1e-5`, accumulation `16`, 2 epochs, length `8192` | Test whether DEPO can retain capability when it is not stacked on an already degraded DMPO model |

Do not add gamma, beta, LoRA-rank, longer-context, or 5,000-task trials until
these three have been screened. The full-data calibration gives a balanced
total-token coefficient of approximately 622, so changing 600 to 622 is not a
useful use of the remaining budget.

### Budget-saving screening and stopping rule

Evaluate each new model on the 100-task development split first. Reject a trial
if it resolves more than three fewer tasks than SFT. Among eligible trials,
rank by `total_tokens_per_resolved_task`, then inspect paired total-token changes
on tasks solved by both the candidate and SFT.

If trial 1 is success-noninferior and clearly improves cost per resolution, it
is acceptable to stop the search and promote it directly to the 200-task split.
Otherwise screen trials 2 and 3 on 100 tasks and promote only the best eligible
candidate to 200. Do not run all three on 200 tasks by default.

Use:

```text
data/splits/swesmith_validation_100_instance_ids.txt
```

for the first screen and:

```text
data/splits/swesmith_validation_200_instance_ids.txt
```

for the single promoted run. If a new confirmatory claim is required later,
draw a fresh repository-balanced membership from the remaining held-out tasks
rather than selecting again on the existing confirmatory 500.

### Trial 1 command

```bash
RUN_NAME=swesmith-train-1000-r2 \
EXPERIMENT_ARM=dmpo \
DMPO_TRIAL_NAME=g07-lr5e7-ga16-e2 \
DMPO_GAMMA=0.7 \
DMPO_LEARNING_RATE=5e-7 \
DMPO_BETA=0.1 \
MAX_LENGTH=8192 \
PER_DEVICE_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=16 \
EPOCHS=2 \
  bash cloud/run.sh dmpo
```

### Trial 2 data and training settings

Build the `MIN_COST_RATIO=1.25` pairs into a new directory and retain their own
summary and hash; do not overwrite `preference-data/dmpo`. Point
`DMPO_DATA_PATH` at the new `pairs.jsonl`, then train with the same settings as
trial 1 and the unique trial name `g07-lr5e7-ga16-e2-min125`.

Do not set `INCLUDE_FAILURE_EFFICIENCY_PAIRS=1` for this run. Unfiltered
failure-efficiency pairs can reward premature termination.

### Trial 3 command

```bash
RUN_NAME=swesmith-train-1000-r2 \
EXPERIMENT_ARM=depo \
DEPO_TRIAL_NAME=direct-total-balanced-lr1e5-ga16-e2 \
DEPO_TOKEN_METRIC=total_tokens \
ALPHA_TOKENS=600 \
ALPHA_STEPS=2 \
DEPO_LEARNING_RATE=1e-5 \
DEPO_BETA=0.2 \
MAX_LENGTH=8192 \
PER_DEVICE_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=16 \
EPOCHS=2 \
  bash cloud/run.sh depo
```

## Research-backed and local choices

| Choice | Evidence status |
| --- | --- |
| DMPO turn weighting and tuning `beta`/`gamma` | DMPO paper and authors' implementation |
| Smaller `gamma` for noisier losing trajectories | DMPO paper finding |
| Unpaired desirable/undesirable foundation | KTO paper |
| Desirable-only inverse token/step bonus | DEPO paper |
| DEPO `beta=0.2`, LR `2e-5`, three epochs and coefficients `(2, 2)` | DEPO paper experiment |
| Gamma values `0.5`, `0.7`, `0.9` | Narrow local comparison |
| Completion-balanced `(12, 2)` | Local pilot-scale calibration |
| Total-balanced `(600, 2)` | Local total-token objective and pilot-scale calibration |
| One 200-task screen followed by 500-task confirmation | Local compute-budget design |
| Three-percentage-point success tolerance | Local operational constraint |
| Remaining-budget follow-up trials | Post-hoc engineering tests; not fresh confirmation |

## References

1. Wentao Shi, Mengqi Yuan, Junkang Wu, Qifan Wang, and Fuli Feng. 2024.
   [Direct Multi-Turn Preference Optimization for Language
   Agents](https://aclanthology.org/2024.emnlp-main.138/). EMNLP 2024. See also
   the authors' [official implementation](https://github.com/swt-user/DMPO/blob/main/fastchat/train/dmpo_trainer_efficient.py).
2. Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe
   Kiela. 2024. [Model Alignment as Prospect Theoretic
   Optimization](https://proceedings.mlr.press/v235/ethayarajh24a.html).
   ICML 2024.
3. Sirui Chen, Mengshi Zhao, Lei Xu, Yuying Zhao, Beier Zhu, Hanwang Zhang,
   Shengjie Zhao, and Chaochao Lu. 2026. [DEPO: Dual-Efficiency Preference
   Optimization for LLM
   Agents](https://ojs.aaai.org/index.php/AAAI/article/view/40279). AAAI 2026,
   40(36):30279--30287.
4. Gavin C. Cawley and Nicola L. C. Talbot. 2010. [On Over-fitting in Model
   Selection and Subsequent Selection Bias in Performance
   Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html). JMLR 11.
