# Compute-light DMPO and DEPO hyperparameter sweep

This is a smaller alternative to [the full sweep](hyperparameter-sweep.md). It
keeps the paper-aligned configurations and tests only a few changes with a clear
reason. There is no 100-task round and no large learning-rate, beta, or epoch
grid.

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
Reuse the existing SFT result on the same 500 tasks as the baseline. If that SFT
result does not exist, it must also be evaluated; otherwise success and cost
comparisons are not valid.

## Fixed settings

Do not vary these settings:

| Setting | Value |
| --- | ---: |
| Training tasks | 1,000 |
| Collection context length | 32,768 for every rollout |
| Training sequence length | 8,192 for DMPO and DEPO |
| Evaluation context length | 32,768 for every arm |
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
- final validation: `data/splits/swesmith_validation_500_instance_ids.txt`.

Build the preference data once and reuse it for every trial.

## Stage 1: three DMPO trials

Keep the learning rate and beta fixed. Test only `gamma`, the DMPO-specific
parameter controlling how strongly later agent turns are discounted.

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

| Trial | Token metric | `alpha_tokens` | `alpha_steps` | Reason |
| --- | --- | ---: | ---: | --- |
| `paper-a2-a2` | `completion_tokens` | `2` | `2` | Closest paper-hyperparameter replication |
| `completion-balanced` | `completion_tokens` | `12` | `2` | Gives token and step bonuses similar scale on the pilot |
| `total-balanced` | `total_tokens` | `600` | `2` | Tests the local total deployment-cost objective |

The first trial is the main paper comparison. The other two answer useful local
questions without creating a large grid. They are engineering variants, not
paper settings.

Evaluate all three on the same 200 tasks and select one DMPO-to-DEPO model. Do
not separately sweep DEPO learning rate or beta.

## Selection rule at 200 tasks

Use SFT as the common baseline:

```text
minimize total_tokens_per_resolved_task
subject to resolution_rate >= SFT resolution_rate - 0.03
```

On 200 tasks, the success tolerance allows at most six fewer resolved tasks than
SFT. Reject incomplete or telemetry-incomplete runs. Among eligible models,
choose the lowest total-token cost per resolved task and inspect the paired step
and token changes.

If two runs are effectively tied, choose the simpler paper/default setting
instead of adding more trials:

- prefer DMPO `gamma=0.7`;
- prefer DEPO `paper-a2-a2`.

## Final validation on 500 tasks

Validate exactly these two preference-trained candidates:

1. the selected DMPO model;
2. the selected DMPO-to-DEPO model.

Compare both with the cached SFT result on the identical 500-task matrix. Apply
the same three-percentage-point constraint, which permits at most fifteen fewer
resolved tasks than SFT. Report resolution rate, total tokens per resolved task,
steps, and gained/lost SFT successes.

The 500-task result is the final selection result. Do not use it to launch a new
hyperparameter round.

## Command templates

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

For final validation, change only the unique run name and task file to
`swesmith_validation_500_instance_ids.txt`. Keep the context, temperature,
rollout count, and step limit unchanged.

## Summary

This reduced plan tests the most defensible choices while avoiding combinatorial
search:

- three values for DMPO's defining `gamma` parameter;
- one DEPO paper setting;
- one locally balanced generated-token setting;
- one locally balanced total-cost setting;
- one 200-task screening round;
- two preference-trained models on the final 500-task validation.
