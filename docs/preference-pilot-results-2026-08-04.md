# DMPO and DEPO pilot results: 2026-08-04

This report records the 64-task validation of the SFT baseline, the DMPO
pilot, and the sequential DMPO-to-DEPO pilot trained from the evaluated
1,000-task SWE-smith collection. It is a snapshot of the pulled artifacts,
not a final model-performance claim.

## Outcome

The end-to-end preference-training pipeline passed. Under the repository's
success-constrained efficiency rule, the selected pilot arm is
**DMPO-to-DEPO**:

- it resolved 5 of 64 tasks, compared with 4 for SFT and 3 for DMPO;
- it used 11.9% fewer total tokens and 7.2% fewer action steps than SFT;
- it retained all four SFT successes and gained one additional success; and
- its total token spend per resolution was 29.5% lower than SFT.

The result is suitable for deciding whether to proceed to full training. It
is not sufficient to conclude that DEPO is statistically better: only a few
tasks were resolved, and each pilot trainer made one optimizer update. The
DMPO pairs used the accepted `min_cost_ratio=1.1`, requiring a 10% resolved
trajectory cost gap, so no preference-data rebuild is required for the full
run.

## Artifact provenance

The source training run is:

```text
scratch/cloud/runs/swesmith-train-1000-r2
```

The authoritative comparison is:

```text
scratch/cloud/runs/swesmith-train-1000-r2/
  experiments/pilot-validation-64-comparison.json
```

The three validation runs are:

```text
scratch/cloud/runs/swesmith-train-1000-r2-pilotval64-sft
scratch/cloud/runs/swesmith-train-1000-r2-pilotval64-dmpo-pilot64-g07-8k-r103
scratch/cloud/runs/swesmith-train-1000-r2-pilotval64-depo-pilot64-paper-a2-a2-8k-r103
```

The comparison verifies an exact 64-instance-ID match with task-matrix hash:

```text
578811f5f5d23068e363439679809343c99b1803927e059811aa0e00b93188d0
```

All 64 tasks in every arm were collected and evaluated, and all token and
step metrics have 100% coverage. The earlier status-143 shard terminations
were successfully retried and did not leave missing or unscored tasks.

## Source collection

The preference data came from a complete collection of 4 rollouts for each
of 1,000 SWE-smith tasks:

| Metric | Result |
| --- | ---: |
| Evaluated trajectories | 4,000 |
| Resolved trajectories | 1,449 (36.225%) |
| Tasks resolved at least once | 524/1,000 (52.4%) |
| Mixed-temperature pass@1 | 36.225% |
| Mixed-temperature pass@2 | 45.45% |
| Mixed-temperature pass@3 | 49.875% |
| Mixed-temperature pass@4 | 52.4% |

Temperature did not materially affect the source resolution rate: the two
rollouts at temperature 0.6 resolved 36.15%, while the two at temperature 0.7
resolved 36.30%.

Across all 4,000 trajectories, the mean trajectory used 37.68 steps, 6,494
completion tokens, and 556,469 total tokens. Resolved trajectories were
shorter on average: 27.80 steps, 4,462 completion tokens, and 286,815 total
tokens.

## Preference artifacts

### DMPO

The DMPO builder produced 2,469 pairs across 523 tasks:

| Preference reason | Pairs |
| --- | ---: |
| Resolved over non-resolved (`task_success`) | 1,107 |
| Cheaper resolved trajectory (`resolved_token_efficiency`) | 1,362 |

The artifact hash used by the pilot was:

```text
07c36eca345d2307f59bc6483f1529e9b980fca543138f67c06e263de743725b
```

The summary records `token_metric=total_tokens` and the accepted
`min_cost_ratio=1.1`, requiring a 10% efficiency gap between resolved
trajectories before creating a cost-based pair.

### DEPO

The DEPO builder labelled all 4,000 trajectories:

| Label | Rows |
| --- | ---: |
| Desirable | 1,449 |
| Undesirable | 2,551 |

The DEPO trajectory artifact hash was:

```text
d7d0ce49de0800aead7382459a6714a10993d542b688394ff2132873b534ad01
```

The DMPO minimum cost ratio does not affect these binary DEPO labels.

## Pilot training configuration

Both pilots trained on a deterministic seed-42 sample of 64 rows with an 8K
maximum sequence length. The common distributed settings were 8 processes,
per-device batch size 1, gradient accumulation 16, BF16, gradient
checkpointing, one epoch, and LoRA rank 64. The effective accumulated batch
was 128 examples, so each training run completed exactly one optimizer step.

The stage-specific settings were:

| Setting | DMPO | DEPO |
| --- | ---: | ---: |
| Learning rate | `1e-6` | `2e-5` |
| Beta | `0.1` | `0.2` |
| Gamma | `0.7` | Not applicable |
| Token metric | `total_tokens` | `completion_tokens` |
| Alpha tokens | Not used by loss | `2` |
| Alpha steps | Not used by loss | `2` |

The DEPO trial correctly initialized from the packaged DMPO pilot model. Both
training manifests report `global_steps=1`, and both stages have standalone
model package manifests. The successful 8K trials are separate from the
preserved failed 16K and 32K attempts.

## Validation results

`Total tokens per resolution` is the token spend of all 64 attempts divided
by the number of resolved tasks. It penalizes an arm that is cheap because it
terminates early without solving tasks.

| Arm | Resolved | Mean steps | Mean completion tokens | Mean total tokens | Total tokens per resolution |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT | 4/64 (6.25%) | 37.83 | 6,803 | 540,584 | 8.65M |
| DMPO | 3/64 (4.69%) | 35.45 | 5,916 | 492,524 | 10.51M |
| DMPO-to-DEPO | **5/64 (7.81%)** | **35.11** | **5,907** | **476,258** | **6.10M** |

### Aggregate changes from SFT

| Arm | Resolution-rate change | Step change | Completion-token change | Total-token change | Tokens-per-resolution change |
| --- | ---: | ---: | ---: | ---: | ---: |
| DMPO | -1.5625 pp | -6.28% | -13.04% | -8.89% | +21.48% |
| DMPO-to-DEPO | +1.5625 pp | -7.19% | -13.17% | -11.90% | -29.52% |

Compared directly with DMPO, the DEPO stage resolved two additional tasks,
used 0.97% fewer steps, and used 3.30% fewer total tokens. Its token spend per
resolution was 41.98% lower than DMPO's.

Collection and evaluation outcomes were:

| Arm | Completed | Model terminated | Empty patch | Patch failed | Resolved | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT | 48 | 16 | 16 | 4 | 4 | 40 |
| DMPO | 52 | 12 | 12 | 2 | 3 | 47 |
| DMPO-to-DEPO | 52 | 12 | 12 | 3 | 5 | 44 |

## Paired task analysis

Resolution transitions against SFT were:

| Arm | Both resolved | Gained | Lost | Both unresolved |
| --- | ---: | ---: | ---: | ---: |
| DMPO | 2 | 1 | 2 | 59 |
| DMPO-to-DEPO | 4 | 1 | 0 | 59 |

DEPO retained all three DMPO successes and added two tasks that SFT had solved
but DMPO had lost. Thus, the sequential stage restored those two regressions
without sacrificing a DMPO success.

On the four tasks resolved by both SFT and DEPO, DEPO averaged:

- 8.25 fewer action steps;
- 2,219 fewer completion tokens; and
- 122,762 fewer total tokens.

Three of these four shared successes were cheaper under DEPO. The individual
total-token deltas were `-302,215`, `-194,485`, `-16,608`, and `+22,260`.

The efficiency change was not uniform over the whole matrix. DMPO used fewer
total tokens on 38 tasks and more on 26. DEPO used fewer on 35 and more on 29.
The lower DEPO aggregate is therefore partly driven by several large savings,
not by a small improvement on every task.

## Uncertainty and interpretation

Approximate 95% Wilson intervals for the resolution rates are:

| Arm | Resolution rate | 95% interval |
| --- | ---: | ---: |
| SFT | 6.25% | 2.46%-15.00% |
| DMPO | 4.69% | 1.61%-12.90% |
| DMPO-to-DEPO | 7.81% | 3.38%-17.02% |

These intervals overlap substantially. With only 3-5 successes per arm, the
cost-per-resolution metric is also sensitive to a single task transition.
The pilot supports the following conclusions:

1. Collection, evaluation, preference training, packaging, sequential DEPO
   initialization, and exact-matrix comparison work end to end.
2. The tested DMPO-to-DEPO configuration shows a positive scaling signal and
   is the pilot winner under the configured selection rule.
3. The result does not establish a statistically reliable model ranking.
4. A full training run followed by the fixed 100/200/500-task validation
   funnel is still required.

## Threshold settings

Two independent thresholds appear in these artifacts:

- `min_cost_ratio` controls whether two successful trajectories are far
  enough apart in cost to create a DMPO efficiency pair. The accepted
  artifact uses `1.1`, or a 10% cost gap.
- `success_tolerance` controls the permitted resolution-rate drop during
  model selection. This pilot comparison used `0.02`, or two percentage
  points.

These settings serve different purposes and should not be interpreted as the
same threshold. The accepted 10% pair threshold does not require any change
to the current preference artifact or pilot lineage.
