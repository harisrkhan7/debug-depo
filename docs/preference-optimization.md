# Preference optimization notes

This note records the efficiency objective, pilot findings, implementation
coverage, and paper-informed experiment plan. See the [project README](../README.md)
for commands and operational defaults.

## Objective

Treat deployment efficiency as constrained cost optimization:

```text
minimize  E[c_input * prompt_tokens
            + c_output * completion_tokens
            + c_call * steps]
subject to resolution_rate >= baseline_resolution_rate - delta
```

For local inference with equally valued tokens, set `c_input=c_output=1` and
`c_call=0`; the objective then reduces to accumulated `total_tokens`. For a
paid API, use its input, output, cached-input, and per-call costs.

Success remains the hard preference boundary: a resolved trajectory must beat
an unresolved trajectory regardless of cost, and efficiency is optimized
within successful trajectories. This follows DEPO's successful-trajectory-only
principle and avoids rewarding premature submission.

## Pilot findings

The selected four-rollout pilot contains 120 trajectories. Resolved
trajectories have medians of 23 steps and 151,473 total tokens; non-resolved
trajectories have medians of 35.5 steps and 391,184 total tokens. Step count
and total tokens have correlation 0.963, so reducing unnecessary interactions
is likely to reduce token cost substantially.

The pilot also exposes two calibration issues:

- Two of the 55 successful token-efficiency DMPO pairs prefer a trajectory
  with more steps. A step-aware Pareto rule would remove that conflicting
  supervision.
- With `DEPO_TOKEN_METRIC=total_tokens` and both alphas set to 2, the median
  token component of the desirable bonus is `0.000294`, versus `0.08696` for
  the step component. The nominally equal coefficients are not balanced for
  SWE-smith.

## Implementation coverage

| Recommendation | Status | Remaining work |
| --- | --- | --- |
| DMPO multi-turn weighting | Implemented | The trainer matches the authors' official code: `gamma^t * (1-gamma^(T-t))/(1-gamma^T)`. The displayed definition after equation 16 omits `gamma^t`, but that conflicts with the paper's `gamma -> 0` corollary and official implementation. |
| Success-first DMPO preferences | Implemented | Resolved beats unresolved; cost comparisons between failures are disabled by default. |
| Successful-trajectory cost pairs | Implemented | `total_tokens` and `completion_tokens` are selectable; `MIN_COST_RATIO` removes negligible differences. |
| DEPO/KTO objective | Implemented | Raw inverse tokens-per-step and inverse-step bonuses apply only to desirable trajectories. The repository labels every scored failure undesirable; the paper filters out low-quality failures. |
| Reproducible trials and lineage | Implemented | Configuration, data hash, parent DMPO trial, checkpoints, and packages are isolated and validated. |
| Gamma sweep | Configuration only | Run separately named trials for `gamma={0.7,0.9,0.99}`; no trainer change is required. |
| DEPO paper hyperparameters | Configuration only | Run `DEPO_TOKEN_METRIC=completion_tokens`, `ALPHA_TOKENS=2`, and `ALPHA_STEPS=2`; the data and environment remain repository-specific. |
| Full 65,536-token training/evaluation | Capacity test needed | `MAX_LENGTH` and `EVAL_CONTEXT_LENGTH` support it. Sixteen of 120 selected pilot trajectories exceed a 32K prompt history, so compare 32K and 65K if GPU memory permits. |
| Deployment-weighted trajectory cost | Partial | Prompt, completion, total tokens, and steps are recorded, but builders do not yet accept input/output/cached-token/per-call prices. |
| Step-aware DMPO pairs | Missing | Require the preferred successful trajectory to be Pareto-better in token cost and steps, or use an explicit composite cost for non-dominated cases. |
| High-quality losing trajectories | Missing | Exclude empty-patch, timeout, patch-application failure, and model-terminated trajectories by default; prefer evaluated unresolved near misses. |
| Cost-gap pair strength | Missing | Store a clipped `log(cost_ratio)` weight or target margin and consume it in the DMPO loss. SimPO supports explicit preference margins, but this would be another local extension. |
| Scale-normalized DEPO bonus | Missing | Normalize tokens-per-step and steps against successful reference values, preferably per task with repository/global fallbacks, and cap outlier ratios. |
| Cost-aware evaluation and checkpoint selection | Implemented | Both analysis paths report coverage plus mean, median, and p90 token/action-step distributions for all and resolved attempts. `debug-depo-compare-preference-arms` requires an exact scored task matrix and selects the lowest-token success-noninferior arm. |

## Proposed normalized DEPO variant

Use a dimensionless desirable-only bonus:

```text
b(trajectory) =
  lambda_tokens * clip(reference_tokens_per_step / tokens_per_step, 0, cap)
  + lambda_steps * clip(reference_steps / steps, 0, cap)
```

Compute references only from successful training trajectories. Prefer a
per-task successful median when enough rollouts exist, then fall back to a
repository or global successful median. The selected pilot's global medians
are approximately 6,808 total tokens per step and 23 steps.

A first diagnostic setting of `lambda_tokens=lambda_steps=0.1` gives both
terms the same median scale. In the current unnormalized parameterization this
is roughly equivalent to `ALPHA_TOKENS=680` and `ALPHA_STEPS=2.3`. Do not make
those raw values global defaults before implementing normalization and
validating it on held-out data.

## Paper-informed main experiment

1. Train the primary DMPO arm with `DMPO_GAMMA=0.7` and the implemented
   multi-turn loss.
2. Train the primary DEPO stage with
   `DEPO_TOKEN_METRIC=completion_tokens`, `ALPHA_TOKENS=2`, and
   `ALPHA_STEPS=2`, retaining desirable-only efficiency bonuses.
3. Evaluate baseline SFT, DMPO, and DMPO-to-DEPO on the same held-out 500-task
   split. Keep unique trial names and immutable lineage for every arm.
4. Select the lowest `total_tokens_per_resolved_task` arm whose resolution
   rate is within the declared tolerance. Inspect telemetry coverage and
   paired deltas before accepting the automatic ranking.
5. Defer token/step Pareto pairing, cost-gap margins, normalized DEPO bonuses,
   and deployment-price weighting until the paper-informed result is
   established.

The five-task notebook evaluation is a pipeline smoke test, not evidence for
model selection. Use the 30-task pilot to reject broken configurations and the
held-out 500-task evaluation for the final performance/cost decision.

## References

1. Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning,
   Stefano Ermon, and Chelsea Finn. 2023.
   [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html).
   NeurIPS 2023. Pairwise direct-preference foundation from which DMPO is
   derived.
2. Wentao Shi, Mengqi Yuan, Junkang Wu, Qifan Wang, and Fuli Feng. 2024.
   [Direct Multi-Turn Preference Optimization for Language Agents](https://aclanthology.org/2024.emnlp-main.138/).
   EMNLP 2024. Source of the multi-turn objective, length normalization,
   discount-factor analysis, and evidence for selecting high-quality losing
   trajectories. The authors' [official code](https://github.com/swt-user/DMPO/blob/main/fastchat/train/dmpo_trainer_efficient.py)
   resolves the paper's inconsistent presentation of `phi(t,T)`.
3. Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe
   Kiela. 2024.
   [Model Alignment as Prospect Theoretic Optimization](https://proceedings.mlr.press/v235/ethayarajh24a.html).
   ICML 2024. Source of the KTO desirable/undesirable objective and its
   reference-dependent value formulation used by DEPO.
4. Yu Meng, Mengzhou Xia, and Danqi Chen. 2024.
   [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e099c1c9699814af0be873a175361713-Abstract-Conference.html).
   NeurIPS 2024. Its length-normalized reward and target-margin results
   motivate the proposed cost-gap-margin ablation; that ablation is not part
   of DMPO.
5. Sirui Chen, Mengshi Zhao, Lei Xu, Yuying Zhao, Beier Zhu, Hanwang Zhang,
   Shengjie Zhao, and Chaochao Lu. 2026.
   [DEPO: Dual-Efficiency Preference Optimization for LLM Agents](https://ojs.aaai.org/index.php/AAAI/article/view/40279).
   AAAI 2026, 40(36):30279-30287. Source of the
   successful-trajectory-only efficiency principle,
   desirable-only inverse tokens-per-step and inverse-step bonus, balanced
   token/step ablations, and the undesirable-penalty ablation.

The deployment-priced cost function, token/step Pareto pairing,
reference-scale normalization, and success-constrained checkpoint selection
are engineering recommendations derived from this repository's SWE-smith
pilot measurements. Evaluate them as explicit extensions rather than
attributing them to the papers above.
