# Results and discussion: compute-light DMPO and DEPO sweep

## Results

### Main finding

Preference optimisation produced a benchmark-dependent efficiency outcome.
SFT achieved the best success-constrained efficiency in both the 200-task
SWE-smith screen and the 500-task confirmation. On SWE-bench Verified,
however, sequential DMPO-to-DEPO satisfied the pre-declared resolution
constraint and achieved the lowest observed token cost per resolution: 3.032M
compared with 3.105M for SFT, a reduction of 2.4%. DEPO was therefore selected
on the fixed SWE-bench Verified matrix, although this benchmark-specific result
does not establish a general efficiency advantage on new tasks or repositories.

### Hyperparameter screening

The 200-task screen required resolution of at least 8.5%, three percentage
points below SFT. Cost is the total token spend over **all** attempts divided by
the number resolved; this prevents cheap, prematurely terminated failures from
appearing efficient.

| Arm | Varied setting | Resolved | Mean total tokens | Tokens per resolution | Eligible |
| --- | --- | ---: | ---: | ---: | :---: |
| SFT | Baseline | **23/200 (11.5%)** | 544,596 | **4.74M** | Yes |
| **DMPO, `gamma=0.7`** | Paper-informed | **21/200 (10.5%)** | **542,602** | **5.17M** | Yes |
| DMPO, `gamma=0.9` | Later-turn emphasis | 19/200 (9.5%) | 551,688 | 5.81M | Yes |
| DMPO, `gamma=0.5` | Earlier-turn emphasis | 20/200 (10.0%) | 562,808 | 5.63M | Yes |
| DEPO paper | Completion, `alpha_tokens=2` | 17/200 (8.5%) | 589,251 | 6.93M | Yes |
| DEPO completion-balanced | Completion, `alpha_tokens=12` | 14/200 (7.0%) | 529,990 | 7.57M | No |
| **DEPO total-balanced** | Total, `alpha_tokens=600` | **21/200 (10.5%)** | **527,192** | **5.02M** | Yes |

![Resolution-cost trade-off for the 200-task screen](assets/hyperparameter-screening.svg)

SFT was the global winner. Within the trained model families, `gamma=0.7` was
the strongest DMPO setting and total-balanced was the strongest DEPO setting,
so these were promoted as planned. The screen does not support the hypothesis
that changing DMPO's turn discount alone improves performance: all three values
lost 1.0–2.0 percentage points relative to SFT. DEPO's total-token objective
was materially better than its paper and completion-balanced alternatives. It
reduced mean token use by 3.2% relative to SFT, but two fewer resolutions made
its cost per resolution 6.0% higher. This distinction between per-attempt
economy and useful work is central to the later results.

### SWE-smith confirmation

| SWE-smith arm | Resolved (95% Wilson CI) | Mean steps | Mean total tokens | Tokens per resolution | Change from SFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT | **70/500, 14.0% (11.2–17.3%)** | 40.89 | 0.678M | **4.84M** | — |
| DMPO (`gamma=0.7`) | 61/500, 12.2% (9.6–15.4%) | **40.52** | **0.655M** | 5.37M | +10.9% |
| DMPO→DEPO (total-balanced) | 58/500, 11.6% (9.1–14.7%) | 42.38 | 0.735M | 6.34M | +30.9% |

Both preference-trained arms met the *point-estimate* tolerance of 11%, but
neither improved the objective. Against SFT, DMPO gained 13 task successes and
lost 22; sequential DEPO gained 13 and lost 25. Exact paired McNemar tests did
not reject equal resolution probability (respectively, *p* = 0.176 and
*p* = 0.073), although absence of significance is not evidence of
non-inferiority. In 20,000 paired bootstrap resamples of these 500 evaluation
tasks, DEPO's 30.9% cost increase lay between **+4.2% and +66.5%** at 95%,
making the negative efficiency result more convincing than the resolution
comparison alone.

The 45 tasks solved by both SFT and DEPO were about 30,006 tokens cheaper under
DEPO on average, yet DEPO used 8.5% more tokens over the complete task matrix.
Thus, the attractive shared-success result is conditional on which tasks were
solved and does not translate into lower deployment cost.

### SWE-bench Verified test results

| Test arm | Resolved (95% Wilson CI) | Mean steps | Mean total tokens | Tokens per resolution | Change from SFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT | **196/500, 39.2% (35.0–43.6%)** | 58.46 | 1.217M | 3.105M | — |
| DMPO (`gamma=0.7`) | 182/500, 36.4% (32.3–40.7%) | 57.90 | **1.156M** | 3.176M | +2.3% |
| DMPO→DEPO (total-balanced) | 194/500, 38.8% (34.6–43.1%) | **57.41** | 1.176M | **3.032M** | **-2.4%** |

![Resolution and efficiency on the two 500-task evaluations](assets/hyperparameter-final-results.svg)

Sequential DEPO recovered most of the capability lost by DMPO. Relative to
SFT it gained 40 successes and lost 42, whereas DMPO gained 39 and lost 53.
The DEPO–SFT resolution difference was consequently only -0.4 percentage
points (exact McNemar *p* = 0.912). Across the 154 shared successes, DEPO used
1.83 fewer steps and 88,194 fewer total tokens on average. Over all tasks it
reduced mean tokens by 3.4%, producing the only preference-trained result that
ranked ahead of SFT under the pre-declared operational objective.

### Paired task-level efficiency results

The paired decomposition below reports `DEPO - SFT`, so negative values denote
savings. The four SWE-bench bars in panel A sum to the 20.42M aggregate token
saving. They do not sum to the 73,245-token reduction in cost per resolution,
which additionally depends on the different numbers of tasks resolved.

![Paired outcome decomposition and token use by no-patch attempts](assets/hyperparameter-token-delta-decomposition.svg)

Across the 154 SWE-bench tasks solved by both models, DEPO saves 88,194 tokens
per task on average, but the median paired difference is a 5,100-token increase
and only 72/154 attempts are cheaper. DEPO saves 18.24M tokens on its 40 unique
successes but adds 18.80M on the 42 SFT-only successes. The remaining aggregate
saving comes from the shared successes (-13.58M) and jointly unresolved tasks
(-7.40M).

The jointly unresolved group contains opposing subgroups. On SWE-bench, 237
patch-producing attempts save 38.73M tokens and 829 interaction steps in
aggregate, whereas 27 no-patch attempts add 31.34M tokens and 582 steps. On
SWE-smith, 399 patch-producing joint failures save 17.74M tokens, but 18
no-patch failures add 41.60M and make the aggregate *Neither* group costlier.

Panel B isolates the no-patch attempts using a field recorded consistently in
both result matrices. The 18 SWE-smith no-patch attempts comprise 3.6% of tasks
but consume 21.4% of DEPO's tokens and add 41.60M tokens relative to SFT on the
same tasks. On SWE-bench, 31 no-patch attempts comprise 6.2% of tasks but
consume 22.6% of DEPO's tokens and add 39.70M tokens. Twenty-eight are
classified as context-limited. This cause is not plotted separately because it
overlaps with the no-patch group and SWE-smith lacks an equivalent cause
classification.

Selected distributional summaries show the same heterogeneity. On SWE-smith,
the paired median changes are a 1,182-token saving and zero steps, yet the
all-attempt 90th percentiles rise from 1.647M to 2.029M total tokens and from
74.1 to 89.0 steps. On SWE-bench, the corresponding tails are similar between
SFT and DEPO (3.107M versus 3.095M tokens and 108.2 versus 105.1 steps).
Within each arm's resolved subset, DEPO's 90th percentiles are lower: 1.205M
versus 1.688M tokens and 69.7 versus 80.0 steps. This resolved-only comparison
is descriptive because the subsets contain different tasks. Across all paired
SWE-bench attempts, DEPO saves a mean 40,459 prompt tokens, 380 completion
tokens and 1.05 interaction steps per task.

The ranked plot orders tasks independently within each benchmark by their
paired token difference; horizontal position therefore represents rank rather
than task identity.

![Ranked paired task-level token differences for SWE-smith and SWE-bench Verified](assets/hyperparameter-paired-token-distribution.svg)

Most differences cluster near zero, while both tails extend to several million
tokens. On SWE-smith, DEPO is cheaper on 265 tasks and costlier on 235, with a
median saving of 1,182 tokens, yet the positive tail produces a 28.78M
aggregate increase. On SWE-bench, DEPO is cheaper on 252 tasks and costlier on
248, with a median saving of 2,200 tokens, while the aggregate result is a
20.42M saving. Its paired differences range from a 5.21M saving to a 5.08M
increase; gross savings of 198.93M are largely cancelled by 178.51M of
additional spend.

### Repository-level results

![Repository contributions to aggregate SWE-bench Verified changes](assets/hyperparameter-repository-effects.svg)

DEPO saves aggregate tokens in nine of the twelve repositories, but resolves
more tasks in only three; it ties in four and performs worse in five. In
`matplotlib`, lower spend accompanies five fewer resolutions, whereas
`requests`, `sympy` and `scikit-learn` improve both outcomes. The `django`
tasks constitute 46.2% of the benchmark.

## Discussion

### Interpretation of the main findings

The result is benchmark-dependent. SFT retained the best success-constrained
efficiency on both SWE-smith evaluations, whereas sequential DEPO met the
pre-declared resolution constraint and achieved the lowest observed token cost
per resolution on SWE-bench Verified. The large numbers of successes gained
and lost show that preference training changed which tasks were solved rather
than uniformly improving the SFT policy.

The relative screening advantage of total-balanced DEPO over DMPO did not
carry into the larger SWE-smith confirmation: sequential DEPO exhibited the
larger cost regression. Because DEPO was trained after DMPO, and there is no
matched direct SFT-to-DEPO arm or replication across training seeds, this
difference cannot be attributed specifically to sequential training.

Prompt tokens account for approximately 99% of total tokens because the full
interaction history is repeatedly supplied during an agent trajectory.
Consequently, small changes in trajectory length can dominate completion-token
savings. The total-balanced DEPO objective is directly aligned with this
reported metric, and its stronger screening result is consistent with that
alignment. This does not identify the causal mechanism, and unweighted total
tokens remain a compute proxy rather than monetary cost when prompt and
completion prices differ.

### Interpretation of task-level heterogeneity

The paired results qualify the aggregate efficiency ranking. The mean saving
on shared successes is driven by a minority of large reductions, while the
median shared-success task is slightly costlier under DEPO. The near-cancellation
between gained and lost successes further shows that preference training
redistributes capability across tasks. Consequently, neither the shared-success
mean nor the overall token reduction supports a claim of uniform per-task
improvement.

The jointly unresolved results also rule out a general inability to stop. DEPO
uses fewer resources across the larger patch-producing subgroup, but a smaller
no-patch tail continues for substantially longer and consumes disproportionate
resources. The SFT-only successes show why shorter failure trajectories cannot
be treated as equivalent to successful repair. The appropriate intervention is
therefore conditional recovery and stopping: detect stalled progress, attempt
a strategy change, and terminate only if progress remains absent. More
aggressive stopping on every difficult task could reduce correctness. This
mechanism remains a hypothesis rather than a causal finding.

The repository analysis adds a second source of heterogeneity. Improvements in
`requests`, `sympy` and `scikit-learn` coexist with a five-resolution decline in
`matplotlib`, while `django` supplies almost half of the benchmark. The
micro-average is therefore sensitive to repository composition; claims about
deployment should report repository-level effects alongside the overall result.

### Robustness, uncertainty and limitations

On the fixed SWE-bench Verified matrix, DEPO is success-eligible and has the
lowest observed token cost per resolution, so it is selected by the declared
rule. This finite-matrix decision does not require an inferential test. The
effect itself is modest: DEPO uses 588.15M tokens rather than 608.57M and
resolves two fewer tasks, reducing cost per resolution from 3.105M to 3.032M
(73,245 tokens, or 2.4%).

A post-hoc paired bootstrap assesses sensitivity beyond that fixed comparison.
Each replicate samples 500 task pairs with replacement and recomputes the
metric. Across 20,000 replicates, the middle 95% of cost-per-resolution changes
ranges from a 14.8% saving to a 12.0% increase; the resolution-rate change
ranges from -4.0 to +3.2 percentage points and crosses the pre-declared -3-point
margin. The 20,000 figure is only a Monte Carlo repetition count chosen to
stabilise the percentiles, not an experimental sample size. More repetitions
would reduce simulation noise but would not add evidence or materially narrow
the range.

This range is a sensitivity interval, not a conventional confidence interval
for a defined population. SWE-bench Verified is curated rather than randomly
sampled, and tasks within repositories may be correlated. The result therefore
shows that DEPO wins under the operational rule on the observed matrix, but
that the ranking is not robust to plausible task reweightings. Similarly, the
high McNemar *p* value indicates no detectable resolution difference; it does
not establish equality or non-inferiority.

The opposing tails in the ranked task-level distributions explain this
sensitivity: reweighting a relatively small number of large changes can alter
the aggregate ranking even though the median paired difference is near zero.

The main threats to validity are the single training seed, one deterministic
rollout per task, hyperparameter selection on only 200 tasks, and the absence
of uncertainty over training stochasticity. Wilson intervals and McNemar tests
address task-level outcome uncertainty, while the bootstrap measures
sensitivity to task reweighting; none captures training stochasticity.
Moreover, shared-success efficiency is a post-outcome conditional analysis and
should not replace the all-attempt metric. The bootstrap also treats tasks as
independent, whereas the repository effects indicate shared structure;
generalisation to new repositories may therefore be less stable than the
task-level bootstrap suggests.

## Conclusions and future work

The pre-declared operational rule produces a benchmark-dependent conclusion:
SFT wins the held-out SWE-smith confirmation, whereas sequential DEPO wins on
the fixed SWE-bench Verified matrix. DEPO would therefore be selected if that
benchmark and the stated metric represent the deployment target. Its near-SFT
resolution rate and lower aggregate token use constitute a **positive observed
test result under the rule**. The analysis does not establish that DEPO will
remain more efficient across new task samples, training seeds or repositories.

Future work should first determine whether the result is limited by data or
optimisation scale before introducing further objective variants:

1. **Scale the preference data.** Extend collection beyond the current 1,000
   tasks and four selected rollouts per task, potentially towards the existing
   5,000-task training membership. Within-task coverage should also be tested by
   increasing from four to eight selected rollouts per task, balanced as four at
   temperature 0.6 and four at 0.7. A staged learning curve should vary task
   count and rollouts per task separately; compute-matched comparisons such as
   more tasks with four rollouts versus fewer tasks with eight would distinguish
   the value of task diversity from additional samples of the same task. The
   current validation and test tasks must remain excluded from expanded
   training data. Collection should deliberately include efficient successful
   trajectories and matched costly no-progress trajectories, allowing
   preference construction to distinguish productive persistence from
   unproductive continuation rather than merely adding undifferentiated data.
   Correctness must remain primary: a cheaper jointly failed trajectory should
   not be preferred over a successful trajectory solely because it stops
   earlier.
2. **Calibrate compute and effective batch size.** Training used eight
   processes, per-device batch size 1 and gradient accumulation 32, giving a
   nominal global batch of 256 sequences. With three epochs, this produced only
   approximately 30 DMPO and 48 DEPO optimiser updates. Controlled runs should
   vary effective batch size and update count separately, using matched data
   and compute budgets where possible, and report accelerator-hours, processed
   tokens and optimiser steps. This would test whether the preference stages
   received enough parameter updates before attributing the observed
   regressions to the objectives themselves.
3. **Replicate the selected regime.** After choosing the data and compute
   configuration, repeat training across seeds, use multiple evaluation
   rollouts where affordable, and evaluate on a new untouched task matrix.
   Resolution non-inferiority and cost-per-resolution improvement should both
   be assessed rather than inferred from a single deterministic run.
4. **Target the observed failure modes and heterogeneity.** Prospectively test
   a conditional controller that detects stalled progress, switches strategy
   where useful and terminates only if progress remains absent. Loop detection
   and calibrated context budgets should be tested specifically against the
   no-patch and context-exhaustion tails. These interventions should be
   evaluated under a resolution-rate constraint, since premature stopping
   could reduce token use by terminating tasks that might otherwise be solved.
   Repository-stratified reporting should accompany the micro-average, and
   monetary-cost analyses should weight prompt and completion tokens by their
   actual prices.

Building on previous work in multi-turn and efficiency-aware preference
optimisation, this study provides a promising but preliminary positive result
for repository-repair agents. On SWE-bench Verified, DEPO met the pre-declared
success constraint and reduced observed token cost per resolution by 2.4%
relative to SFT. This improvement was not reproduced on SWE-smith, so it cannot
yet be considered a general advantage. However, the experiment used only 1,000
preference tasks, four selected rollouts per task and relatively few optimiser
updates, leaving substantial room for controlled scaling of the data, rollout
coverage and training compute. The key implication is therefore not that DEPO
is already proven superior, but that token-aware preference optimisation has
produced a credible signal that warrants larger and better-resourced
experiments.

## Reproducibility

Frozen comparison summaries and per-task analysis matrices are stored under
[`results`](../results/README.md). Figures and
uncertainty summaries are regenerated directly from these files with
[`generate-hyperparameter-results.py`](../results/generate-hyperparameter-results.py)
using NumPy seed 42.
