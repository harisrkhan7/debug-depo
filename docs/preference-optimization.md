# Preference optimisation

The aim is to reduce deployment cost without materially reducing the task
resolution rate.

```text
minimise total_tokens_per_resolved_task
subject to resolution_rate >= SFT resolution_rate - tolerance
```

Resolved trajectories are always preferred to unresolved trajectories. Cost
is used only to compare successful trajectories, so short failures are not
rewarded.

## Current setup

| Item | Status |
| --- | --- |
| Multi-turn DMPO weighting | Implemented |
| Success-first DMPO pairs | Implemented |
| Completion- or total-token pairs | Implemented |
| DEPO efficiency bonus | Implemented |
| Trial lineage and data hashes | Implemented |
| Cost-aware evaluation | Implemented |
| Step-aware pair filtering | Not implemented |
| Deployment-price weighting | Not implemented |

DMPO was tested with `gamma` values of 0.5, 0.7 and 0.9. The selected value was
0.7. DEPO was tested with paper, completion-balanced and total-balanced token
settings. Total-balanced was selected.

The completed benchmark results are in `hyperparameter-sweep-results.md`.
