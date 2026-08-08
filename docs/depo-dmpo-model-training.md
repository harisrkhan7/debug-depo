# How we train with DMPO and DEPO

## The setup in one minute

We first run the SFT model several times on each SWE-smith debugging task and
score every rollout as resolved or unresolved. A rollout is the full agent
trajectory: the initial task, assistant actions, and environment observations.

The normal training path is:

```text
SFT model -> DMPO on chosen/rejected trajectories -> DEPO on labelled trajectories
```

DMPO teaches the model which complete trajectory is better. DEPO then adds an
explicit preference for successful trajectories that use fewer tokens per step
and fewer steps. Both stages train a LoRA adapter; the model at the start of the
stage stays frozen and acts as the reference policy. The final adapter is merged
into a standalone model.

## Constructing the training data

By default, we select up to four temperature-balanced, fully evaluated rollouts
per task. We keep the initial system/user prompt and the multi-turn conversation,
but remove provider response payloads. Token counts are summed from the API usage
record of every assistant call. A **step** is one assistant call.

### DMPO: actual preference pairs

For each task, we compare every pair of selected rollouts:

1. A resolved rollout always beats an unresolved rollout, regardless of cost.
2. If both are resolved, the lower-cost rollout wins only when the other costs
   at least `MIN_COST_RATIO` times as much (`1.1` by default).
3. Two unresolved rollouts do not form a pair by default.
4. Equal-cost pairs and differences below the ratio threshold are discarded.

Cost defaults to accumulated `total_tokens`; `completion_tokens` can be selected
instead. Each retained row contains the common prompt, a `chosen` trajectory, and
a `rejected` trajectory.

### DEPO: labels, not pairs

DEPO reuses the selected rollouts individually:

- resolved -> `desirable`
- scored but unresolved -> `undesirable`

For every trajectory it stores the step count, tokens per step, and their
inverses. This is an important distinction: only DMPO trains on pairs. DEPO uses
unpaired KTO-style binary labels. A deliberately mismatched prompt/completion is
created inside training only to estimate the KTO KL term; it is not a preference
pair.

The repository labels every scored failure as undesirable. This is simpler than
the DEPO paper, which filters out very low-quality failures.

## How DMPO trains the model

Let `pi_theta` be the LoRA-trained policy and `pi_ref` the frozen model from the
start of the stage. For trajectory `tau`, the code computes a
weighted sum of assistant-token log-probabilities:

```text
S_pi(tau) = sum over turns t [
    phi(t, T) * sum of log pi(token | previous context)
                for assistant tokens in turn t
]
```

The turn weight is:

```text
phi(t, T) = gamma^t * (1 - gamma^(T-t)) / (1 - gamma^T)
```

Earlier actions receive more weight. With the project default `gamma = 0.7`,
later turns still matter but are discounted.

For a chosen trajectory `tau_w` and rejected trajectory `tau_l`, define
the policy and reference margins:

```text
policy_margin    = S_pi_theta(tau_w) - S_pi_theta(tau_l)
reference_margin = S_pi_ref(tau_w)   - S_pi_ref(tau_l)
```

The implemented loss is:

```text
DMPO loss = -log sigmoid(
    beta * (policy_margin - reference_margin)
)
```

Minimizing it makes the trained model prefer the chosen trajectory more strongly
than the reference model does. There is no separate reward model or online RL
loop.

This is the practical form of **DMPO paper Equation 16**. Equations 6--15 derive
why a multi-turn, length-normalized preference loss has this form. The paper's
displayed definition after Equation 16 omits the leading `gamma^t`, but its
official code and its `gamma -> 0` result include it; this repository follows
the official code.

## How DEPO trains the model

DEPO starts either from SFT directly or, in the normal sequential arm, from the
merged DMPO model. For one labelled trajectory, it computes the reference-relative
log-probability:

```text
q_theta(tau) = log pi_theta(tau) - log pi_ref(tau)
```

It also estimates a non-negative KL reference point `z0` from mismatched
prompt/completion examples. Efficient desirable trajectories receive:

```text
bonus = alpha_tokens / tokens_per_step
      + alpha_steps  / steps
```

while undesirable trajectories receive `bonus = 0`. The two per-example
losses implemented in the trainer are:

```text
desirable loss = lambda_D * [1 - sigmoid(beta * (q_theta - z0 + bonus))]

undesirable loss = lambda_U * [1 - sigmoid(beta * (z0 - q_theta))]
```

The batch loss is their mean. In plain language, desirable trajectories are
pushed above the reference point, undesirable ones below it, and successful
trajectories get an efficiency-aware bonus.

These correspond directly to the DEPO paper:

| Paper equation | Role in this repository |
| --- | --- |
| Eq. 5--7 | KTO outer loss and separate desirable/undesirable sigmoid branches |
| Eq. 8 | KL reference point `z0` |
| Eq. 9 | Reference-relative log-probability plus efficiency bonus |
| Eq. 10 | Desirable-only inverse tokens-per-step and inverse-steps bonus |

Using `completion_tokens` most closely matches the paper's generated-token
definition. The repository default, `total_tokens`, also counts accumulated
prompt context and is a local deployment-cost extension.

## What tokens receive gradients?

Only assistant-action tokens contribute to either loss. The system prompt and
environment/user observations remain in the context so the model can condition
on them, but their loss weights are zero. DMPO additionally applies the turn
weights `phi(t, T)`; DEPO gives all assistant turns weight 1. In this setup,
both algorithms use `MAX_LENGTH=8192` (an 8K-token training sequence limit).

The main script defaults are three epochs, batch size 1 with gradient
accumulation 32, and LoRA rank 64. The important objective defaults are
`beta=0.1, gamma=0.7` for DMPO and
`beta=0.2, alpha_tokens=alpha_steps=2` for DEPO.

## Sources of truth

- Data construction: `src/debug_depo/build_dmpo_pairs.py`,
  `src/debug_depo/build_depo_data.py`, and `src/debug_depo/preference_data.py`
- Losses and token masking: `src/debug_depo/preference_training.py`
- Runtime defaults: `scripts/preference_defaults.sh`
- [DMPO paper](https://aclanthology.org/2024.emnlp-main.138/) and
  [official implementation](https://github.com/swt-user/DMPO)
- [DEPO paper](https://ojs.aaai.org/index.php/AAAI/article/view/40279)
