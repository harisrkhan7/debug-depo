# SWE-smith Python split and sampling policy

`SWE-bench/SWE-smith-py` publishes one upstream split named `train`. This
directory tracks a local holdout and the smaller task sets used by this
project. All files were generated from dataset revision
`77cab9055d42ab4a5c25c89a8f937096db13558e` with seed 42.

## Files

- `train_instance_ids.txt`: 45,809 tasks from 117 repository snapshots.
- `validation_instance_ids.txt`: 5,099 tasks from 14 repository snapshots.
- `swesmith_train_5000_instance_ids.txt`: the trajectory-collection sample,
  drawn only from `train_instance_ids.txt` and covering all 117 training
  repository snapshots.
- `swesmith_train_1000_instance_ids.txt`: a proportional subset of the tracked
  5,000-task training sample covering all 117 training repository snapshots,
  used for the reduced trajectory collection. Its ordered-file SHA-256 is
  `4d60dbdc69aca4a1704d8c23ed0a72161e96fe93d077212aedb68e1312412965`.
- `swesmith_validation_500_instance_ids.txt`: the fixed exclusion membership
  used to reproduce the confirmatory sample.
- `swesmith_validation_unavailable_instance_ids.txt`: task refs that are absent
  from their pinned repository images and therefore cannot be initialized by
  the harness. These tasks are excluded before confirmatory sampling.
- `swesmith_validation_100_instance_ids.txt` and
  `swesmith_validation_200_instance_ids.txt`: deterministic nested screening
  budgets for the DMPO/DEPO
  [hyperparameter sweep](../../docs/hyperparameter-sweep-light.md). Both
  cover all 13 eligible validation repository snapshots, and the 100-task
  sample is a subset of the 200-task sample.
- `swesmith_validation_confirmatory_balanced_500_instance_ids.txt`: the fixed
  repository-balanced confirmation sample. It is disjoint from the 200-task
  screening set and excludes both the fixed prior membership and the tracked
  unavailable task refs.
- `swesmith_validation_64_instance_ids.txt`: the fixed 64-instance validation
  set used for the DMPO/DEPO end-to-end pilot comparison. Its order reproduces
  the original eight-shard pilot assignment.
- `swesmith_cache_5700_instance_ids.txt`: the exact union of the 5,000
  trajectory tasks, 200 screening tasks, and 500 confirmatory tasks. Use this
  to prebuild the cache for the active design.
  The cache builder deduplicates these task IDs into 130 usable SWE-smith
  repository-image SIFs. The excluded image is not required by these subsets.
- `swesmith_py_split_manifest.json`: dataset revision, policies, counts,
  memberships, and ordered-file SHA-256 hashes for the parent memberships and
  the active 5,000/500/200/100 subsets.

The 5,000 training IDs and 500 validation IDs are disjoint. Their parent
memberships are also repository-disjoint, so no repository snapshot selected
for training can appear in validation.

## How the 90/10 holdout was chosen

The upstream tasks are grouped by their `repo` field. Repository groups are
ranked by `SHA256(seed || repo)`. In that stable order, an entire repository is
added to validation only if doing so moves the validation task count closer to
10% of the dataset. Repositories are never divided across the two sides.

For the pinned 50,908-task dataset, the rounded target is 5,091 tasks. Whole
repository groups make an exact target unlikely; the deterministic greedy
selection produced 5,099 validation tasks (10.0161%) across 14 repositories
and left 45,809 tasks across 117 repositories for training.

This is a repository-disjoint holdout, not a task-level random 90/10 split.
That choice reduces leakage from closely related mutations of the same
repository into both training and validation. It also means the task counts
cannot generally be exactly 90/10.

## How the training and validation task samples were chosen

Training and screening samples use a task-proportional policy. The procedure
is deterministic and does not depend on the input file order:

1. Group IDs by repository snapshot (the instance ID without its final
   dot-separated mutation component).
2. Reserve one task for every repository, ensuring repository coverage.
3. Allocate each remaining slot to the repository currently furthest below
   its proportional ideal, `sample_size * repository_size / parent_size`,
   without exceeding that repository's available tasks.
4. Within each repository, rank tasks by a seed- and namespace-specific SHA-256
   hash and sample without replacement.
5. Order the selected file by a separate SHA-256 hash. This avoids
   repository- or source-order clustering when the file is sharded.

This policy balances two goals: preserve the parent distribution as closely as
possible while ensuring that small repositories are not absent. SHA-256
ranking supplies reproducible pseudo-random sampling; it is not intended for
cryptographic secrecy.

The confirmatory 500 uses a separate repository-balanced policy. Its candidate
pool excludes the 200 screening tasks, the fixed exclusion membership, and the
tracked technically unavailable task refs.
Slots are allocated as evenly as repository capacity permits, after which
tasks are selected and ordered by a separate seeded SHA-256 namespace.
Consequently, screening and confirmation are task-disjoint while covering the
same 13 eligible held-out repositories.

The tracked validation subsets exclude
`stanfordnlp__string2string.c4a72f59` because its upstream image does not
contain the synthetic task branches. The exclusion applies only while deriving
the screening and confirmatory task samples; the 14-repository validation
parent is left unchanged for provenance. Replacement tasks are
sampled from the other 13 held-out repositories, never from the training
membership.

The confirmatory candidate pool also excludes the three Autograd PR refs in
`swesmith_validation_unavailable_instance_ids.txt`. A read-only preflight of
the cached repository images established that these refs do not exist in the
Autograd image. The same deterministic repository-balanced sampler then fills
those slots from the remaining eligible Autograd tasks. This preserves the
repository quota and sample size without using model outcomes.

## Regeneration

Regenerate the proportional screening budgets, disjoint balanced confirmation
sample, cache union, and manifest provenance from the tracked parent
memberships without downloading the dataset:

```bash
uv run debug-depo-prepare-swesmith-splits \
  --subsets-only \
  --validation-design disjoint-balanced \
  --confirmation-exclude-instance-ids-file \
    data/splits/swesmith_validation_500_instance_ids.txt \
  --confirmation-exclude-instance-ids-file \
    data/splits/swesmith_validation_unavailable_instance_ids.txt \
  --exclude-repository stanfordnlp__string2string.c4a72f59
```

Regenerate the parent 90/10 memberships and all derived files from the pinned
Hugging Face dataset:

```bash
uv run debug-depo-prepare-swesmith-splits \
  --validation-design disjoint-balanced \
  --confirmation-exclude-instance-ids-file \
    data/splits/swesmith_validation_500_instance_ids.txt \
  --confirmation-exclude-instance-ids-file \
    data/splits/swesmith_validation_unavailable_instance_ids.txt \
  --exclude-repository stanfordnlp__string2string.c4a72f59
```

Custom sample sizes are supported:

```bash
uv run debug-depo-prepare-swesmith-splits \
  --subsets-only \
  --trajectory-subset-size 5000 \
  --validation-subset-size 500
```

The generated filenames include their task counts. Generation fails if the
screening budgets are not nested or if screening and confirmation overlap.
Review and commit the manifest and ID files together if the dataset revision,
seed, or sizes change.

## Cluster usage

Build all 500 SWE-bench Verified images and the 130 SWE-smith images needed by
the selected 5,700 tasks:

```bash
SWESMITH_TASK_IDS_FILE=data/splits/swesmith_cache_5700_instance_ids.txt \
  cluster/submit_apptainer_cache_full.sh
```

After the SIFs are available, verify that every selected task ref resolves in
its repository image before starting collection:

```bash
source cloud/env.sh
PYTHONPATH=src .venv/bin/python scripts/preflight_swesmith_branches.py \
  --missing-output /tmp/swesmith-missing-branches.txt
```

The tracked confirmatory membership should report zero unavailable refs. The
script is read-only, starts each repository image once, and requires neither a
model server nor a GPU. A nonzero unavailable count exits with status 1; image
or Apptainer failures exit with status 2.

Collect the 5,000-task training set:

```bash
RUN_NAME=swesmith-train-5000 \
TASK_IDS_FILE=data/splits/swesmith_train_5000_instance_ids.txt \
EXPECTED_TASKS=5000 \
NUM_SHARDS=25 \
  cluster/submit_swesmith_full.sh
```

Run the collection, evaluation, and analysis pipeline on the 500-task
validation set:

```bash
RUN_NAME=swesmith-validation-500 \
TASK_IDS_FILE=data/splits/swesmith_validation_confirmatory_balanced_500_instance_ids.txt \
EXPECTED_TASKS=500 \
NUM_SHARDS=100 \
  cluster/submit_swesmith_full.sh
```

Keep `SPLIT=train`: validation here means local task-ID membership, not a
separate Hugging Face split. Preview any submission by prefixing the command
with `DRY_RUN=1`. With the tracked two temperatures and four runs per
temperature, these commands produce eight trajectories per task: 40,000 for
the training sample and 4,000 for the validation sample.
