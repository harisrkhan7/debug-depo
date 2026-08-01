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
  used as the initial Hyperstack collection. Its ordered-file SHA-256 is
  `4d60dbdc69aca4a1704d8c23ed0a72161e96fe93d077212aedb68e1312412965`.
- `swesmith_validation_500_instance_ids.txt`: the validation sample, drawn
  only from `validation_instance_ids.txt` and covering all 14 validation
  repository snapshots.
- `swesmith_validation_100_instance_ids.txt` and
  `swesmith_validation_200_instance_ids.txt`: deterministic nested screening
  budgets for the DMPO/DEPO
  [hyperparameter sweep](../../docs/hyperparameter-sweep.md). Both
  cover all 14 validation repository snapshots, the 100-task sample is a
  subset of the 200-task sample, and both are subsets of the tracked
  500-task sample.
- `swesmith_cache_5500_instance_ids.txt`: the exact union of the 5,000
  trajectory tasks and 500 validation tasks. Use this to prebuild the cache.
  The cache builder deduplicates these task IDs into 131 SWE-smith
  repository-image SIFs.
- `swesmith_py_split_manifest.json`: dataset revision, policies, counts,
  memberships, and ordered-file SHA-256 hashes.

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

Each sample is selected independently inside its parent membership. The
procedure is deterministic and does not depend on the input file order:

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

## Regeneration

Regenerate only the 5,000/500/cache files from the tracked parent memberships
without downloading the dataset:

```bash
python -m debug_depo.prepare_swesmith_splits --subsets-only
```

Regenerate the parent 90/10 memberships and all derived files from the pinned
Hugging Face dataset:

```bash
python -m debug_depo.prepare_swesmith_splits
```

Custom sample sizes are supported:

```bash
python -m debug_depo.prepare_swesmith_splits \
  --subsets-only \
  --trajectory-subset-size 5000 \
  --validation-subset-size 500
```

The generated filenames include their task counts. Review and commit the
manifest and ID files together if the dataset revision, seed, or sizes change.

## Cluster usage

Build all 500 SWE-bench Verified images and the 131 SWE-smith images needed by
the selected 5,500 tasks:

```bash
SWESMITH_TASK_IDS_FILE=data/splits/swesmith_cache_5500_instance_ids.txt \
  cluster/submit_apptainer_cache_full.sh
```

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
TASK_IDS_FILE=data/splits/swesmith_validation_500_instance_ids.txt \
EXPECTED_TASKS=500 \
NUM_SHARDS=100 \
  cluster/submit_swesmith_full.sh
```

Keep `SPLIT=train`: validation here means local task-ID membership, not a
separate Hugging Face split. Preview any submission by prefixing the command
with `DRY_RUN=1`. With the tracked two temperatures and four runs per
temperature, these commands produce eight trajectories per task: 40,000 for
the training sample and 4,000 for the validation sample.
