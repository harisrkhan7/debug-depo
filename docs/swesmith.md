# SWE-smith trajectory workflow

See the [project README](../README.md) for an overview and workflow index.

## Setup and dataset

Install the official harness, SWE-smith package, and repository profiles:

```bash
scripts/install_mini_swe_agent_plus.sh
scripts/install_swesmith.sh
```

The tracked pilot settings are:

```text
dataset:          SWE-bench/SWE-smith-py (train)
pilot tasks:      30
temperatures:     0.6, 0.7
runs/temperature: 4
total runs/task:  8
base seed:        42
```

Each task/sample pair has a stable derived seed and a separate prediction file,
so duplicate task IDs are never collapsed.

The upstream dataset has only a `train` split. Reproducible,
repository-disjoint memberships are tracked as:

- `data/splits/train_instance_ids.txt`: 45,809 tasks;
- `data/splits/validation_instance_ids.txt`: 5,099 tasks;
- derived samples: 5,000 training, 200 screening, and 500 disjoint
  confirmatory-validation tasks.

The pinned revision, policy, repository membership, and hashes are in
`data/splits/swesmith_py_split_manifest.json`; see the
[split guide](../data/splits/README.md) for the complete policy. Regenerate
only the derived samples with:

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

After building the repository SIF cache, preflight the selected task refs
against those images:

```bash
source cloud/env.sh
PYTHONPATH=src .venv/bin/python scripts/preflight_swesmith_branches.py \
  --missing-output /tmp/swesmith-missing-branches.txt
```

The tracked set is expected to report zero unavailable refs. The technical
exclusion membership is stored in
`data/splits/swesmith_validation_unavailable_instance_ids.txt` and is supplied
to the deterministic sampler during regeneration.

Submit the validation sample without changing the upstream split name:

```bash
RUN_NAME=swesmith-validation-500 \
SPLIT=train \
TASK_IDS_FILE=data/splits/swesmith_validation_confirmatory_balanced_500_instance_ids.txt \
EXPECTED_TASKS=500 \
NUM_SHARDS=100 \
DRY_RUN=1 \
cluster/submit_swesmith_full.sh
```

Collection selects tasks once, then sends one local task JSON at a time through
the same AgentForge adapter as the Verified collector. The generated startup
command checks out the task's SWE-smith branch in its repository image. Use the
standard `swebench` runner with Docker or the default `singularity` runner;
mini-swe-agent-plus's `pool_way` runner cannot execute that startup command.

## Artifacts and smoke test

```text
<run-root>/
  cluster-logs/
  collection/shard-*/
    collection_manifest.json
    samples/sample-0/ ... sample-7/
      predictions.jsonl
      summary.json
      trajectories/<instance-id>/
  merged/sample-0/ ... sample-7/
  evaluation/sample-0/ ... sample-7/
  analysis/
    rollouts.csv
    tasks.csv
    summary.json
```

Run a two-task artifact smoke test without an agent or task container:

```bash
MOCK=1 MOCK_PATCH=gold LIMIT=2 scripts/collect_swesmith.sh
```

SWE-smith gold patches introduce bugs, so mock predictions are marked for
reverse application by the evaluator.

## Cluster submission

Preview or submit the tracked modes:

```bash
DRY_RUN=1 cluster/submit_swesmith_smoke.sh
DRY_RUN=1 cluster/submit_swesmith_pilot.sh
DRY_RUN=1 cluster/submit_swesmith_full.sh

cluster/submit_swesmith_smoke.sh
cluster/submit_swesmith_pilot.sh
cluster/submit_swesmith_full.sh
```

Each submits `collect -> evaluate -> analyze` with `afterok` dependencies.
Override `TASK_LIMIT`, `EXPECTED_TASKS`, or `NUM_SHARDS`, or provide an
immutable `TASK_IDS_FILE`. In bounded modes, `EXPECTED_TASKS` defaults to
`TASK_LIMIT`; full mode verifies the complete selected dataset. Evaluation
also rejects an incomplete sample matrix. Cluster logs live under the
ephemeral run root, not in the repository.

For a direct multi-GPU Lambda VM instead of a scheduler, use the
[cloud guide](../cloud/README.md) and its
[setup and recovery runbook](../cloud/RUNBOOK.md). They cover storage,
credentials, SIF restoration, runtime setup, and replacement-VM resume.

Important reproducibility and recovery behavior:

- The dataset defaults to revision
  `77cab9055d42ab4a5c25c89a8f937096db13558e`; override
  `SWESMITH_DATASET_REVISION` deliberately.
- The installers use fixed commits, configurable through
  `MINI_SWE_AGENT_PLUS_REVISION` and `SWESMITH_REVISION`. Manifests record
  these, the dataset revision, and any deterministic installer-patch hash.
- Step/context-limit terminations remain model outcomes with empty patches.
  Infrastructure failures fail the shard; reruns retry only failed slots.
- The full wrapper defaults to
  `data/splits/swesmith_train_5000_instance_ids.txt`, 5,000 tasks, 50 shards,
  and six concurrent trajectories per shard. Set `TASK_IDS_FILE` and
  `EXPECTED_TASKS` together for another subset.

On clusters, SWE-smith evaluation uses Apptainer while preserving the official
repository profiles, tests, and grading. Persistent SIF and cache paths are
configured by `SWESMITH_APPTAINER_SIF_DIR` and
`SWESMITH_APPTAINER_CACHE_DIR`. Images are pulled once under a filesystem lock;
collection creates writable sandboxes from the cached local SIF.

Prebuild shared caches before collection:

```bash
# One Verified image and one SWE-smith image
cluster/submit_apptainer_cache_smoke.sh

# All Verified and unique split-specific SWE-smith images
cluster/submit_apptainer_cache_full.sh
```

See the [cluster guide](../cluster/README.md) for environment, cache, resource,
and submission details. Use `notebooks/cluster_agentforge_swesmith.ipynb` for
the interactive workflow and `notebooks/inspect_swesmith_collection.ipynb` to
inspect coverage, temperatures, messages, patches, reports, and
per-temperature pass@1...4.
