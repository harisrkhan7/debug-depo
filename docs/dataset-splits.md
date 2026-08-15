# Dataset splits

The experiment uses `SWE-bench/SWE-smith-py` revision
`77cab9055d42ab4a5c25c89a8f937096db13558e` with seed 42. The upstream dataset
contains 50,908 tasks in a single published `train` split, so a local training
and held-out partition is constructed before drawing the experiment samples.

## Training and held-out partition

Tasks are grouped by the dataset's `repo` field, and whole repositories are
assigned to only one side of the partition. Repository names are ordered by
`SHA256(seed || repository)`. In that deterministic order, a repository is
added to the held-out side only when doing so moves its task count closer to
the target of 10% of all tasks. This produces:

| Parent partition | Tasks | Repositories |
| --- | ---: | ---: |
| Training | 45,809 | 117 |
| Held out | 5,099 | 14 |

The resulting held-out fraction is 10.0161%. Keeping complete repositories on
one side prevents related mutations from the same repository appearing in
both training and validation. The parent memberships are stored in
`data/splits/train_instance_ids.txt` and
`data/splits/validation_instance_ids.txt`.

One held-out repository, `stanfordnlp__string2string.c4a72f59`, is excluded
from the derived experiment samples because its image does not contain the
required synthetic task branches. This leaves 13 eligible validation
repositories; the 14-repository parent membership is retained unchanged.

Repository-image preflight also identified three unavailable Autograd PR refs:
`HIPS__autograd.ac044f0d.pr_607`, `HIPS__autograd.ac044f0d.pr_618`, and
`HIPS__autograd.ac044f0d.pr_672`. They are recorded in
`data/splits/swesmith_validation_unavailable_instance_ids.txt` and excluded
from the confirmation candidate pool before sampling. Their slots are filled
by the same deterministic repository-balanced policy, so neither repository
allocation nor sample size changes.

## Experiment samples

The reduced experiment uses these immutable memberships:

| Role | Tasks | Sampling policy | Ordered-file SHA-256 |
| --- | ---: | --- | --- |
| Training | 1,000 | Task-proportional with every training repository covered | `4d60dbdc69aca4a1704d8c23ed0a72161e96fe93d077212aedb68e1312412965` |
| Early screening | 100 | Task-proportional across the 13 eligible held-out repositories | `ba2049878ec15f274d18139432ab08b9b4a0777f342a7ebac308ad8f7cb3979e` |
| Screening | 200 | Task-proportional across the 13 eligible held-out repositories | `36fc626d59cede1ea53ae0bb7934f15d5d8d95c5ac2777a57a1ee11bcd94997c` |
| Confirmation | 500 | Repository-balanced across the same 13 repositories | `0a4dbb16a6bf309573db1a809d53302d398d9e3b9d5d2ddd942c7bc811e7928a` |

The 100-task early-screening set is nested within the 200-task screening set.
Screening uses `data/splits/swesmith_validation_200_instance_ids.txt`.
Confirmation uses
`data/splits/swesmith_validation_confirmatory_balanced_500_instance_ids.txt`.
Their memberships are disjoint.

Both samples use deterministic SHA-256 ranking and ordering. Screening uses
task-proportional repository quotas. Confirmation allocates tasks as evenly as
availability permits: 17 tasks from each of the two smallest repositories and
42 or 43 from each of the other eleven. No outcome label is used during
sampling.

Technical eligibility is checked independently of model output with
`scripts/preflight_swesmith_branches.py`. The read-only check verifies that
each selected instance ref resolves in its cached repository image before
collection begins.

The 200-task set is used for hyperparameter selection. The selected models are
then evaluated once on the disjoint 500-task set, which is treated as
confirmatory validation. Full provenance and membership hashes are stored in
`data/splits/swesmith_py_split_manifest.json`.
