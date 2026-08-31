# Notebook Index

The recommended way to run the experiment is through the repository's shell
scripts and environment guides. These notebooks provide guarded, interactive
references for development, smoke testing, and inspection; all job-submission
switches default to off.

| Notebook | Status | Purpose |
| --- | --- | --- |
| `local_llm_server_check.ipynb` | Current | Check a local OpenAI-compatible MLX server |
| `local_agentforge_swebench_smoke.ipynb` | Current | Run the local SWE-bench artifact and real-harness smoke tests |
| `inspect_swesmith_collection.ipynb` | Current, read-only | Inspect collection coverage, trajectories, evaluation, and failures |
| `cluster_agentforge_swebench.ipynb` | Maintained PBS reference | Preview or run Verified smoke and shard workflows |
| `cluster_agentforge_swesmith.ipynb` | Maintained PBS reference | Preview or run SWE-smith smoke, pilot, and scheduled workflows |
| `cluster_agentforge_swesmith_train.ipynb` | Proposed full-scale workflow | Describe the uncompleted 5,000-task, eight-rollout PBS design |
| `cluster_preference_training.ipynb` | Historical pilot | Preserve the 30-task PBS preference-training workflow |

The completed 1,000-task main experiment used the command-driven
[Lambda Cloud workflow](../cloud/README.md), not the proposed 5,000-task PBS
notebook.
