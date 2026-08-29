# Experiment results

This directory contains the frozen inputs used to generate
`docs/hyperparameter-sweep-results.md` and its figures.

- `generate-hyperparameter-results.py`: reproducible analysis and figure
  generator.
- `swesmith_screening_200/`: seven-arm comparison summary and all seven
  per-task matrices.
- `swesmith_confirmation_500/`: comparison summary and the SFT, DMPO and
  DMPO-to-DEPO per-task matrices.
- `swebench_verified_500/`: comparison summary and the SFT, DMPO and
  DMPO-to-DEPO per-task matrices.

The result data files are direct copies of the corresponding comparison and
analysis artifacts under `scratch/cloud/runs`. The reporting script reads only
this directory, so the document can be reproduced without the complete run
tree. The script writes matching SVG and vector-PDF figures to `docs/assets`;
the PDFs can be included directly in LaTeX. Run it from the repository root
with:

```bash
uv sync --extra notebooks
.venv/bin/python results/generate-hyperparameter-results.py
```
