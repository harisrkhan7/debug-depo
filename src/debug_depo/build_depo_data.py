"""Build DEPO desirable/undesirable trajectory data from a SWE-smith run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from debug_depo.preference_data import (
    TrajectoryRecord,
    artifact_record,
    load_evaluated_trajectories,
    parse_sample_indices,
    select_sample_indices,
    selected_temperature_counts,
    write_jsonl,
)
from debug_depo.utils import write_json


def _row(record: TrajectoryRecord) -> dict[str, Any]:
    label = "desirable" if record.resolved else "undesirable"
    metadata = record.metadata()
    return {
        "id": f"{record.instance_id}:s{record.sample_index}",
        "instance_id": record.instance_id,
        "label": label,
        "prompt": record.prompt,
        "completion": record.completion,
        # DEPO equation (10) uses these two denominators for desirable samples.
        "efficiency": {
            "completion_tokens_per_step": metadata["completion_tokens_per_step"],
            "total_tokens_per_step": metadata["total_tokens_per_step"],
            "steps": record.steps,
            "inverse_completion_tokens_per_step": (
                record.steps / record.completion_tokens
                if record.completion_tokens
                else 0.0
            ),
            "inverse_total_tokens_per_step": record.steps / record.total_tokens,
            "inverse_steps": 1.0 / record.steps,
        },
        "metadata": metadata,
    }


def build_depo_data(
    run_root: str | Path,
    *,
    output_path: str | Path,
    desirable_output: str | Path,
    undesirable_output: str | Path,
    summary_path: str | Path,
    max_rollouts: int = 4,
    sample_indices: list[int] | None = None,
) -> dict[str, Any]:
    selected_samples = select_sample_indices(
        run_root,
        max_rollouts=max_rollouts,
        sample_indices=sample_indices,
    )
    records = load_evaluated_trajectories(run_root, sample_indices=selected_samples)
    rows = [_row(record) for record in records]
    desirable = [row for row in rows if row["label"] == "desirable"]
    undesirable = [row for row in rows if row["label"] == "undesirable"]
    row_count = write_jsonl(output_path, rows)
    desirable_count = write_jsonl(desirable_output, desirable)
    undesirable_count = write_jsonl(undesirable_output, undesirable)
    status_counts = Counter(record.evaluation_status for record in records)
    summary = {
        "schema_version": 1,
        "objective": "depo",
        "format": "unpaired KTO-style binary trajectory labels",
        "run_root": str(run_root),
        "output_path": str(output_path),
        "desirable_output": str(desirable_output),
        "undesirable_output": str(undesirable_output),
        "max_rollouts": max_rollouts,
        "selected_sample_indices": selected_samples,
        "selected_temperature_counts": selected_temperature_counts(
            run_root, selected_samples
        ),
        "trajectories": len(rows),
        "desirable": len(desirable),
        "undesirable": len(undesirable),
        "evaluation_status_counts": dict(sorted(status_counts.items())),
        "efficiency_bonus": (
            "For desirable rows: alpha1 * inverse_*_tokens_per_step "
            "+ alpha2 * inverse_steps. Use inverse_total_tokens_per_step "
            "to optimize billed debugging cost, or inverse_completion_tokens_per_step "
            "to reproduce the paper's generated-token definition."
        ),
        "artifacts": {
            "trajectories": artifact_record(
                output_path,
                rows=row_count,
                summary_path=summary_path,
            ),
            "desirable": artifact_record(
                desirable_output,
                rows=desirable_count,
                summary_path=summary_path,
            ),
            "undesirable": artifact_record(
                undesirable_output,
                rows=undesirable_count,
                summary_path=summary_path,
            ),
        },
        "complete": True,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--desirable-output")
    parser.add_argument("--undesirable-output")
    parser.add_argument("--summary-output")
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=4,
        help="Maximum temperature-balanced rollouts per task; use 0 for all.",
    )
    parser.add_argument(
        "--sample-indices",
        help="Explicit comma-, colon-, or whitespace-separated sample indices.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.run_root) / "preference-data" / "depo"
    build_depo_data(
        args.run_root,
        output_path=args.output or output_dir / "trajectories.jsonl",
        desirable_output=args.desirable_output or output_dir / "desirable.jsonl",
        undesirable_output=args.undesirable_output or output_dir / "undesirable.jsonl",
        summary_path=args.summary_output or output_dir / "summary.json",
        max_rollouts=args.max_rollouts,
        sample_indices=(
            parse_sample_indices(args.sample_indices) if args.sample_indices else None
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
