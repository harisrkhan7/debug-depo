"""Build full-trajectory DMPO preference pairs from evaluated SWE-smith rollouts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from debug_depo.preference_data import (
    TOKEN_METRICS,
    TrajectoryRecord,
    artifact_record,
    load_evaluated_trajectories,
    parse_sample_indices,
    select_sample_indices,
    selected_temperature_counts,
    write_jsonl,
)
from debug_depo.utils import write_json


def _prefer(
    left: TrajectoryRecord,
    right: TrajectoryRecord,
    *,
    token_metric: str,
    min_cost_ratio: float,
    include_failure_efficiency_pairs: bool,
) -> tuple[TrajectoryRecord, TrajectoryRecord, str] | None:
    if left.resolved != right.resolved:
        return (left, right, "task_success") if left.resolved else (right, left, "task_success")
    if not left.resolved and not include_failure_efficiency_pairs:
        return None

    left_cost = left.token_cost(token_metric)
    right_cost = right.token_cost(token_metric)
    if left_cost == right_cost:
        return None
    chosen, rejected = (left, right) if left_cost < right_cost else (right, left)
    ratio = rejected.token_cost(token_metric) / chosen.token_cost(token_metric)
    if ratio < min_cost_ratio:
        return None
    reason = "resolved_token_efficiency" if chosen.resolved else "failure_token_efficiency"
    return chosen, rejected, reason


def build_dmpo_pairs(
    run_root: str | Path,
    *,
    output_path: str | Path,
    summary_path: str | Path,
    token_metric: str = "total_tokens",
    min_cost_ratio: float = 1.1,
    include_failure_efficiency_pairs: bool = False,
    max_pairs_per_task: int = 0,
    max_rollouts: int = 4,
    sample_indices: list[int] | None = None,
) -> dict[str, Any]:
    if token_metric not in TOKEN_METRICS:
        raise ValueError(f"token_metric must be one of: {', '.join(sorted(TOKEN_METRICS))}")
    if min_cost_ratio < 1:
        raise ValueError("min_cost_ratio must be at least 1")
    if max_pairs_per_task < 0:
        raise ValueError("max_pairs_per_task cannot be negative")

    selected_samples = select_sample_indices(
        run_root,
        max_rollouts=max_rollouts,
        sample_indices=sample_indices,
    )
    grouped: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    records = load_evaluated_trajectories(run_root, sample_indices=selected_samples)
    for record in records:
        grouped[record.instance_id].append(record)

    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    tasks_with_pairs = 0
    for instance_id, task_records in sorted(grouped.items()):
        prompts = {json.dumps(record.prompt, sort_keys=True) for record in task_records}
        if len(prompts) != 1:
            raise ValueError(
                f"Trajectories for {instance_id} do not share the same initial prompt"
            )
        task_rows: list[dict[str, Any]] = []
        for left, right in combinations(task_records, 2):
            preference = _prefer(
                left,
                right,
                token_metric=token_metric,
                min_cost_ratio=min_cost_ratio,
                include_failure_efficiency_pairs=include_failure_efficiency_pairs,
            )
            if preference is None:
                continue
            chosen, rejected, reason = preference
            chosen_cost = chosen.token_cost(token_metric)
            rejected_cost = rejected.token_cost(token_metric)
            task_rows.append(
                {
                    "id": f"{instance_id}:s{chosen.sample_index}>s{rejected.sample_index}",
                    "instance_id": instance_id,
                    "prompt": chosen.prompt,
                    "chosen": chosen.completion,
                    "rejected": rejected.completion,
                    "preference_reason": reason,
                    "token_metric": token_metric,
                    "chosen_cost": chosen_cost,
                    "rejected_cost": rejected_cost,
                    "cost_ratio": rejected_cost / chosen_cost,
                    "chosen_metadata": chosen.metadata(),
                    "rejected_metadata": rejected.metadata(),
                }
            )
        task_rows.sort(
            key=lambda row: (
                row["preference_reason"] != "task_success",
                -float(row["cost_ratio"]),
                row["id"],
            )
        )
        if max_pairs_per_task:
            task_rows = task_rows[:max_pairs_per_task]
        if task_rows:
            tasks_with_pairs += 1
        reason_counts.update(str(row["preference_reason"]) for row in task_rows)
        rows.extend(task_rows)

    count = write_jsonl(output_path, rows)
    summary = {
        "schema_version": 1,
        "objective": "dmpo",
        "run_root": str(run_root),
        "output_path": str(output_path),
        "token_metric": token_metric,
        "min_cost_ratio": min_cost_ratio,
        "include_failure_efficiency_pairs": include_failure_efficiency_pairs,
        "max_pairs_per_task": max_pairs_per_task,
        "max_rollouts": max_rollouts,
        "selected_sample_indices": selected_samples,
        "selected_temperature_counts": selected_temperature_counts(
            run_root, selected_samples
        ),
        "evaluated_trajectories": len(records),
        "resolved_trajectories": sum(record.resolved for record in records),
        "tasks": len(grouped),
        "tasks_with_pairs": tasks_with_pairs,
        "pairs": count,
        "preference_reason_counts": dict(sorted(reason_counts.items())),
        "artifacts": {
            "pairs": artifact_record(
                output_path,
                rows=count,
                summary_path=summary_path,
            )
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
    parser.add_argument("--summary-output")
    parser.add_argument("--token-metric", choices=sorted(TOKEN_METRICS), default="total_tokens")
    parser.add_argument("--min-cost-ratio", type=float, default=1.1)
    parser.add_argument("--include-failure-efficiency-pairs", action="store_true")
    parser.add_argument("--max-pairs-per-task", type=int, default=0)
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
    output_dir = Path(args.run_root) / "preference-data" / "dmpo"
    build_dmpo_pairs(
        args.run_root,
        output_path=args.output or output_dir / "pairs.jsonl",
        summary_path=args.summary_output or output_dir / "summary.json",
        token_metric=args.token_metric,
        min_cost_ratio=args.min_cost_ratio,
        include_failure_efficiency_pairs=args.include_failure_efficiency_pairs,
        max_pairs_per_task=args.max_pairs_per_task,
        max_rollouts=args.max_rollouts,
        sample_indices=(
            parse_sample_indices(args.sample_indices) if args.sample_indices else None
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
