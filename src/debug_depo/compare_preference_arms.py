"""Compare preference-model evaluations under a success constraint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from debug_depo.efficiency import (
    EFFICIENCY_METRICS,
    distribution,
    numeric_value,
    resolved_value,
    summarize_efficiency,
)
from debug_depo.utils import write_json


SCORED_EVALUATION_STATUSES = frozenset(
    {
        "resolved",
        "unresolved",
        "completed",
        "cached_report",
        "empty_patch",
        "patch_failed",
        "timeout",
    }
)

FIELD_CANDIDATES = {
    "action_steps": ("agent_action_steps", "model_api_calls"),
    "prompt_tokens": ("prompt_tokens_total", "prompt_tokens"),
    "completion_tokens": ("completion_tokens_total", "completion_tokens"),
    "total_tokens": ("total_tokens",),
}

DEFAULT_SUCCESS_TOLERANCE = 0.03


def _parse_arm_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=CSV_PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise argparse.ArgumentTypeError(
            "Arm names must contain only letters, numbers, dots, underscores, and hyphens"
        )
    if not raw_path:
        raise argparse.ArgumentTypeError("CSV_PATH must not be empty")
    return name, Path(raw_path).expanduser()


def _metric_fields(fieldnames: Sequence[str] | None, *, path: Path) -> dict[str, str]:
    available = set(fieldnames or ())
    fields: dict[str, str] = {}
    for metric, candidates in FIELD_CANDIDATES.items():
        field = next((candidate for candidate in candidates if candidate in available), None)
        if field is None:
            raise ValueError(
                f"{path} has no supported {metric} column; expected one of {candidates}"
            )
        fields[metric] = field
    return fields


def _load_arm(
    name: str,
    path: Path,
    *,
    require_complete_telemetry: bool,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Arm {name!r} CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        required = {"instance_id", "evaluation_status", "resolved"}
        missing = sorted(required - set(fieldnames or ()))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        fields = _metric_fields(fieldnames, path=path)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Arm {name!r} has no evaluation rows: {path}")

    indexed: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    invalid_statuses: dict[str, str] = {}
    invalid_resolved: list[str] = []
    inconsistent_resolved: list[str] = []
    incomplete_telemetry: dict[str, list[str]] = {}
    for row_number, row in enumerate(rows, 2):
        instance_id = str(row.get("instance_id", "")).strip()
        if not instance_id:
            raise ValueError(f"{path}:{row_number} has an empty instance_id")
        if instance_id in indexed:
            duplicates.append(instance_id)
        indexed[instance_id] = row

        status = str(row.get("evaluation_status", "")).strip()
        if status not in SCORED_EVALUATION_STATUSES:
            invalid_statuses[instance_id] = status or "<empty>"
        resolved = resolved_value(row.get("resolved"))
        if resolved is None:
            invalid_resolved.append(instance_id)
        elif resolved != (status == "resolved"):
            inconsistent_resolved.append(instance_id)
        missing_metrics = [
            metric
            for metric, field in fields.items()
            if numeric_value(row.get(field)) is None
        ]
        if missing_metrics:
            incomplete_telemetry[instance_id] = missing_metrics

    if duplicates:
        raise ValueError(
            f"Arm {name!r} contains duplicate instance IDs: {sorted(set(duplicates))[:10]}"
        )
    if invalid_statuses:
        examples = sorted(invalid_statuses.items())[:10]
        raise ValueError(
            f"Arm {name!r} contains unscored evaluation outcomes: {examples}"
        )
    if invalid_resolved:
        raise ValueError(
            f"Arm {name!r} contains invalid resolved values: {invalid_resolved[:10]}"
        )
    if inconsistent_resolved:
        raise ValueError(
            f"Arm {name!r} has resolved values inconsistent with evaluation_status: "
            f"{inconsistent_resolved[:10]}"
        )
    if require_complete_telemetry and incomplete_telemetry:
        examples = sorted(incomplete_telemetry.items())[:10]
        raise ValueError(
            f"Arm {name!r} contains incomplete efficiency telemetry: {examples}"
        )

    ordered_ids = sorted(indexed)
    ordered_rows = [indexed[instance_id] for instance_id in ordered_ids]
    return {
        "name": name,
        "path": path.resolve(),
        "ids": ordered_ids,
        "rows": ordered_rows,
        "by_id": indexed,
        "fields": fields,
    }


def _validate_task_matrix(baseline: Mapping[str, Any], arm: Mapping[str, Any]) -> None:
    baseline_ids = set(baseline["ids"])
    arm_ids = set(arm["ids"])
    missing = sorted(baseline_ids - arm_ids)
    unexpected = sorted(arm_ids - baseline_ids)
    if missing or unexpected:
        raise ValueError(
            f"Arm {arm['name']!r} does not match baseline task IDs; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )


def _paired_deltas(
    baseline: Mapping[str, Any],
    arm: Mapping[str, Any],
    *,
    instance_ids: Sequence[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"tasks": len(instance_ids)}
    for metric in EFFICIENCY_METRICS:
        baseline_field = baseline["fields"][metric]
        arm_field = arm["fields"][metric]
        values: list[float] = []
        for instance_id in instance_ids:
            baseline_value = numeric_value(
                baseline["by_id"][instance_id].get(baseline_field)
            )
            arm_value = numeric_value(arm["by_id"][instance_id].get(arm_field))
            if baseline_value is not None and arm_value is not None:
                values.append(arm_value - baseline_value)
        summary[metric] = distribution(values, expected=len(instance_ids))
    return summary


def _resolution_transitions(
    baseline: Mapping[str, Any],
    arm: Mapping[str, Any],
) -> dict[str, int]:
    transitions = {
        "both_resolved": 0,
        "both_unresolved": 0,
        "gained": 0,
        "lost": 0,
    }
    for instance_id in baseline["ids"]:
        baseline_resolved = resolved_value(baseline["by_id"][instance_id]["resolved"])
        arm_resolved = resolved_value(arm["by_id"][instance_id]["resolved"])
        if baseline_resolved and arm_resolved:
            transitions["both_resolved"] += 1
        elif not baseline_resolved and not arm_resolved:
            transitions["both_unresolved"] += 1
        elif arm_resolved:
            transitions["gained"] += 1
        else:
            transitions["lost"] += 1
    return transitions


def compare_preference_arms(
    *,
    baseline: tuple[str, Path],
    arms: Sequence[tuple[str, Path]],
    output: str | Path,
    success_tolerance: float = DEFAULT_SUCCESS_TOLERANCE,
    expected_tasks: int | None = None,
    allow_incomplete_telemetry: bool = False,
) -> dict[str, Any]:
    """Select the cheapest success-noninferior arm on an exact task matrix."""

    if not 0 <= success_tolerance <= 1:
        raise ValueError("success_tolerance must be between 0 and 1")
    if not arms:
        raise ValueError("At least one candidate arm is required")
    if expected_tasks is not None and expected_tasks < 1:
        raise ValueError("expected_tasks must be at least 1")

    specs = [baseline, *arms]
    names = [name for name, _ in specs]
    duplicate_names = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicate_names:
        raise ValueError(f"Duplicate arm names: {duplicate_names}")

    loaded = [
        _load_arm(
            name,
            Path(path),
            require_complete_telemetry=not allow_incomplete_telemetry,
        )
        for name, path in specs
    ]
    baseline_data = loaded[0]
    if expected_tasks is not None and len(baseline_data["ids"]) != expected_tasks:
        raise ValueError(
            f"Expected {expected_tasks} tasks, found {len(baseline_data['ids'])}"
        )
    for arm in loaded[1:]:
        _validate_task_matrix(baseline_data, arm)

    baseline_efficiency = summarize_efficiency(
        baseline_data["rows"], fields=baseline_data["fields"]
    )
    baseline_rate = float(baseline_efficiency["resolution_rate"])
    success_threshold = max(0.0, baseline_rate - success_tolerance)

    arm_summaries: list[dict[str, Any]] = []
    for index, arm in enumerate(loaded):
        efficiency = summarize_efficiency(arm["rows"], fields=arm["fields"])
        resolution_rate = float(efficiency["resolution_rate"])
        total_tokens_per_resolved = efficiency["total_tokens_per_resolved_task"]
        success_noninferior = resolution_rate >= success_threshold
        rankable = total_tokens_per_resolved is not None
        both_resolved_ids = [
            instance_id
            for instance_id in baseline_data["ids"]
            if resolved_value(baseline_data["by_id"][instance_id]["resolved"])
            and resolved_value(arm["by_id"][instance_id]["resolved"])
        ]
        arm_summaries.append(
            {
                "name": arm["name"],
                "path": str(arm["path"]),
                "is_baseline": index == 0,
                "efficiency": efficiency,
                "resolution_rate_delta_vs_baseline": resolution_rate - baseline_rate,
                "success_noninferior": success_noninferior,
                "rankable": rankable,
                "selection_eligible": success_noninferior and rankable,
                "resolution_transitions_vs_baseline": _resolution_transitions(
                    baseline_data, arm
                ),
                "paired_deltas_vs_baseline": {
                    "all": _paired_deltas(
                        baseline_data,
                        arm,
                        instance_ids=baseline_data["ids"],
                    ),
                    "both_resolved": _paired_deltas(
                        baseline_data,
                        arm,
                        instance_ids=both_resolved_ids,
                    ),
                },
            }
        )

    ranking = sorted(
        (
            {
                "name": arm["name"],
                "total_tokens_per_resolved_task": arm["efficiency"][
                    "total_tokens_per_resolved_task"
                ],
                "resolution_rate": arm["efficiency"]["resolution_rate"],
            }
            for arm in arm_summaries
            if arm["selection_eligible"]
        ),
        key=lambda arm: (
            float(arm["total_tokens_per_resolved_task"]),
            -float(arm["resolution_rate"]),
            names.index(str(arm["name"])),
        ),
    )
    summary = {
        "schema_version": 1,
        "baseline": baseline_data["name"],
        "task_count": len(baseline_data["ids"]),
        "task_matrix": "exact_instance_id_match",
        "task_matrix_sha256": hashlib.sha256(
            ("\n".join(baseline_data["ids"]) + "\n").encode()
        ).hexdigest(),
        "expected_tasks": expected_tasks,
        "success_tolerance": success_tolerance,
        "allow_incomplete_telemetry": allow_incomplete_telemetry,
        "baseline_resolution_rate": baseline_rate,
        "success_threshold": success_threshold,
        "ranking_metric": "total_tokens_per_resolved_task",
        "cost_scope": "all_evaluated_attempts",
        "selected_arm": ranking[0]["name"] if ranking else None,
        "eligible_ranking": ranking,
        "arms": arm_summaries,
    }
    write_json(output, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare evaluation CSVs on identical tasks and select the lowest-token "
            "success-noninferior arm."
        )
    )
    parser.add_argument(
        "--baseline",
        required=True,
        type=_parse_arm_spec,
        metavar="NAME=CSV_PATH",
    )
    parser.add_argument(
        "--allow-incomplete-telemetry",
        action="store_true",
        help=(
            "Permit missing step/token values for exploratory summaries; incomplete "
            "total-token arms remain ineligible for selection."
        ),
    )
    parser.add_argument(
        "--arm",
        required=True,
        action="append",
        type=_parse_arm_spec,
        metavar="NAME=CSV_PATH",
        help="Candidate arm; repeat for every model or checkpoint.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-tasks",
        type=int,
        help="Reject the comparison unless the exact task matrix has this size.",
    )
    parser.add_argument(
        "--success-tolerance",
        type=float,
        default=DEFAULT_SUCCESS_TOLERANCE,
        help=(
            "Maximum absolute resolution-rate drop from baseline (default: 0.03, "
            "or three percentage points)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = compare_preference_arms(
        baseline=args.baseline,
        arms=args.arm,
        output=args.output,
        success_tolerance=args.success_tolerance,
        expected_tasks=args.expected_tasks,
        allow_incomplete_telemetry=args.allow_incomplete_telemetry,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["selected_arm"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
