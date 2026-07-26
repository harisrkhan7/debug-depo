"""Shared efficiency metrics for rollout analysis and model selection."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


EFFICIENCY_METRICS = (
    "action_steps",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


def numeric_value(value: Any) -> float | None:
    """Return a finite non-negative number, or ``None`` for missing data."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def resolved_value(value: Any) -> bool | None:
    """Parse the boolean representation used by in-memory and CSV rows."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    return None


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile for a non-empty sequence."""

    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def distribution(values: Sequence[float], *, expected: int | None = None) -> dict[str, Any]:
    """Summarize a metric while making missing-value coverage explicit."""

    count = len(values)
    denominator = count if expected is None else expected
    return {
        "available": count,
        "missing": max(0, denominator - count),
        "coverage": count / denominator if denominator else None,
        "sum": sum(values) if values else None,
        "min": min(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": percentile(values, 0.9),
        "max": max(values) if values else None,
    }


def summarize_efficiency(
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Mapping[str, str],
) -> dict[str, Any]:
    """Aggregate success and cost for one consistently evaluated rollout set."""

    missing_fields = sorted(set(EFFICIENCY_METRICS) - set(fields))
    if missing_fields:
        raise ValueError(f"Missing efficiency field mappings: {missing_fields}")

    resolved_flags: list[bool] = []
    for index, row in enumerate(rows, 1):
        resolved = resolved_value(row.get("resolved"))
        if resolved is None:
            raise ValueError(f"Row {index} has an invalid resolved value")
        resolved_flags.append(resolved)

    resolved_rows = [
        row for row, resolved in zip(rows, resolved_flags, strict=True) if resolved
    ]

    def summarize_group(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {"trajectories": len(group)}
        for metric in EFFICIENCY_METRICS:
            field = fields[metric]
            values = [
                value
                for row in group
                if (value := numeric_value(row.get(field))) is not None
            ]
            summary[metric] = distribution(values, expected=len(group))
        return summary

    all_summary = summarize_group(rows)
    resolved_summary = summarize_group(resolved_rows)
    resolved_count = len(resolved_rows)
    total_tokens = all_summary["total_tokens"]
    complete_total_tokens = (
        total_tokens["available"] == len(rows) and total_tokens["sum"] is not None
    )

    return {
        "trajectories": len(rows),
        "resolved_trajectories": resolved_count,
        "resolution_rate": resolved_count / len(rows) if rows else None,
        "all": all_summary,
        "resolved": resolved_summary,
        "total_tokens_per_resolved_task": (
            total_tokens["sum"] / resolved_count
            if complete_total_tokens and resolved_count
            else None
        ),
    }
