"""Analyze repeated SWE-smith collection and evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from debug_depo.efficiency import summarize_efficiency
from debug_depo.utils import read_json, read_jsonl, write_json


ROLLOUT_COLUMNS = (
    "instance_id",
    "repo",
    "shard",
    "sample_index",
    "temperature",
    "temperature_run_index",
    "seed",
    "collection_status",
    "patch_present",
    "patch_chars",
    "trajectory_messages",
    "model_api_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "evaluation_status",
    "resolved",
    "trajectory_path",
    "raw_trajectory_path",
    "evaluation_report_path",
)

TASK_COLUMNS = (
    "instance_id",
    "repo",
    "runs_expected",
    "runs_collected",
    "runs_with_patch",
    "runs_evaluated",
    "runs_resolved",
    "mixed_temperature_pass_at_1",
    "mixed_temperature_pass_at_4",
    "resolved_at_least_once",
    "resolved_samples",
)

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
COMPLETED_COLLECTION_STATUSES = frozenset({"completed", "mocked", "model_terminated"})


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _raw_trajectory(instance_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    paths = sorted(instance_dir.glob("**/*.traj.json"), key=lambda path: path.stat().st_mtime)
    if not paths:
        return None, {}
    path = paths[-1]
    return path, _safe_json(path)


def _usage_metrics(raw: dict[str, Any]) -> dict[str, int | str]:
    messages = raw.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    prompt = 0
    completion = 0
    total = 0
    has_usage = False
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        extra = message.get("extra", {})
        response = extra.get("response", {}) if isinstance(extra, dict) else {}
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        if not isinstance(usage, dict):
            continue
        for key, accumulator in (
            ("prompt_tokens", "prompt"),
            ("completion_tokens", "completion"),
            ("total_tokens", "total"),
        ):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                has_usage = True
                if accumulator == "prompt":
                    prompt += value
                elif accumulator == "completion":
                    completion += value
                else:
                    total += value
    info = raw.get("info", {}) if isinstance(raw.get("info"), dict) else {}
    model_stats = info.get("model_stats", {}) if isinstance(info.get("model_stats"), dict) else {}
    api_calls = model_stats.get("api_calls", "")
    return {
        "trajectory_messages": len(messages) if messages else "",
        "model_api_calls": api_calls if isinstance(api_calls, int) else "",
        "prompt_tokens": prompt if has_usage else "",
        "completion_tokens": completion if has_usage else "",
        "total_tokens": total if has_usage else "",
    }


def _evaluation_index(
    run_root: Path,
    sample_index: int,
) -> dict[str, dict[str, Any]]:
    sample_root = run_root / "evaluation" / f"sample-{sample_index}"
    summary = _safe_json(sample_root / "summary.json")
    outcomes: dict[str, dict[str, Any]] = {}
    for status, instance_ids in summary.get("status_ids", {}).items():
        if isinstance(instance_ids, list):
            for instance_id in instance_ids:
                outcomes[str(instance_id)] = {
                    "evaluation_status": status,
                    "resolved": False,
                    "evaluation_report_path": "",
                }
    for instance_id in summary.get("resolved_ids", []):
        outcomes.setdefault(str(instance_id), {})["evaluation_status"] = "resolved"
        outcomes[str(instance_id)]["resolved"] = True

    logs_root = sample_root / "logs"
    if logs_root.is_dir():
        for report_path in logs_root.glob("*/report.json"):
            report = _safe_json(report_path)
            instance_id = str(report.get("instance_id") or report_path.parent.name)
            existing = outcomes.get(instance_id, {})
            existing_status = str(existing.get("evaluation_status", ""))
            if existing_status and existing_status not in SCORED_EVALUATION_STATUSES:
                continue
            resolved = bool(report.get("resolved", False))
            outcomes[instance_id] = {
                "evaluation_status": "resolved" if resolved else "unresolved",
                "resolved": resolved,
                "evaluation_report_path": str(report_path),
            }
    return outcomes


def _sample_predictions(
    run_root: Path, sample_index: int
) -> tuple[list[dict[str, Any]], list[Path]]:
    merged = run_root / "merged" / f"sample-{sample_index}" / "predictions.jsonl"
    paths = (
        [merged]
        if merged.is_file()
        else sorted(
            run_root.glob(f"collection/shard-*/samples/sample-{sample_index}/predictions.jsonl")
        )
    )
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows, paths


def _trajectory_index(
    run_root: Path,
    sample_index: int,
) -> dict[str, tuple[str, Path, dict[str, Any]]]:
    index: dict[str, tuple[str, Path, dict[str, Any]]] = {}
    pattern = f"collection/shard-*/samples/sample-{sample_index}/trajectories/*/trajectory.json"
    for path in sorted(run_root.glob(pattern)):
        wrapper = _safe_json(path)
        instance_id = str(wrapper.get("instance_id") or path.parent.name)
        shard = next(
            (part for part in path.parts if part.startswith("shard-")),
            "",
        )
        index[instance_id] = (shard, path, wrapper)
    return index


def _task_repo(instance_id: str, wrapper_path: Path | None) -> str:
    if wrapper_path is not None:
        task = _safe_json(wrapper_path.parent / "task.json")
        if task.get("repo"):
            return str(task["repo"])
    return instance_id.split("__", 1)[0]


def _pass_at_k(n: int, c: int, k: int) -> float:
    if n < k or n <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _is_scored_evaluation(row: dict[str, Any]) -> bool:
    return (
        str(row["collection_status"]) in COMPLETED_COLLECTION_STATUSES
        and str(row["evaluation_status"]) in SCORED_EVALUATION_STATUSES
    )


def _has_complete_temperature_layout(
    rows: list[dict[str, Any]],
    *,
    total_samples: int,
    runs_per_temperature: int,
) -> bool:
    if len(rows) != total_samples:
        return False
    sample_indices = {int(row["sample_index"]) for row in rows}
    if len(sample_indices) != total_samples:
        return False
    temperature_counts = Counter(str(row["temperature"]) for row in rows)
    expected_temperatures = total_samples // runs_per_temperature
    return len(temperature_counts) == expected_temperatures and set(
        temperature_counts.values()
    ) == {runs_per_temperature}


def _sample_rollout_rows(root: Path, sample_index: int) -> tuple[list[dict[str, Any]], list[Path]]:
    predictions, paths = _sample_predictions(root, sample_index)
    trajectories = _trajectory_index(root, sample_index)
    evaluations = _evaluation_index(root, sample_index)
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        instance_id = str(prediction["instance_id"])
        patch_value = prediction.get("model_patch", "")
        patch = patch_value if isinstance(patch_value, str) else ""
        shard = ""
        wrapper_path: Path | None = None
        wrapper: dict[str, Any] = {}
        if instance_id in trajectories:
            shard, wrapper_path, wrapper = trajectories[instance_id]
        raw_path, raw = _raw_trajectory(wrapper_path.parent) if wrapper_path else (None, {})
        evaluation = evaluations.get(
            instance_id,
            {
                "evaluation_status": "not_evaluated",
                "resolved": False,
                "evaluation_report_path": "",
            },
        )
        config = wrapper.get("config", {}) if isinstance(wrapper.get("config"), dict) else {}
        rows.append(
            {
                "instance_id": instance_id,
                "repo": _task_repo(instance_id, wrapper_path),
                "shard": shard,
                "sample_index": sample_index,
                "temperature": prediction.get("temperature", config.get("temperature", "")),
                "temperature_run_index": prediction.get("temperature_run_index", ""),
                "seed": prediction.get("seed", config.get("seed", "")),
                "collection_status": wrapper.get("status", "missing"),
                "patch_present": bool(patch.strip()),
                "patch_chars": len(patch),
                **_usage_metrics(raw),
                **evaluation,
                "trajectory_path": str(wrapper_path or ""),
                "raw_trajectory_path": str(raw_path or ""),
            }
        )
    return rows, paths


def _task_rows(
    grouped: dict[str, list[dict[str, Any]]], total_samples: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instance_id, rollouts in sorted(grouped.items()):
        evaluated = [row for row in rollouts if _is_scored_evaluation(row)]
        resolved = [row for row in evaluated if row["resolved"]]
        complete = len(evaluated) == total_samples
        resolved_count = len(resolved)
        rows.append(
            {
                "instance_id": instance_id,
                "repo": rollouts[0]["repo"],
                "runs_expected": total_samples,
                "runs_collected": len(rollouts),
                "runs_with_patch": sum(bool(row["patch_present"]) for row in rollouts),
                "runs_evaluated": len(evaluated),
                "runs_resolved": resolved_count,
                "mixed_temperature_pass_at_1": (
                    _pass_at_k(total_samples, resolved_count, 1) if complete else ""
                ),
                "mixed_temperature_pass_at_4": (
                    _pass_at_k(total_samples, resolved_count, min(4, total_samples))
                    if complete
                    else ""
                ),
                "resolved_at_least_once": bool(resolved) if complete else "",
                "resolved_samples": ",".join(str(row["sample_index"]) for row in resolved),
            }
        )
    return rows


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _temperature_summaries(
    scored_evaluations: list[dict[str, Any]], runs_per_temperature: int
) -> dict[str, dict[str, Any]]:
    by_temperature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_evaluations:
        by_temperature[str(row["temperature"])].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for temperature, rows in sorted(by_temperature.items()):
        resolved = sum(bool(row["resolved"]) for row in rows)
        temperature_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            temperature_tasks[str(row["instance_id"])].append(row)
        complete_tasks = [
            task_rows
            for task_rows in temperature_tasks.values()
            if len(task_rows) == runs_per_temperature
        ]
        summaries[temperature] = {
            "evaluated": len(rows),
            "resolved": resolved,
            "resolution_rate": resolved / len(rows) if rows else 0.0,
            "fully_evaluated_tasks": len(complete_tasks),
            "pass_at_k": {
                str(k): (
                    sum(
                        _pass_at_k(
                            runs_per_temperature,
                            sum(bool(row["resolved"]) for row in task_rows),
                            k,
                        )
                        for task_rows in complete_tasks
                    )
                    / len(complete_tasks)
                    if complete_tasks
                    else None
                )
                for k in range(1, runs_per_temperature + 1)
            },
        }
    return summaries


def analyze_swesmith(
    run_root: str | Path,
    *,
    rollouts_csv: str | Path,
    tasks_csv: str | Path,
    summary_output: str | Path,
    total_samples: int = 8,
    runs_per_temperature: int = 4,
    expected_tasks: int | None = None,
) -> dict[str, Any]:
    if total_samples < 1:
        raise ValueError("total_samples must be at least 1")
    if runs_per_temperature < 1:
        raise ValueError("runs_per_temperature must be at least 1")
    if total_samples % runs_per_temperature:
        raise ValueError("total_samples must be divisible by runs_per_temperature")

    root = Path(run_root)
    rollout_rows: list[dict[str, Any]] = []
    source_paths: list[str] = []

    for sample_index in range(total_samples):
        sample_rows, paths = _sample_rollout_rows(root, sample_index)
        source_paths.extend(str(path) for path in paths)
        rollout_rows.extend(sample_rows)

    if not rollout_rows:
        raise FileNotFoundError(f"No SWE-smith sample predictions found under {root}")

    rollout_rows.sort(key=lambda row: (str(row["instance_id"]), int(row["sample_index"])))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        grouped[str(row["instance_id"])].append(row)

    task_rows = _task_rows(grouped, total_samples)
    _write_csv(Path(rollouts_csv), ROLLOUT_COLUMNS, rollout_rows)
    _write_csv(Path(tasks_csv), TASK_COLUMNS, task_rows)

    scored_evaluations = [row for row in rollout_rows if _is_scored_evaluation(row)]
    resolved_rollouts = sum(bool(row["resolved"]) for row in scored_evaluations)
    complete_tasks = [row for row in task_rows if int(row["runs_evaluated"]) == total_samples]
    temperatures = _temperature_summaries(scored_evaluations, runs_per_temperature)

    mixed_temperature_pass_at_k = {
        str(k): (
            sum(_pass_at_k(total_samples, int(row["runs_resolved"]), k) for row in complete_tasks)
            / len(complete_tasks)
            if complete_tasks
            else None
        )
        for k in range(1, total_samples + 1)
    }
    status_counts = Counter(str(row["collection_status"]) for row in rollout_rows)
    evaluation_status_counts = Counter(str(row["evaluation_status"]) for row in rollout_rows)
    unscored_evaluation_status_counts = {
        status: count
        for status, count in sorted(evaluation_status_counts.items())
        if status != "not_evaluated" and status not in SCORED_EVALUATION_STATUSES
    }
    complete_collection = all(
        _has_complete_temperature_layout(
            rows,
            total_samples=total_samples,
            runs_per_temperature=runs_per_temperature,
        )
        and all(str(row["collection_status"]) in COMPLETED_COLLECTION_STATUSES for row in rows)
        for rows in grouped.values()
    )
    summary = {
        "schema_version": 1,
        "run_root": str(root),
        "runs_per_temperature": runs_per_temperature,
        "total_samples_per_task": total_samples,
        "expected_tasks": expected_tasks,
        "tasks": len(task_rows),
        "matches_expected_tasks": (
            len(task_rows) == expected_tasks if expected_tasks is not None else None
        ),
        "expected_rollouts": len(task_rows) * total_samples,
        "rollouts": len(rollout_rows),
        "complete_collection": complete_collection,
        "rollouts_with_patch": sum(bool(row["patch_present"]) for row in rollout_rows),
        "collection_status_counts": dict(sorted(status_counts.items())),
        "evaluated_rollouts": len(scored_evaluations),
        "resolved_rollouts": resolved_rollouts,
        "rollout_resolution_rate": (
            resolved_rollouts / len(scored_evaluations) if scored_evaluations else None
        ),
        "efficiency": summarize_efficiency(
            scored_evaluations,
            fields={
                "action_steps": "model_api_calls",
                "prompt_tokens": "prompt_tokens",
                "completion_tokens": "completion_tokens",
                "total_tokens": "total_tokens",
            },
        ),
        "evaluation_status_counts": dict(sorted(evaluation_status_counts.items())),
        "unscored_evaluation_status_counts": unscored_evaluation_status_counts,
        "fully_evaluated_tasks": len(complete_tasks),
        "tasks_resolved_at_least_once": sum(
            int(row["runs_resolved"]) > 0 for row in complete_tasks
        ),
        "mixed_temperature_pass_at_k": mixed_temperature_pass_at_k,
        "mixed_temperature_policy": (
            "Uniform draws without replacement from the fixed pool containing "
            f"{runs_per_temperature} samples at each temperature."
        ),
        "temperatures": temperatures,
        "prediction_sources": source_paths,
        "rollouts_csv": str(rollouts_csv),
        "tasks_csv": str(tasks_csv),
    }
    write_json(summary_output, summary)
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a repeated SWE-smith run.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--rollouts-csv")
    parser.add_argument("--tasks-csv")
    parser.add_argument("--summary-output")
    parser.add_argument("--total-samples", type=int, default=8)
    parser.add_argument("--runs-per-temperature", type=int, default=4)
    parser.add_argument("--expected-tasks", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.run_root) / "analysis"
    analyze_swesmith(
        args.run_root,
        rollouts_csv=args.rollouts_csv or output_dir / "rollouts.csv",
        tasks_csv=args.tasks_csv or output_dir / "tasks.csv",
        summary_output=args.summary_output or output_dir / "summary.json",
        total_samples=args.total_samples,
        runs_per_temperature=args.runs_per_temperature,
        expected_tasks=args.expected_tasks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
