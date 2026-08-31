"""Shared readers and serializers for SWE-smith preference datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from debug_depo.swesmith_analyze import (
    COMPLETED_COLLECTION_STATUSES,
    SCORED_EVALUATION_STATUSES,
    _evaluation_index,
    _raw_trajectory,
    _safe_json,
)
from debug_depo.swesmith_progress import inspect_collection


TOKEN_METRICS = frozenset({"total_tokens", "completion_tokens"})


@dataclass(frozen=True)
class TrajectoryRecord:
    """One evaluated agent rollout with training-ready messages and usage."""

    instance_id: str
    sample_index: int
    collection_status: str
    evaluation_status: str
    resolved: bool
    prompt: list[dict[str, str]]
    completion: list[dict[str, str]]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    steps: int
    trajectory_path: str
    raw_trajectory_path: str

    def token_cost(self, metric: str) -> int:
        if metric not in TOKEN_METRICS:
            raise ValueError(f"Unsupported token metric: {metric}")
        return int(getattr(self, metric))

    def metadata(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "resolved": self.resolved,
            "collection_status": self.collection_status,
            "evaluation_status": self.evaluation_status,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "steps": self.steps,
            "completion_tokens_per_step": self.completion_tokens / self.steps,
            "total_tokens_per_step": self.total_tokens / self.steps,
            "trajectory_path": self.trajectory_path,
            "raw_trajectory_path": self.raw_trajectory_path,
        }


def discover_sample_indices(run_root: str | Path) -> list[int]:
    """Discover sample slots from sharded SWE-smith collection directories."""

    root = Path(run_root)
    indices: set[int] = set()
    for path in root.glob("collection/shard-*/samples/sample-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("sample-")
        if suffix.isdigit():
            indices.add(int(suffix))
    if not indices:
        raise FileNotFoundError(f"No SWE-smith sample directories found under {root}")
    return sorted(indices)


def parse_sample_indices(spec: str) -> list[int]:
    """Parse a comma-, colon-, or whitespace-separated sample-index list."""

    values = [value for value in re.split(r"[,:\s]+", spec.strip()) if value]
    if not values:
        raise ValueError("at least one sample index is required")
    indices = [int(value) for value in values]
    if any(index < 0 for index in indices):
        raise ValueError("sample indices cannot be negative")
    if len(indices) != len(set(indices)):
        raise ValueError("sample indices must be unique")
    return indices


def _collection_layout(run_root: Path) -> tuple[list[float], int] | None:
    layouts: set[tuple[tuple[float, ...], int]] = set()
    for path in run_root.glob("collection/shard-*/collection_manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            temperatures = tuple(float(value) for value in manifest["temperatures"])
            runs_per_temperature = int(manifest["runs_per_temperature"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        layouts.add((temperatures, runs_per_temperature))
    if not layouts:
        return None
    if len(layouts) != 1:
        raise ValueError("Collection shards have incompatible temperature layouts")
    temperatures, runs_per_temperature = layouts.pop()
    return list(temperatures), runs_per_temperature


def select_sample_indices(
    run_root: str | Path,
    *,
    max_rollouts: int = 4,
    sample_indices: Iterable[int] | None = None,
) -> list[int]:
    """Select explicit samples or a deterministic temperature-balanced subset."""

    if max_rollouts < 0:
        raise ValueError("max_rollouts cannot be negative")
    root = Path(run_root)
    available = discover_sample_indices(root)
    available_set = set(available)
    if sample_indices is not None:
        selected = list(sample_indices)
        if not selected:
            raise ValueError("sample_indices cannot be empty")
        if len(selected) != len(set(selected)):
            raise ValueError("sample_indices must be unique")
        missing = sorted(set(selected) - available_set)
        if missing:
            raise ValueError(f"Requested sample indices are unavailable: {missing}")
        return sorted(selected)
    if max_rollouts == 0 or len(available) <= max_rollouts:
        return available

    layout = _collection_layout(root)
    if layout is None:
        return available[:max_rollouts]
    temperatures, runs_per_temperature = layout
    balanced: list[int] = []
    # Collection sample slots are temperature-major. Iterate run-major here so
    # each temperature contributes once before a second rollout is selected.
    for run_index in range(runs_per_temperature):
        for temperature_index in range(len(temperatures)):
            sample_index = temperature_index * runs_per_temperature + run_index
            if sample_index in available_set:
                balanced.append(sample_index)
                if len(balanced) == max_rollouts:
                    return sorted(balanced)
    # Accommodate incomplete or legacy layouts without silently returning fewer
    # samples than requested.
    balanced.extend(index for index in available if index not in set(balanced))
    return sorted(balanced[:max_rollouts])


def selected_temperature_counts(
    run_root: str | Path,
    sample_indices: Iterable[int],
) -> dict[str, int]:
    """Describe the temperature mix selected from a collection manifest."""

    layout = _collection_layout(Path(run_root))
    if layout is None:
        return {}
    temperatures, runs_per_temperature = layout
    counts: dict[str, int] = {}
    for sample_index in sample_indices:
        temperature_index = sample_index // runs_per_temperature
        if temperature_index >= len(temperatures):
            continue
        key = str(temperatures[temperature_index])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _clean_messages(raw: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    messages = raw.get("messages", [])
    if not isinstance(messages, list):
        return [], []

    cleaned = [
        {"role": str(message["role"]), "content": str(message.get("content", ""))}
        for message in messages
        if isinstance(message, dict) and message.get("role") in {"system", "user", "assistant"}
    ]
    first_assistant = next(
        (index for index, message in enumerate(cleaned) if message["role"] == "assistant"),
        None,
    )
    if first_assistant is None:
        return [], []

    prompt = cleaned[:first_assistant]
    completion = cleaned[first_assistant:]
    while completion and completion[-1]["role"] != "assistant":
        completion.pop()
    if not prompt or not completion:
        return [], []
    return prompt, completion


def _usage(raw: dict[str, Any]) -> tuple[int, int, int] | None:
    prompt = 0
    completion = 0
    total = 0
    assistant_calls = 0
    complete_usage_calls = 0
    for message in raw.get("messages", []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        assistant_calls += 1
        extra = message.get("extra", {})
        response = extra.get("response", {}) if isinstance(extra, dict) else {}
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        if not isinstance(usage, dict):
            continue
        prompt_value = usage.get("prompt_tokens")
        completion_value = usage.get("completion_tokens")
        total_value = usage.get("total_tokens")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (prompt_value, completion_value)
        ):
            continue
        prompt += int(prompt_value)
        completion += int(completion_value)
        total += (
            int(total_value)
            if isinstance(total_value, int) and not isinstance(total_value, bool)
            else int(prompt_value) + int(completion_value)
        )
        complete_usage_calls += 1
    if assistant_calls == 0 or complete_usage_calls != assistant_calls:
        return None
    return prompt, completion, total


def _expected_trajectory_keys(
    run_root: Path,
    sample_indices: list[int],
) -> set[tuple[int, str]]:
    """Return the exact selected sample/task matrix declared by the manifests."""

    try:
        progress = inspect_collection(run_root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            "Preference data requires readable, complete collection manifests"
        ) from exc

    problems = list(progress.warnings)
    if len(progress.expected_task_ids) != progress.expected_tasks:
        problems.append(
            "collection manifests declare "
            f"{len(progress.expected_task_ids)} unique task IDs for "
            f"{progress.expected_tasks} expected tasks"
        )
    missing_shards = [
        f"shard-{shard.index}"
        for shard in progress.shards
        if not shard.manifest_present
    ]
    if missing_shards:
        problems.append(f"missing collection manifests: {', '.join(missing_shards)}")
    out_of_range = [
        sample_index
        for sample_index in sample_indices
        if sample_index >= progress.samples_per_task
    ]
    if out_of_range:
        problems.append(
            "selected sample indices exceed the manifest layout: "
            + ", ".join(map(str, out_of_range))
        )
    if problems:
        raise ValueError(
            "Preference data requires a consistent collection manifest: "
            + "; ".join(problems)
        )

    return {
        (sample_index, instance_id)
        for sample_index in sample_indices
        for instance_id in progress.expected_task_ids
    }


def load_evaluated_trajectories(
    run_root: str | Path,
    *,
    sample_indices: Iterable[int] | None = None,
) -> list[TrajectoryRecord]:
    """Load an exact matrix of complete, scored SWE-smith rollout artifacts."""

    root = Path(run_root)
    records: list[TrajectoryRecord] = []
    selected = (
        discover_sample_indices(root)
        if sample_indices is None
        else select_sample_indices(root, sample_indices=sample_indices)
    )
    expected_keys = _expected_trajectory_keys(root, selected)
    for sample_index in selected:
        evaluations = _evaluation_index(root, sample_index)
        pattern = (
            f"collection/shard-*/samples/sample-{sample_index}/"
            "trajectories/*/trajectory.json"
        )
        for wrapper_path in sorted(root.glob(pattern)):
            wrapper = _safe_json(wrapper_path)
            instance_id = str(wrapper.get("instance_id") or wrapper_path.parent.name)
            collection_status = str(wrapper.get("status", "missing"))
            evaluation = evaluations.get(instance_id)
            if (
                collection_status not in COMPLETED_COLLECTION_STATUSES
                or not evaluation
                or str(evaluation.get("evaluation_status")) not in SCORED_EVALUATION_STATUSES
            ):
                continue
            raw_path, raw = _raw_trajectory(wrapper_path.parent)
            if raw_path is None:
                continue
            prompt, completion_messages = _clean_messages(raw)
            usage = _usage(raw)
            if not prompt or not completion_messages or usage is None:
                continue
            prompt_tokens, completion_tokens, total_tokens = usage
            steps = sum(message["role"] == "assistant" for message in completion_messages)
            if steps < 1 or completion_tokens < 1 or total_tokens < 1:
                continue
            records.append(
                TrajectoryRecord(
                    instance_id=instance_id,
                    sample_index=sample_index,
                    collection_status=collection_status,
                    evaluation_status=str(evaluation["evaluation_status"]),
                    resolved=bool(evaluation.get("resolved", False)),
                    prompt=prompt,
                    completion=completion_messages,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    steps=steps,
                    trajectory_path=str(wrapper_path),
                    raw_trajectory_path=str(raw_path),
                )
            )
    records.sort(key=lambda item: (item.instance_id, item.sample_index))
    actual_keys = [(record.sample_index, record.instance_id) for record in records]
    actual_key_set = set(actual_keys)
    key_counts: dict[tuple[int, str], int] = {}
    for key in actual_keys:
        key_counts[key] = key_counts.get(key, 0) + 1
    missing = sorted(expected_keys - actual_key_set)
    unexpected = sorted(actual_key_set - expected_keys)
    duplicates = sorted(key for key, count in key_counts.items() if count != 1)
    if missing or unexpected or duplicates:
        details = []
        if missing:
            details.append(
                f"{len(missing)} missing/unusable (examples: {missing[:3]})"
            )
        if unexpected:
            details.append(f"{len(unexpected)} unexpected (examples: {unexpected[:3]})")
        if duplicates:
            details.append(f"{len(duplicates)} duplicated (examples: {duplicates[:3]})")
        raise ValueError(
            "Preference data requires complete evaluated trajectories for every "
            "selected task/sample slot; " + "; ".join(details)
        )
    return records


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Atomically write rows as JSONL and return their count."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: str | Path, *, rows: int, summary_path: str | Path) -> dict[str, Any]:
    artifact = Path(path).resolve()
    summary_parent = Path(summary_path).resolve().parent
    return {
        "path": os.path.relpath(artifact, summary_parent),
        "rows": rows,
        "sha256": file_sha256(artifact),
    }


def validate_preference_artifacts(
    objective: str,
    data_dir: str | Path,
) -> dict[str, Any]:
    """Require a completed preference-data manifest and matching immutable files."""

    directory = Path(data_dir).expanduser().resolve()
    summary_path = directory / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Preference-data summary is missing: {summary_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Preference-data summary is invalid: {summary_path}") from exc
    if not isinstance(summary, dict) or summary.get("objective") != objective:
        raise ValueError(f"Preference-data summary has the wrong objective: {summary_path}")
    if summary.get("complete") is not True:
        raise ValueError(
            f"Preference-data build is not marked complete: {summary_path}. "
            "Run cluster/submit_preference_data.sh once."
        )
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"Preference-data artifact manifest is missing: {summary_path}")
    for name, record in artifacts.items():
        if not isinstance(record, dict):
            raise ValueError(f"Invalid artifact record {name!r}: {summary_path}")
        relative_path = record.get("path")
        expected_hash = record.get("sha256")
        expected_rows = record.get("rows")
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"Incomplete artifact record {name!r}: {summary_path}")
        artifact_path = (directory / relative_path).resolve()
        if not artifact_path.is_file():
            raise ValueError(f"Preference-data artifact is missing: {artifact_path}")
        if file_sha256(artifact_path) != expected_hash:
            raise ValueError(f"Preference-data artifact hash mismatch: {artifact_path}")
        with artifact_path.open(encoding="utf-8") as handle:
            actual_rows = sum(bool(line.strip()) for line in handle)
        if actual_rows != expected_rows:
            raise ValueError(
                f"Preference-data artifact row count mismatch: {artifact_path} "
                f"({actual_rows} != {expected_rows})"
            )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate immutable preference-data artifacts.")
    parser.add_argument("--objective", choices=("dmpo", "depo"), required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args(argv)
    summary = validate_preference_artifacts(args.objective, args.data_dir)
    print(
        json.dumps(
            {
                "objective": args.objective,
                "data_dir": str(Path(args.data_dir).resolve()),
                "artifacts": summary["artifacts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
