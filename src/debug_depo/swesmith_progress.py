"""Report read-only progress for a sharded SWE-smith collection run."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


FINISHED_STATUSES = frozenset({"completed", "mocked", "model_terminated"})
SCORED_EVALUATION_STATUSES = frozenset(
    {"completed", "cached_report", "empty_patch", "patch_failed", "timeout"}
)
SHARD_PATTERN = re.compile(r"shard-(\d+)$")


@dataclass
class ShardProgress:
    """Progress derived from one shard manifest and its trajectory files."""

    index: int
    expected_tasks: int
    samples_per_task: int
    collected_task_ids: set[str] = field(default_factory=set)
    started_task_ids: set[str] = field(default_factory=set)
    finished_rollouts: int = 0
    error_rollouts: int = 0
    active_rollouts: int = 0
    unreadable_rollouts: int = 0
    error_reasons: dict[str, int] = field(default_factory=dict)
    error_examples: dict[str, str] = field(default_factory=dict)
    state: str = "running"
    manifest_present: bool = True

    @property
    def expected_rollouts(self) -> int:
        return self.expected_tasks * self.samples_per_task

    @property
    def recorded_rollouts(self) -> int:
        return self.finished_rollouts + self.error_rollouts


@dataclass
class CollectionProgress:
    """Progress for a complete collection directory."""

    run_root: Path
    expected_tasks: int
    samples_per_task: int
    shards: list[ShardProgress]
    expected_task_ids: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)

    @property
    def collected_tasks(self) -> int:
        return len(
            {
                task_id
                for shard in self.shards
                for task_id in shard.collected_task_ids
            }
        )

    @property
    def started_tasks(self) -> int:
        return len(
            {
                task_id
                for shard in self.shards
                for task_id in shard.started_task_ids
            }
        )

    @property
    def expected_rollouts(self) -> int:
        return self.expected_tasks * self.samples_per_task

    @property
    def finished_rollouts(self) -> int:
        return sum(shard.finished_rollouts for shard in self.shards)

    @property
    def error_rollouts(self) -> int:
        return sum(shard.error_rollouts for shard in self.shards)

    @property
    def unreadable_rollouts(self) -> int:
        return sum(shard.unreadable_rollouts for shard in self.shards)

    @property
    def active_rollouts(self) -> int:
        return sum(shard.active_rollouts for shard in self.shards)

    @property
    def recorded_rollouts(self) -> int:
        return self.finished_rollouts + self.error_rollouts

    @property
    def pending_rollouts(self) -> int:
        return max(
            0,
            self.expected_rollouts - self.recorded_rollouts - self.active_rollouts,
        )


@dataclass
class EvaluationSampleProgress:
    """Progress for one temperature/run sample evaluation."""

    index: int
    expected_tasks: int
    evaluated_task_ids: set[str] = field(default_factory=set)
    started_task_ids: set[str] = field(default_factory=set)
    error_count: int = 0
    phase: str = "waiting"


@dataclass
class EvaluationProgress:
    """Evaluation progress across all sample slots."""

    expected_tasks: int
    samples: list[EvaluationSampleProgress]
    warnings: list[str] = field(default_factory=list)

    @property
    def expected_evaluations(self) -> int:
        return self.expected_tasks * len(self.samples)

    @property
    def evaluated(self) -> int:
        return sum(len(sample.evaluated_task_ids) for sample in self.samples)

    @property
    def started(self) -> int:
        return sum(len(sample.started_task_ids) for sample in self.samples)

    @property
    def errors(self) -> int:
        return sum(sample.error_count for sample in self.samples)

    @property
    def fully_evaluated_task_ids(self) -> set[str]:
        if not self.samples:
            return set()
        return set.intersection(
            *(set(sample.evaluated_task_ids) for sample in self.samples)
        )


def _slugify(value: Any, default: str = "item") -> str:
    text = str(value or default).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or default


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _tail_text(path: Path, max_bytes: int = 65_536) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _error_reason(payload: dict[str, Any]) -> str:
    mini_status = payload.get("mini_swe_exit_status")
    if mini_status:
        return str(mini_status)
    returncode = payload.get("returncode")
    if returncode == -9:
        return "process killed (-9)"
    if isinstance(returncode, int) and returncode != 0:
        return f"process exit {returncode}"
    return "error"


def _error_example(trajectory_dir: Path) -> str:
    stderr = _tail_text(trajectory_dir / "stderr.txt", max_bytes=16_384)
    if "Connection refused" in stderr:
        return "connection refused by vLLM"
    if "Connection timed out" in stderr:
        return "vLLM request timed out"
    if "Server disconnected" in stderr:
        return "vLLM disconnected"
    if "No space left on device" in stderr:
        return "no space left on device"
    if "Out of memory" in stderr or "MemoryError" in stderr:
        return "out of memory"
    return ""


def _shard_index(path: Path) -> int | None:
    match = SHARD_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _expected_shard_size(total_tasks: int, num_shards: int, shard_index: int) -> int:
    quotient, remainder = divmod(total_tasks, num_shards)
    return quotient + int(shard_index < remainder)


def _inspect_shard(
    shard_dir: Path,
    manifest: dict[str, Any],
    *,
    fallback_samples: int,
    warnings: list[str],
) -> ShardProgress:
    index = _shard_index(shard_dir)
    if index is None:
        raise ValueError(f"Invalid shard directory name: {shard_dir.name}")

    raw_task_ids = manifest.get("task_instance_ids")
    task_ids = (
        [str(task_id) for task_id in raw_task_ids]
        if isinstance(raw_task_ids, list)
        else []
    )
    expected_tasks = _positive_int(manifest.get("n_tasks")) or len(task_ids)
    samples_per_task = (
        _positive_int(manifest.get("total_samples_per_task")) or fallback_samples
    )
    if not task_ids:
        warnings.append(f"{shard_dir.name}: manifest has no task_instance_ids")
    if expected_tasks != len(task_ids):
        warnings.append(
            f"{shard_dir.name}: manifest n_tasks={expected_tasks} but lists "
            f"{len(task_ids)} task IDs"
        )

    expected_ids = set(task_ids)
    ids_by_slug: dict[str, list[str]] = {}
    for task_id in task_ids:
        ids_by_slug.setdefault(_slugify(task_id), []).append(task_id)

    finished_samples: dict[str, set[int]] = {task_id: set() for task_id in task_ids}
    started_ids: set[str] = set()
    finished_rollouts = 0
    error_rollouts = 0
    active_rollouts = 0
    unreadable_rollouts = 0
    error_reasons: dict[str, int] = {}
    error_examples: dict[str, str] = {}

    for sample_index in range(samples_per_task):
        trajectories_dir = shard_dir / "samples" / f"sample-{sample_index}" / "trajectories"
        try:
            trajectory_dirs = list(trajectories_dir.iterdir())
        except FileNotFoundError:
            continue
        except OSError as exc:
            warnings.append(f"Could not scan {trajectories_dir}: {exc}")
            continue

        for trajectory_dir in trajectory_dirs:
            if not trajectory_dir.is_dir():
                continue
            slug_matches = ids_by_slug.get(trajectory_dir.name, [])
            if len(slug_matches) == 1:
                started_ids.add(slug_matches[0])

            trajectory_path = trajectory_dir / "trajectory.json"
            if not trajectory_path.exists():
                if len(slug_matches) == 1:
                    active_rollouts += 1
                continue
            payload = _read_json(trajectory_path)
            if payload is None:
                unreadable_rollouts += 1
                if len(slug_matches) == 1:
                    active_rollouts += 1
                continue

            instance_id = str(payload.get("instance_id", ""))
            if instance_id not in expected_ids:
                if len(slug_matches) == 1:
                    instance_id = slug_matches[0]
                else:
                    warnings.append(f"Ignoring unexpected trajectory: {trajectory_path}")
                    continue
            started_ids.add(instance_id)

            status = str(payload.get("status", ""))
            if status in FINISHED_STATUSES:
                finished_samples[instance_id].add(sample_index)
                finished_rollouts += 1
            elif status == "error":
                error_rollouts += 1
                reason = _error_reason(payload)
                error_reasons[reason] = error_reasons.get(reason, 0) + 1
                if reason not in error_examples:
                    hint = _error_example(trajectory_dir)
                    if hint:
                        error_examples[reason] = hint

    collected_ids = {
        task_id
        for task_id, sample_indices in finished_samples.items()
        if len(sample_indices) == samples_per_task
    }
    vllm_tail_lines = _tail_text(shard_dir / "vllm.log").rstrip().splitlines()
    vllm_stopped = "Killed" in "\n".join(vllm_tail_lines[-5:])
    if (shard_dir / "summary.json").is_file():
        state = "complete" if error_rollouts == 0 else "failed"
    elif vllm_stopped:
        state = "vLLM down"
    elif active_rollouts or finished_rollouts or error_rollouts:
        state = "running"
    else:
        state = "starting"
    return ShardProgress(
        index=index,
        expected_tasks=expected_tasks,
        samples_per_task=samples_per_task,
        collected_task_ids=collected_ids,
        started_task_ids=started_ids,
        finished_rollouts=finished_rollouts,
        error_rollouts=error_rollouts,
        active_rollouts=active_rollouts,
        unreadable_rollouts=unreadable_rollouts,
        error_reasons=error_reasons,
        error_examples=error_examples,
        state=state,
    )


def inspect_collection(run_root: str | Path) -> CollectionProgress:
    """Read manifests and trajectories without modifying collection state."""

    root = Path(run_root).expanduser().resolve()
    collection_dir = root / "collection"
    if not collection_dir.is_dir():
        raise FileNotFoundError(f"Collection directory does not exist: {collection_dir}")

    warnings: list[str] = []
    manifests: dict[int, tuple[Path, dict[str, Any]]] = {}
    for manifest_path in collection_dir.glob("shard-*/collection_manifest.json"):
        index = _shard_index(manifest_path.parent)
        if index is None:
            continue
        manifest = _read_json(manifest_path)
        if manifest is None:
            warnings.append(f"Manifest is currently unreadable: {manifest_path}")
            continue
        manifests[index] = (manifest_path.parent, manifest)

    if not manifests:
        raise FileNotFoundError(
            f"No readable collection manifests found under {collection_dir}"
        )

    first_manifest = manifests[min(manifests)][1]
    expected_tasks = _positive_int(first_manifest.get("expected_tasks"))
    num_shards = _positive_int(first_manifest.get("num_shards"))
    samples_per_task = _positive_int(first_manifest.get("total_samples_per_task"))
    if expected_tasks is None or num_shards is None or samples_per_task is None:
        raise ValueError(
            "Collection manifest must contain positive expected_tasks, num_shards, "
            "and total_samples_per_task values"
        )

    shards: list[ShardProgress] = []
    all_expected_ids: set[str] = set()
    for index in range(num_shards):
        manifest_entry = manifests.get(index)
        if manifest_entry is None:
            shards.append(
                ShardProgress(
                    index=index,
                    expected_tasks=_expected_shard_size(
                        expected_tasks,
                        num_shards,
                        index,
                    ),
                    samples_per_task=samples_per_task,
                    state="not started",
                    manifest_present=False,
                )
            )
            continue

        shard_dir, manifest = manifest_entry
        for key, expected in (
            ("expected_tasks", expected_tasks),
            ("num_shards", num_shards),
            ("total_samples_per_task", samples_per_task),
        ):
            if manifest.get(key) != expected:
                warnings.append(
                    f"{shard_dir.name}: {key}={manifest.get(key)!r}, expected {expected}"
                )
        shard = _inspect_shard(
            shard_dir,
            manifest,
            fallback_samples=samples_per_task,
            warnings=warnings,
        )
        raw_ids = manifest.get("task_instance_ids", [])
        shard_ids = {str(task_id) for task_id in raw_ids} if isinstance(raw_ids, list) else set()
        duplicates = all_expected_ids.intersection(shard_ids)
        if duplicates:
            warnings.append(
                f"{shard_dir.name}: {len(duplicates)} task IDs also occur in an earlier shard"
            )
        all_expected_ids.update(shard_ids)
        shards.append(shard)

    if len(manifests) > num_shards:
        warnings.append(
            f"Found {len(manifests)} manifests but the run declares {num_shards} shards"
        )

    return CollectionProgress(
        run_root=root,
        expected_tasks=expected_tasks,
        samples_per_task=samples_per_task,
        shards=shards,
        expected_task_ids=all_expected_ids,
        warnings=warnings,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    return None
                rows.append(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return rows


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _terminal_evaluation_status(log_dir: Path) -> str | None:
    """Infer only evaluation outcomes that have durable terminal artifacts."""

    if _read_json(log_dir / "report.json") is not None and _read_json(
        log_dir / "cache_key.json"
    ) is not None:
        return "completed"

    patch_status = _read_text(log_dir / "patch_status.txt")
    if patch_status == "failed":
        return "patch_failed"

    metadata = _read_json(log_dir / "metadata.json")
    stdout_exists = (log_dir / "apptainer_stdout.txt").is_file()
    stderr_exists = (log_dir / "apptainer_stderr.txt").is_file()
    if metadata is not None and "returncode" not in metadata:
        if stdout_exists and stderr_exists:
            return "timeout"
        return None

    if metadata is not None and "returncode" in metadata:
        if patch_status != "applied":
            return "patch_failed"
        if not (log_dir / "test_output.txt").is_file():
            return "error"
    return None


def _progress_from_evaluation_summary(
    summary: dict[str, Any],
    *,
    expected_ids: set[str],
) -> tuple[set[str], set[str], int]:
    evaluated_ids: set[str] = set()
    started_ids: set[str] = set()
    error_count = 0
    status_ids = summary.get("status_ids")
    if isinstance(status_ids, dict):
        for status, raw_ids in status_ids.items():
            if not isinstance(raw_ids, list):
                continue
            ids = {str(instance_id) for instance_id in raw_ids}
            if expected_ids:
                ids.intersection_update(expected_ids)
            started_ids.update(ids)
            if status in SCORED_EVALUATION_STATUSES:
                evaluated_ids.update(ids)
            else:
                error_count += len(ids)
        return evaluated_ids, started_ids, error_count

    results = summary.get("results")
    if not isinstance(results, list):
        return evaluated_ids, started_ids, error_count
    for result in results:
        if not isinstance(result, dict):
            continue
        instance_id = str(result.get("instance_id", ""))
        if not instance_id or (expected_ids and instance_id not in expected_ids):
            continue
        started_ids.add(instance_id)
        status = str(result.get("status", ""))
        if status in SCORED_EVALUATION_STATUSES:
            evaluated_ids.add(instance_id)
        else:
            error_count += 1
    return evaluated_ids, started_ids, error_count


def inspect_evaluation(
    run_root: str | Path,
    collection: CollectionProgress,
) -> EvaluationProgress:
    """Inspect completed summaries and durable live evaluator artifacts."""

    root = Path(run_root).expanduser().resolve()
    evaluation_root = root / "evaluation"
    merged_root = root / "merged"
    warnings: list[str] = []
    samples: list[EvaluationSampleProgress] = []

    for sample_index in range(collection.samples_per_task):
        sample_dir = evaluation_root / f"sample-{sample_index}"
        summary_path = sample_dir / "summary.json"
        summary = _read_json(summary_path) if summary_path.is_file() else None
        if summary is not None:
            evaluated_ids, started_ids, error_count = _progress_from_evaluation_summary(
                summary,
                expected_ids=collection.expected_task_ids,
            )
            samples.append(
                EvaluationSampleProgress(
                    index=sample_index,
                    expected_tasks=collection.expected_tasks,
                    evaluated_task_ids=evaluated_ids,
                    started_task_ids=started_ids,
                    error_count=error_count,
                    phase="complete",
                )
            )
            continue
        if summary_path.is_file():
            warnings.append(f"Evaluation summary is currently unreadable: {summary_path}")

        predictions_path = merged_root / f"sample-{sample_index}" / "predictions.jsonl"
        prediction_rows = _read_jsonl(predictions_path) if predictions_path.is_file() else None
        if predictions_path.is_file() and prediction_rows is None:
            warnings.append(f"Merged predictions are currently unreadable: {predictions_path}")

        expected_ids = {
            str(row.get("instance_id", ""))
            for row in prediction_rows or []
            if row.get("instance_id")
        }
        if collection.expected_task_ids:
            expected_ids.intersection_update(collection.expected_task_ids)
        empty_patch_ids = {
            str(row["instance_id"])
            for row in prediction_rows or []
            if row.get("instance_id")
            and not str(row.get("model_patch", "")).strip()
            and (not collection.expected_task_ids or str(row["instance_id"]) in expected_ids)
        }
        evaluated_ids = set(empty_patch_ids) if sample_dir.is_dir() else set()
        started_ids = set(evaluated_ids)
        error_count = 0

        logs_dir = sample_dir / "logs"
        try:
            log_dirs = list(logs_dir.iterdir())
        except (FileNotFoundError, NotADirectoryError):
            log_dirs = []
        except OSError as exc:
            warnings.append(f"Could not scan {logs_dir}: {exc}")
            log_dirs = []
        for log_dir in log_dirs:
            if not log_dir.is_dir():
                continue
            instance_id = log_dir.name
            if expected_ids and instance_id not in expected_ids:
                continue
            started_ids.add(instance_id)
            status = _terminal_evaluation_status(log_dir)
            if status in SCORED_EVALUATION_STATUSES:
                evaluated_ids.add(instance_id)
            elif status == "error":
                error_count += 1

        phase = "running" if sample_dir.is_dir() and prediction_rows is not None else "waiting"
        samples.append(
            EvaluationSampleProgress(
                index=sample_index,
                expected_tasks=collection.expected_tasks,
                evaluated_task_ids=evaluated_ids,
                started_task_ids=started_ids,
                error_count=error_count,
                phase=phase,
            )
        )

    return EvaluationProgress(
        expected_tasks=collection.expected_tasks,
        samples=samples,
        warnings=warnings,
    )


def _progress_bar(completed: int, expected: int, width: int) -> str:
    ratio = min(max(completed / expected, 0.0), 1.0) if expected else 0.0
    filled = min(width, int(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {completed:,}/{expected:,} ({ratio:6.2%})"


def render_progress(
    collection: CollectionProgress,
    evaluation: EvaluationProgress,
    *,
    bar_width: int = 24,
) -> str:
    """Render compact collection and evaluation progress reports."""

    progress_column_width = bar_width + 27
    lines = [
        "SWE-smith run progress",
        f"Run ID: {collection.run_root.name}",
        f"Run root: {collection.run_root}",
        f"Checked: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        (
            f"Expected: {collection.expected_tasks:,} tasks x "
            f"{collection.samples_per_task} samples = "
            f"{collection.expected_rollouts:,} collection/evaluation slots"
        ),
        "",
        "COLLECTION",
        f"{'Shard':<10} {'State':<12} {'Complete tasks':<{progress_column_width}} "
        f"{'Usable rollouts':>20} {'Failed':>8} {'Active':>8}",
    ]
    for shard in collection.shards:
        label = f"shard-{shard.index}"
        task_bar = _progress_bar(
            len(shard.collected_task_ids),
            shard.expected_tasks,
            bar_width,
        )
        rollout_count = f"{shard.finished_rollouts:,}/{shard.expected_rollouts:,}"
        lines.append(
            f"{label:<10} {shard.state:<12} {task_bar:<{progress_column_width}} "
            f"{rollout_count:>20} "
            f"{shard.error_rollouts:>8,} {shard.active_rollouts:>8,}"
        )

    collection_bar = _progress_bar(
        collection.collected_tasks,
        collection.expected_tasks,
        bar_width,
    )
    collection_rollouts = (
        f"{collection.finished_rollouts:,}/{collection.expected_rollouts:,}"
    )
    lines.extend(
        [
            "-" * (bar_width + 95),
            f"{'OVERALL':<10} {'':<12} {collection_bar:<{progress_column_width}} "
            f"{collection_rollouts:>20} "
            f"{collection.error_rollouts:>8,} {collection.active_rollouts:>8,}",
            (
                f"Recorded rollout outcomes: {collection.recorded_rollouts:,}/"
                f"{collection.expected_rollouts:,} "
                f"({collection.finished_rollouts:,} usable, "
                f"{collection.error_rollouts:,} failed); "
                f"{collection.active_rollouts:,} active; "
                f"{collection.pending_rollouts:,} without an outcome yet."
            ),
            (
                f"Complete task = all {collection.samples_per_task} rollouts are usable. "
                "Failed rollout artifacts do not count as complete and are retried on resume."
            ),
            "",
            "EVALUATION",
            f"{'Sample':<11} {'Evaluated tasks':<{progress_column_width}} "
            f"{'State':>10} {'Started':>9} {'Errors':>8}",
        ]
    )
    for sample in evaluation.samples:
        evaluated_bar = _progress_bar(
            len(sample.evaluated_task_ids),
            sample.expected_tasks,
            bar_width,
        )
        lines.append(
            f"{f'sample-{sample.index}':<11} "
            f"{evaluated_bar:<{progress_column_width}} "
            f"{sample.phase:>10} {len(sample.started_task_ids):>9,} "
            f"{sample.error_count:>8,}"
        )

    evaluation_bar = _progress_bar(
        evaluation.evaluated,
        evaluation.expected_evaluations,
        bar_width,
    )
    fully_evaluated_bar = _progress_bar(
        len(evaluation.fully_evaluated_task_ids),
        evaluation.expected_tasks,
        bar_width,
    )
    lines.extend(
        [
            "-" * (bar_width + 75),
            f"{'OVERALL':<11} {evaluation_bar:<{progress_column_width}} "
            f"{'':>10} {evaluation.started:>9,} {evaluation.errors:>8,}",
            f"Fully evaluated tasks: {fully_evaluated_bar}",
        ]
    )

    error_shards = [shard for shard in collection.shards if shard.error_rollouts]
    if error_shards:
        lines.append("\nCollection failure breakdown (rollout attempts, not tasks):")
        for shard in error_shards:
            details = []
            for reason, count in sorted(
                shard.error_reasons.items(),
                key=lambda item: (-item[1], item[0]),
            ):
                example = shard.error_examples.get(reason)
                detail = f"{count:,} {reason}"
                if example:
                    detail += f"; example: {example}"
                details.append(detail)
            lines.append(f"  - shard-{shard.index}: " + "; ".join(details))
    stopped_shards = [shard.index for shard in collection.shards if shard.state == "vLLM down"]
    if stopped_shards:
        labels = ", ".join(f"shard-{index}" for index in stopped_shards)
        lines.append(
            f"\nAlert: {labels} vLLM log ends with a killed server process; "
            "model connection failures may continue accumulating in that shard."
        )
    if any(not shard.manifest_present for shard in collection.shards):
        lines.append(
            "\nNot started = no manifest exists yet. The script cannot distinguish "
            "queued array elements from elements that failed before manifest creation."
        )
    if all(sample.phase == "waiting" for sample in evaluation.samples):
        lines.append(
            "Evaluation is waiting because no evaluation sample has started; the PBS "
            "dependency normally releases it only after the whole collection array succeeds."
        )
    if collection.unreadable_rollouts:
        lines.append(
            f"\nNote: {collection.unreadable_rollouts:,} trajectory JSON file(s) were unreadable "
            "and counted as unfinished."
        )
    warnings = [*collection.warnings, *evaluation.warnings]
    if warnings:
        lines.append("\nWarnings:")
        lines.extend(f"  - {warning}" for warning in warnings)
    return "\n".join(lines)


def _candidate_runs_roots(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for variable in ("DEBUG_DEPO_SCRATCH", "DEBUG_DEPO_EPHEMERAL"):
        if value := os.getenv(variable):
            candidates.append(Path(value).expanduser() / "runs")
    if value := os.getenv("RDS"):
        candidates.append(Path(value).expanduser() / "ephemeral" / "debug-depo" / "runs")
    for variable in ("EPHEMERAL", "SCRATCH"):
        if value := os.getenv(variable):
            candidates.append(Path(value).expanduser() / "debug-depo" / "runs")
    candidates.extend(
        [
            repo_root / "scratch" / "runs",
            repo_root / "scratch" / "cluster-artifacts" / "runs",
        ]
    )
    return list(dict.fromkeys(path.resolve() for path in candidates))


def resolve_run_root(run: str, *, repo_root: Path) -> Path:
    """Resolve the required run ID, also accepting an explicit run-root path."""

    roots = _candidate_runs_roots(repo_root)
    supplied = Path(run).expanduser()
    if supplied.is_dir():
        supplied = supplied.resolve()
        return supplied.parent if supplied.name == "collection" else supplied
    if supplied.is_absolute() or supplied.parent != Path("."):
        raise FileNotFoundError(f"Run path does not exist: {supplied}")
    matches = [root / run for root in roots if (root / run / "collection").is_dir()]
    if matches:
        return matches[0]
    searched = "\n  ".join(str(root / run) for root in roots)
    raise FileNotFoundError(f"Could not find run {run!r}. Searched:\n  {searched}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Show read-only SWE-smith collection and evaluation progress for one run ID."
        )
    )
    parser.add_argument(
        "run",
        metavar="RUN_ID",
        help="run ID (an explicit run-root path is also accepted)",
    )
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="refresh repeatedly at this interval (minimum: 1 second)",
    )
    parser.add_argument("--bar-width", type=int, default=24)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.watch is not None and args.watch < 1:
        raise SystemExit("--watch must be at least 1 second")
    if args.bar_width < 5:
        raise SystemExit("--bar-width must be at least 5")

    repo_root = Path(__file__).resolve().parents[2]
    try:
        run_root = resolve_run_root(args.run, repo_root=repo_root)
    except (FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    first_render = True
    try:
        while True:
            try:
                collection = inspect_collection(run_root)
                evaluation = inspect_evaluation(run_root, collection)
                report = render_progress(
                    collection,
                    evaluation,
                    bar_width=args.bar_width,
                )
            except (FileNotFoundError, ValueError, OSError) as exc:
                report = f"SWE-smith run progress\nRun: {run_root}\nerror: {exc}"
            if not first_render and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(report, flush=True)
            first_render = False
            if args.watch is None:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
