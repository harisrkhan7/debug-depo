"""Build a per-instance CSV from a completed debug-depo run directory."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


CSV_COLUMNS = (
    "instance_id",
    "repo",
    "shard",
    "rollout_status",
    "rollout_returncode",
    "mini_swe_exit_status",
    "agent_exit_status",
    "patch_present",
    "patch_chars",
    "patch_files",
    "patch_added_lines",
    "patch_deleted_lines",
    "patch_source",
    "discarded_error_patch_chars",
    "trajectory_steps",
    "trajectory_messages",
    "agent_action_steps",
    "model_api_calls",
    "prompt_tokens_total",
    "completion_tokens_total",
    "total_tokens",
    "max_prompt_tokens",
    "final_prompt_tokens",
    "finish_reason_stop",
    "finish_reason_length",
    "finish_reason_other",
    "step_limit",
    "step_limit_utilization",
    "context_limit",
    "context_limit_utilization",
    "commands_executed",
    "commands_succeeded",
    "commands_failed",
    "commands_unknown_returncode",
    "repeated_commands",
    "test_command_steps",
    "edit_command_steps",
    "format_errors",
    "command_timeouts",
    "truncated_observations",
    "submission_step",
    "duration_seconds",
    "seconds_per_action",
    "duration_source",
    "duration_is_estimate",
    "evaluation_status",
    "resolved",
    "evaluation_returncode",
    "fail_to_pass_success",
    "fail_to_pass_failure",
    "pass_to_pass_success",
    "pass_to_pass_failure",
    "failure_stage",
    "failure_category",
    "failure_reason",
    "failure_evidence",
    "needs_llm_review",
    "trajectory_path",
    "evaluation_report_path",
)


def _read_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                rows.append(payload)
    return rows


def _prediction_rows(run_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    merged = run_root / "merged" / "predictions.jsonl"
    paths = [merged] if merged.is_file() else sorted(
        (run_root / "rollouts").glob("shard-*/predictions.jsonl")
    )
    if not paths:
        raise FileNotFoundError(
            f"No merged or sharded predictions.jsonl found under {run_root}"
        )
    rows = _read_jsonl(paths)
    ids = [str(row.get("instance_id", "")) for row in rows]
    missing = [index for index, instance_id in enumerate(ids, 1) if not instance_id]
    duplicates = sorted(instance_id for instance_id, count in Counter(ids).items() if count > 1)
    if missing:
        raise ValueError(f"Prediction rows missing instance_id: {missing[:10]}")
    if duplicates:
        raise ValueError(f"Duplicate prediction instance IDs: {duplicates[:10]}")
    return rows, paths


def _prediction_shards(run_root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in sorted((run_root / "rollouts").glob("shard-*/predictions.jsonl")):
        for row in _read_jsonl([path]):
            if row.get("instance_id"):
                index[str(row["instance_id"])] = path.parent.name
    return index


def _trajectory_index(run_root: Path) -> dict[str, tuple[Path, dict[str, Any], str]]:
    index: dict[str, tuple[Path, dict[str, Any], str]] = {}
    for path in sorted((run_root / "rollouts").glob("shard-*/trajectories/*/trajectory.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        instance_id = str(payload.get("instance_id") or path.parent.name)
        shard = path.parents[2].name
        index[instance_id] = (path, payload, shard)
    return index


def _instance_directory_index(run_root: Path) -> dict[str, tuple[Path, str]]:
    index: dict[str, tuple[Path, str]] = {}
    for path in sorted((run_root / "rollouts").glob("shard-*/trajectories/*")):
        if not path.is_dir():
            continue
        task = _read_json(path / "task.json")
        instance_id = str(task.get("instance_id") or path.name) if isinstance(task, dict) else path.name
        index[instance_id] = (path, path.parents[1].name)
    return index


def _summary_results(run_root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in sorted((run_root / "rollouts").glob("shard-*/summary.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        for result in payload.get("results", []):
            if isinstance(result, dict) and result.get("instance_id"):
                index[str(result["instance_id"])] = result
    return index


def _raw_trajectory(instance_dir: Path | None) -> tuple[Path | None, dict[str, Any]]:
    if instance_dir is None:
        return None, {}
    paths = sorted(instance_dir.glob("**/*.traj.json"), key=lambda path: path.stat().st_mtime)
    if not paths:
        return None, {}
    path = paths[-1]
    payload = _read_json(path)
    return path, payload if isinstance(payload, dict) else {}


def _trajectory_step_count(
    raw: dict[str, Any], messages: list[Any], wrapper: dict[str, Any]
) -> int | str:
    explicit_steps = raw.get("steps")
    if isinstance(explicit_steps, list):
        return len(explicit_steps)

    if messages:
        # Match mini-swe-agent's inspector: a step/page ends at each user message,
        # with any trailing messages forming one final incomplete step.
        steps = 0
        has_unfinished_step = False
        for message in messages:
            has_unfinished_step = True
            if isinstance(message, dict) and message.get("role") == "user":
                steps += 1
                has_unfinished_step = False
        return steps + int(has_unfinished_step)

    wrapper_steps = wrapper.get("steps")
    if isinstance(wrapper_steps, list):
        return len(wrapper_steps)
    return ""


def _int_value(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _trajectory_metrics(
    raw: dict[str, Any], messages: list[Any], wrapper: dict[str, Any]
) -> dict[str, int | float | str]:
    assistants = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    users = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]

    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    total_tokens: list[int] = []
    finish_reasons: Counter[str] = Counter()
    commands: list[str] = []
    for message in assistants:
        extra = message.get("extra", {})
        response = extra.get("response", {}) if isinstance(extra, dict) else {}
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        if isinstance(usage, dict):
            if (value := _int_value(usage.get("prompt_tokens"))) is not None:
                prompt_tokens.append(value)
            if (value := _int_value(usage.get("completion_tokens"))) is not None:
                completion_tokens.append(value)
            if (value := _int_value(usage.get("total_tokens"))) is not None:
                total_tokens.append(value)
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if choices and isinstance(choices[0], dict) and choices[0].get("finish_reason"):
            finish_reasons[str(choices[0]["finish_reason"])] += 1

        content = message.get("content", "")
        if isinstance(content, str):
            action_blocks = re.findall(r"```bash\s*\n(.*?)\n```", content, re.DOTALL)
            if len(action_blocks) == 1:
                commands.append(action_blocks[0].strip())

    user_texts = [
        str(message.get("content", ""))
        for message in users
        if isinstance(message.get("content", ""), str)
    ]
    returncodes = [
        int(match)
        for text in user_texts
        for match in re.findall(r"<returncode>(-?\d+)</returncode>", text)
    ]
    commands_succeeded = sum(returncode == 0 for returncode in returncodes)
    commands_failed = sum(returncode != 0 for returncode in returncodes)

    raw_info = raw.get("info", {}) if isinstance(raw.get("info"), dict) else {}
    raw_config = raw_info.get("config", {}) if isinstance(raw_info.get("config"), dict) else {}
    agent_config = raw_config.get("agent", {}) if isinstance(raw_config.get("agent"), dict) else {}
    step_limit = _int_value(agent_config.get("step_limit"))
    if step_limit is None:
        wrapper_config = wrapper.get("config", {}) if isinstance(wrapper.get("config"), dict) else {}
        step_limit = _int_value(wrapper_config.get("max_steps"))

    wrapper_config = wrapper.get("config", {}) if isinstance(wrapper.get("config"), dict) else {}
    context_limit = _int_value(wrapper_config.get("context_length"))
    model_stats = raw_info.get("model_stats", {}) if isinstance(raw_info.get("model_stats"), dict) else {}
    api_calls = _int_value(model_stats.get("api_calls"))
    max_prompt = max(prompt_tokens) if prompt_tokens else None
    action_steps = len(assistants) if messages else None

    stop_count = finish_reasons.get("stop", 0)
    length_count = finish_reasons.get("length", 0)
    other_finish_count = sum(finish_reasons.values()) - stop_count - length_count
    normalized_commands = [" ".join(command.split()) for command in commands]
    test_pattern = re.compile(
        r"(^|[;&|]\s*|\s)(pytest|py\.test|tox|nox|unittest|npm\s+test|cargo\s+test|go\s+test)(\s|$)"
    )
    edit_pattern = re.compile(
        r"(^|[;&|]\s*|\s)(sed\s+-i|perl\s+-pi|apply_patch|edit_via_str_replace|cat\s+.*>|tee\s+)(\s|$)"
    )

    return {
        "agent_action_steps": action_steps if action_steps is not None else "",
        "prompt_tokens_total": sum(prompt_tokens) if prompt_tokens else "",
        "completion_tokens_total": sum(completion_tokens) if completion_tokens else "",
        "total_tokens": sum(total_tokens) if total_tokens else "",
        "max_prompt_tokens": max_prompt if max_prompt is not None else "",
        "final_prompt_tokens": prompt_tokens[-1] if prompt_tokens else "",
        "finish_reason_stop": stop_count if finish_reasons else "",
        "finish_reason_length": length_count if finish_reasons else "",
        "finish_reason_other": other_finish_count if finish_reasons else "",
        "step_limit": step_limit if step_limit is not None else "",
        "step_limit_utilization": round(api_calls / step_limit, 6)
        if api_calls is not None and step_limit
        else "",
        "context_limit": context_limit if context_limit is not None else "",
        "context_limit_utilization": round(max_prompt / context_limit, 6)
        if max_prompt is not None and context_limit
        else "",
        "commands_executed": len(commands) if messages else "",
        "commands_succeeded": commands_succeeded if messages else "",
        "commands_failed": commands_failed if messages else "",
        "commands_unknown_returncode": max(0, len(commands) - len(returncodes))
        if messages
        else "",
        "repeated_commands": len(normalized_commands) - len(set(normalized_commands))
        if messages
        else "",
        "test_command_steps": sum(bool(test_pattern.search(command)) for command in commands)
        if messages
        else "",
        "edit_command_steps": sum(bool(edit_pattern.search(command)) for command in commands)
        if messages
        else "",
        "format_errors": sum(
            "Please always provide EXACTLY ONE action" in text or "FormatError" in text
            for text in user_texts
        )
        if messages
        else "",
        "command_timeouts": sum(
            "timed out and has been killed" in text or "ExecutionTimeoutError" in text
            for text in user_texts
        )
        if messages
        else "",
        "truncated_observations": sum(
            "<elided_chars>" in text or "output of your last command was too long" in text
            for text in user_texts
        )
        if messages
        else "",
        "submission_step": action_steps
        if raw_info.get("exit_status") == "Submitted" and action_steps is not None
        else "",
    }


def _aggregate_evaluation(run_root: Path) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    reports_dir = run_root / "evaluation" / "reports"
    precedence = {
        "incomplete": 0,
        "error": 1,
        "empty_patch": 2,
        "unresolved": 3,
        "resolved": 4,
    }
    for path in sorted(reports_dir.glob("*.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        for field, status in (
            ("incomplete_ids", "incomplete"),
            ("error_ids", "error"),
            ("empty_patch_ids", "empty_patch"),
            ("unresolved_ids", "unresolved"),
            ("resolved_ids", "resolved"),
        ):
            for instance_id in payload.get(field, []):
                key = str(instance_id)
                if precedence[status] >= precedence.get(outcomes.get(key, "incomplete"), 0):
                    outcomes[key] = status
    return outcomes


def _evaluation_artifacts(
    run_root: Path, instance_ids: set[str]
) -> tuple[dict[str, Path], dict[str, Path]]:
    report_paths: dict[str, Path] = {}
    log_dirs: dict[str, Path] = {}
    logs_root = run_root / "evaluation" / "logs"
    if not logs_root.is_dir():
        return report_paths, log_dirs
    for path in logs_root.rglob("report.json"):
        payload = _read_json(path)
        if isinstance(payload, dict):
            for instance_id in payload:
                key = str(instance_id)
                if key in instance_ids:
                    report_paths[key] = path
                    log_dirs[key] = path.parent
    for path in logs_root.rglob("*"):
        if path.is_dir() and path.name in instance_ids:
            log_dirs.setdefault(path.name, path)
    return report_paths, log_dirs


def _entry_for_instance(payload: Any, instance_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    entry = payload.get(instance_id, payload)
    return entry if isinstance(entry, dict) else {}


def _count_test_status(entry: dict[str, Any], group: str, outcome: str) -> int | str:
    tests_status = entry.get("tests_status", {})
    if not isinstance(tests_status, dict):
        return ""
    group_value = tests_status.get(group, tests_status.get(group.lower(), {}))
    if not isinstance(group_value, dict):
        return ""
    values = group_value.get(outcome, group_value.get(outcome.lower()))
    if isinstance(values, (list, tuple, set, dict)):
        return len(values)
    if isinstance(values, int):
        return values
    return ""


def _patch_stats(patch: str) -> tuple[int, int, int]:
    files: set[str] = set()
    added = 0
    deleted = 0
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            if match:
                files.add(match.group(2))
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1
    return len(files), added, deleted


def _duration(
    wrapper_path: Path | None, wrapper: dict[str, Any], instance_dir: Path | None
) -> tuple[float | str, str, bool | str]:
    value = wrapper.get("duration_seconds")
    if isinstance(value, (int, float)) and value >= 0:
        return round(float(value), 3), "recorded", False

    if wrapper_path is None or instance_dir is None:
        return "", "", ""
    end = wrapper_path.stat().st_mtime
    created_at = wrapper.get("created_at")
    if isinstance(created_at, str):
        try:
            end = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass

    exit_status_paths = sorted(
        instance_dir.glob("exit_statuses_*.yaml"), key=lambda path: path.stat().st_mtime
    )
    if exit_status_paths:
        timestamp = exit_status_paths[-1].stem.removeprefix("exit_statuses_")
        try:
            start = float(timestamp)
        except ValueError:
            pass
        else:
            if end >= start:
                return round(end - start, 3), "mini_swe_start_to_trajectory", True

    task_path = instance_dir / "task.json"
    if not task_path.is_file():
        return "", "", ""
    start = task_path.stat().st_mtime
    if end < start:
        return "", "", ""
    return round(end - start, 3), "task_to_trajectory_mtime", True


def _text_excerpt(paths: Iterable[Path], limit: int = 500) -> str:
    for path in paths:
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - 65_536))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return " | ".join(lines[-6:])[-limit:]
    return ""


def _evaluation_returncode(log_dir: Path | None) -> int | str:
    if log_dir is None:
        return ""
    for path in sorted(log_dir.iterdir()) if log_dir.is_dir() else []:
        if not path.is_file() or path.stat().st_size > 1_000_000:
            continue
        payload = _read_json(path)
        if isinstance(payload, dict) and isinstance(payload.get("returncode"), int):
            return payload["returncode"]
    return ""


def _classify_text(text: str) -> tuple[str, str] | None:
    lowered = text.lower()
    patterns = (
        ("timeout", r"timed? out|timeout|time limit"),
        ("context_limit", r"context.{0,20}(length|window|limit|exceeded)|token limit"),
        ("step_limit", r"step.{0,12}limit|limits?[_ ]?exceeded"),
        ("model_server_error", r"connection refused|api connection|vllm|litellm.*error"),
        ("out_of_memory", r"out of memory|cuda.*oom|oom-kill"),
        ("container_error", r"apptainer|singularity|failed to (build|start).*sandbox"),
        ("permission_error", r"permission denied|read-only file system"),
        ("format_error", r"format[_ ]?error|exactly one action|parse.*action"),
        ("python_exception", r"traceback \(most recent call last\)|uncaught .*error"),
    )
    for category, pattern in patterns:
        if re.search(pattern, lowered):
            return category, text
    return None


def _failure(
    *,
    evaluation_status: str,
    patch_present: bool,
    wrapper: dict[str, Any],
    summary_result: dict[str, Any],
    raw: dict[str, Any],
    rollout_evidence: str,
    evaluation_evidence: str,
) -> tuple[str, str, str, str, bool]:
    if evaluation_status == "resolved":
        return "", "", "", "", False
    if evaluation_status == "unresolved":
        return "evaluation", "tests_failed", "Patch evaluated but did not resolve the task", evaluation_evidence, True
    if evaluation_status == "patch_failed":
        return "evaluation", "patch_apply_failed", "Patch could not be applied", evaluation_evidence, False
    if evaluation_status == "timeout":
        return "evaluation", "timeout", "Evaluation exceeded its timeout", evaluation_evidence, False
    if evaluation_status in {"error", "missing_report", "incomplete"} and patch_present:
        classified = _classify_text(evaluation_evidence)
        category = classified[0] if classified else "evaluation_error"
        return "evaluation", category, "Evaluation did not produce a usable report", evaluation_evidence, not bool(classified)

    mini_status = wrapper.get("mini_swe_exit_status") or summary_result.get("mini_swe_exit_status")
    raw_info = raw.get("info", {}) if isinstance(raw.get("info"), dict) else {}
    agent_status = raw_info.get("exit_status")
    combined_evidence = " | ".join(
        value for value in (str(summary_result.get("error", "")), rollout_evidence) if value
    )
    classified = _classify_text(" | ".join(str(value) for value in (mini_status, agent_status, combined_evidence)))
    if mini_status:
        category = classified[0] if classified else "agent_exit_error"
        discarded_patch = wrapper.get("error_patch")
        discarded_note = (
            f"; collector discarded a {len(discarded_patch)}-character patch because the "
            "mini-swe status was not Submitted"
            if isinstance(discarded_patch, str) and discarded_patch
            else ""
        )
        return (
            "rollout",
            category,
            f"mini-swe-agent exit status: {mini_status}{discarded_note}",
            combined_evidence,
            not bool(classified),
        )
    if wrapper.get("status") == "error" or summary_result.get("status") == "error":
        category = classified[0] if classified else "rollout_process_error"
        return "rollout", category, "Rollout process failed before producing a patch", combined_evidence, not bool(classified)
    if agent_status and str(agent_status) != "Submitted":
        category = classified[0] if classified else "agent_exit_error"
        return "rollout", category, f"Agent exit status: {agent_status}", combined_evidence, not bool(classified)
    if agent_status == "Submitted":
        return "rollout", "submitted_empty_patch", "Agent submitted an empty patch", combined_evidence, True
    if not wrapper:
        return "rollout", "missing_trajectory", "No wrapper trajectory was found", combined_evidence, False
    return "rollout", "missing_patch_artifact", "Rollout completed without a patch artifact", combined_evidence, True


def _evaluation_status(
    *,
    patch_present: bool,
    aggregate_status: str | None,
    report_entry: dict[str, Any],
    log_dir: Path | None,
) -> str:
    if not patch_present:
        return "empty_patch"
    if report_entry:
        return "resolved" if bool(report_entry.get("resolved", False)) else "unresolved"
    if log_dir is not None:
        patch_status = log_dir / "patch_status.txt"
        if patch_status.is_file() and patch_status.read_text(encoding="utf-8").strip() == "failed":
            return "patch_failed"
        test_output = _text_excerpt([log_dir / "test_output.txt"])
        if "timeout" in test_output.lower() or "timed out" in test_output.lower():
            return "timeout"
    return aggregate_status or "missing_report"


def _sample_across_shards(
    rows: list[dict[str, Any]], sample_per_shard: int
) -> list[dict[str, Any]]:
    if sample_per_shard <= 0:
        return rows
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("shard") or "unknown"), []).append(row)

    sampled: list[dict[str, Any]] = []
    for shard in sorted(grouped):
        candidates = grouped[shard]
        selected: list[dict[str, Any]] = []
        # Exercise both the patch/evaluation path and empty-patch diagnosis when available.
        for patch_present in (True, False):
            match = next(
                (row for row in candidates if bool(row["patch_present"]) is patch_present),
                None,
            )
            if match is not None and match not in selected:
                selected.append(match)
            if len(selected) >= sample_per_shard:
                break
        for row in candidates:
            if len(selected) >= sample_per_shard:
                break
            if row not in selected:
                selected.append(row)
        sampled.extend(selected)
    return sampled


def analyze_run(
    run_root: str | Path,
    output_csv: str | Path,
    summary_output: str | Path,
    *,
    expected_count: int = 500,
    sample_per_shard: int = 0,
) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve()
    predictions, prediction_paths = _prediction_rows(root)
    prediction_shards = _prediction_shards(root)
    trajectory_index = _trajectory_index(root)
    instance_directories = _instance_directory_index(root)
    summary_results = _summary_results(root)
    aggregate = _aggregate_evaluation(root)
    instance_ids = {str(row["instance_id"]) for row in predictions}
    report_paths, eval_log_dirs = _evaluation_artifacts(root, instance_ids)

    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        instance_id = str(prediction["instance_id"])
        patch = prediction.get("model_patch", "")
        patch = patch if isinstance(patch, str) else ""
        patch_present = bool(patch.strip())
        patch_files, added, deleted = _patch_stats(patch)

        wrapper_path: Path | None = None
        wrapper: dict[str, Any] = {}
        shard = prediction_shards.get(instance_id, "")
        instance_dir: Path | None = None
        if instance_id in instance_directories:
            instance_dir, shard = instance_directories[instance_id]
        if instance_id in trajectory_index:
            wrapper_path, wrapper, shard = trajectory_index[instance_id]
            instance_dir = wrapper_path.parent
        _, raw = _raw_trajectory(instance_dir)
        raw_info = raw.get("info", {}) if isinstance(raw.get("info"), dict) else {}
        model_stats = raw_info.get("model_stats", {}) if isinstance(raw_info.get("model_stats"), dict) else {}
        messages = raw.get("messages", []) if isinstance(raw.get("messages"), list) else []
        api_calls = model_stats.get("api_calls", "")
        steps = _trajectory_step_count(raw, messages, wrapper)
        trajectory_metrics = _trajectory_metrics(raw, messages, wrapper)
        duration, duration_source, duration_is_estimate = _duration(
            wrapper_path, wrapper, instance_dir
        )
        action_steps = trajectory_metrics["agent_action_steps"]
        seconds_per_action = (
            round(float(duration) / int(action_steps), 6)
            if duration != "" and isinstance(action_steps, int) and action_steps > 0
            else ""
        )

        report_path = report_paths.get(instance_id)
        report_entry = _entry_for_instance(_read_json(report_path), instance_id)
        log_dir = eval_log_dirs.get(instance_id)
        eval_status = _evaluation_status(
            patch_present=patch_present,
            aggregate_status=aggregate.get(instance_id),
            report_entry=report_entry,
            log_dir=log_dir,
        )
        rollout_evidence = _text_excerpt(
            [
                instance_dir / "stderr.txt" if instance_dir else Path("/__missing__"),
                instance_dir / "minisweagent.log" if instance_dir else Path("/__missing__"),
                instance_dir / "stdout.txt" if instance_dir else Path("/__missing__"),
            ]
        )
        eval_evidence = _text_excerpt(
            [
                log_dir / "apply_patch.log" if log_dir else Path("/__missing__"),
                log_dir / "apptainer_stderr.txt" if log_dir else Path("/__missing__"),
                log_dir / "test_output.txt" if log_dir else Path("/__missing__"),
            ]
        )
        stage, category, reason, evidence, needs_review = _failure(
            evaluation_status=eval_status,
            patch_present=patch_present,
            wrapper=wrapper,
            summary_result=summary_results.get(instance_id, {}),
            raw=raw,
            rollout_evidence=rollout_evidence,
            evaluation_evidence=eval_evidence,
        )

        task = _read_json(instance_dir / "task.json" if instance_dir else None)
        repo = str(task.get("repo", "")) if isinstance(task, dict) else ""
        if not repo:
            repo = instance_id.split("__", 1)[0]
        row = {
            "instance_id": instance_id,
            "repo": repo,
            "shard": shard,
            "rollout_status": wrapper.get("status", summary_results.get(instance_id, {}).get("status", "missing")),
            "rollout_returncode": wrapper.get("returncode", ""),
            "mini_swe_exit_status": wrapper.get("mini_swe_exit_status", summary_results.get(instance_id, {}).get("mini_swe_exit_status", "")),
            "agent_exit_status": raw_info.get("exit_status", ""),
            "patch_present": patch_present,
            "patch_chars": len(patch),
            "patch_files": patch_files,
            "patch_added_lines": added,
            "patch_deleted_lines": deleted,
            "patch_source": wrapper.get("patch_source", ""),
            "discarded_error_patch_chars": len(wrapper.get("error_patch", ""))
            if isinstance(wrapper.get("error_patch"), str)
            else "",
            "trajectory_steps": steps,
            "trajectory_messages": len(messages) if messages else "",
            **trajectory_metrics,
            "model_api_calls": api_calls,
            "duration_seconds": duration,
            "seconds_per_action": seconds_per_action,
            "duration_source": duration_source,
            "duration_is_estimate": duration_is_estimate,
            "evaluation_status": eval_status,
            "resolved": eval_status == "resolved",
            "evaluation_returncode": _evaluation_returncode(log_dir),
            "fail_to_pass_success": _count_test_status(report_entry, "FAIL_TO_PASS", "success"),
            "fail_to_pass_failure": _count_test_status(report_entry, "FAIL_TO_PASS", "failure"),
            "pass_to_pass_success": _count_test_status(report_entry, "PASS_TO_PASS", "success"),
            "pass_to_pass_failure": _count_test_status(report_entry, "PASS_TO_PASS", "failure"),
            "failure_stage": stage,
            "failure_category": category,
            "failure_reason": reason,
            "failure_evidence": evidence,
            "needs_llm_review": needs_review,
            "trajectory_path": str(wrapper_path or ""),
            "evaluation_report_path": str(report_path or ""),
        }
        rows.append(row)

    source_rows = rows
    rows = _sample_across_shards(source_rows, sample_per_shard)

    csv_path = Path(output_csv).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    durations = [float(row["duration_seconds"]) for row in rows if row["duration_seconds"] != ""]
    evaluation_counts = Counter(str(row["evaluation_status"]) for row in rows)
    rollout_counts = Counter(str(row["rollout_status"]) for row in rows)
    failure_counts = Counter(str(row["failure_category"]) for row in rows if row["failure_category"])
    summary = {
        "run_root": str(root),
        "csv_path": str(csv_path.resolve()),
        "prediction_paths": [str(path) for path in prediction_paths],
        "expected_instances": expected_count,
        "source_instances": len(source_rows),
        "instances": len(rows),
        "sample_per_shard": sample_per_shard,
        "sampled_shards": dict(sorted(Counter(str(row["shard"]) for row in rows).items())),
        "matches_expected_count": expected_count <= 0 or len(source_rows) == expected_count,
        "rollouts_with_patch": sum(bool(row["patch_present"]) for row in rows),
        "rollouts_empty_patch": sum(not bool(row["patch_present"]) for row in rows),
        "rollout_process_errors": rollout_counts["error"],
        "rollout_status_counts": dict(sorted(rollout_counts.items())),
        "evaluations_resolved": evaluation_counts["resolved"],
        "evaluations_failed": len(rows) - evaluation_counts["resolved"],
        "evaluation_status_counts": dict(sorted(evaluation_counts.items())),
        "failure_category_counts": dict(sorted(failure_counts.items())),
        "needs_llm_review": sum(bool(row["needs_llm_review"]) for row in rows),
        "duration_available": len(durations),
        "duration_seconds": {
            "min": min(durations) if durations else None,
            "median": statistics.median(durations) if durations else None,
            "mean": statistics.fmean(durations) if durations else None,
            "max": max(durations) if durations else None,
        },
    }
    summary_path = Path(summary_output).expanduser()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a per-instance CSV and aggregate summary for a debug-depo run."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-csv")
    parser.add_argument("--summary-output")
    parser.add_argument(
        "--expected-count",
        type=int,
        default=500,
        help="Return exit code 2 if the prediction count differs; use 0 to disable.",
    )
    parser.add_argument(
        "--sample-per-shard",
        type=int,
        default=0,
        help="Write at most this many representative rows from every shard; 0 writes all rows.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = Path(args.run_root).expanduser()
    output_dir = run_root / "analysis"
    summary = analyze_run(
        run_root,
        args.output_csv or output_dir / "instances.csv",
        args.summary_output or output_dir / "summary.json",
        expected_count=args.expected_count,
        sample_per_shard=args.sample_per_shard,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["matches_expected_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
