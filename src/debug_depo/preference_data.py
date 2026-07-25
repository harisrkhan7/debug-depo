"""Shared readers and serializers for SWE-smith preference datasets."""

from __future__ import annotations

import json
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


def load_evaluated_trajectories(run_root: str | Path) -> list[TrajectoryRecord]:
    """Load complete, scored SWE-smith rollouts directly from run artifacts."""

    root = Path(run_root)
    records: list[TrajectoryRecord] = []
    for sample_index in discover_sample_indices(root):
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
    if not records:
        raise ValueError(
            f"No evaluated trajectories with complete per-call token usage found under {root}"
        )
    return records


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows as JSONL and return their count."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
