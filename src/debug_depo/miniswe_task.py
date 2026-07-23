"""Run an official mini-swe-agent batch runner on one local task JSON."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


RUNNER_MODULES = {
    "pool_way": "minisweagent.run.extra.swebench_pool_way",
    "swebench": "minisweagent.run.extra.swebench",
}


def load_task_json(
    path: str | Path,
    *,
    expected_instance_id: str,
) -> dict[str, Any]:
    """Load and validate the single task passed to mini-swe-agent."""

    task_path = Path(path)
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one task object in {task_path}")

    instance_id = payload.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError(f"Task in {task_path} has no valid instance_id")
    if instance_id != expected_instance_id:
        raise ValueError(
            f"Task instance_id {instance_id!r} does not match "
            f"the requested instance {expected_instance_id!r}"
        )
    if not isinstance(payload.get("problem_statement"), str):
        raise ValueError(f"Task {instance_id!r} in {task_path} has no problem_statement")
    return payload


def _runner_module(runner: str) -> ModuleType:
    try:
        module_name = RUNNER_MODULES[runner]
    except KeyError as exc:
        choices = ", ".join(sorted(RUNNER_MODULES))
        raise ValueError(f"Unknown mini-swe runner {runner!r}; expected one of: {choices}") from exc
    return importlib.import_module(module_name)


def run_task(
    *,
    task_json: str | Path,
    instance_id: str,
    runner: str,
    runner_args: Sequence[str],
) -> None:
    """Inject one local task into the pinned official mini-swe runner."""

    task = load_task_json(task_json, expected_instance_id=instance_id)
    module = _runner_module(runner)
    original_load_dataset = getattr(module, "load_dataset", None)
    if original_load_dataset is None or not callable(original_load_dataset):
        raise RuntimeError(
            f"Pinned mini-swe runner {module.__name__} no longer exposes load_dataset"
        )

    def load_single_task(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [dict(task)]

    module.load_dataset = load_single_task
    try:
        module.app(
            args=list(runner_args),
            prog_name=module.__name__,
            standalone_mode=False,
        )
    finally:
        module.load_dataset = original_load_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run mini-swe-agent on one preselected local task.",
    )
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--runner", choices=sorted(RUNNER_MODULES), required=True)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner_args = list(args.runner_args)
    if runner_args[:1] == ["--"]:
        runner_args = runner_args[1:]
    run_task(
        task_json=args.task_json,
        instance_id=args.instance_id,
        runner=args.runner,
        runner_args=runner_args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
