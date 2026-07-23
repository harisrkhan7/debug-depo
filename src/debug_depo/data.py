"""SWE-bench task loading and deterministic task selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from debug_depo.constants import (
    DEFAULT_SWEBENCH_DATASET,
    DEFAULT_SWEBENCH_DATASET_REVISION,
    DEFAULT_SWEBENCH_SPLIT,
)
from debug_depo.utils import ensure_dir, load_hf_token_from_file, read_jsonl, write_json, write_jsonl


def _records_from_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("instances", "tasks", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of SWE-bench instances in {path}")
    return [dict(item) for item in payload if isinstance(item, dict)]


def load_swebench_tasks(
    dataset_name: str,
    split: str,
    *,
    revision: str | None = None,
) -> list[dict[str, Any]]:
    """Load SWE-bench instances from Hugging Face or a local JSON/JSONL file."""

    dataset_path = Path(dataset_name)
    if dataset_path.suffix == ".jsonl" and dataset_path.exists():
        return read_jsonl(dataset_path)
    if dataset_path.suffix == ".json" and dataset_path.exists():
        return _records_from_json(dataset_path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required to load SWE-bench from Hugging Face."
        ) from exc

    load_hf_token_from_file()
    return [
        dict(row)
        for row in load_dataset(
            dataset_name,
            split=split,
            revision=revision,
        )
    ]


def resolve_swebench_dataset_revision(
    dataset_name: str,
    revision: str | None,
) -> str | None:
    """Resolve the proven Verified pin without applying it to other datasets."""

    if Path(dataset_name).is_file():
        return None
    if revision:
        return revision
    if dataset_name.casefold() == DEFAULT_SWEBENCH_DATASET.casefold():
        return DEFAULT_SWEBENCH_DATASET_REVISION
    return None


def read_instance_ids_file(path: str | Path) -> list[str]:
    task_path = Path(path)
    if task_path.suffix == ".jsonl":
        return [
            str(row["instance_id"])
            for row in read_jsonl(task_path)
            if row.get("instance_id")
        ]
    if task_path.suffix == ".json":
        payload = json.loads(task_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("instance_ids", payload.get("instances", payload.get("tasks", [])))
        if not isinstance(payload, list):
            raise ValueError(f"Expected instance id list in {task_path}")
        ids: list[str] = []
        for item in payload:
            ids.append(str(item.get("instance_id") if isinstance(item, dict) else item))
        return [item for item in ids if item]
    return [
        line.strip()
        for line in task_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def select_tasks(
    tasks: list[dict[str, Any]],
    *,
    instance_ids: list[str] | None = None,
    start_index: int = 0,
    limit: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[dict[str, Any]]:
    """Select a stable subset, preserving explicit instance-id order when supplied."""

    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    by_id = {str(task.get("instance_id")): task for task in tasks if task.get("instance_id")}
    if instance_ids:
        missing = [instance_id for instance_id in instance_ids if instance_id not in by_id]
        if missing:
            raise ValueError("Instance ids not found: " + ", ".join(missing))
        selected = [by_id[instance_id] for instance_id in instance_ids]
    else:
        selected = list(tasks)

    selected = selected[start_index:]
    if limit is not None:
        selected = selected[:limit]
    if num_shards > 1:
        selected = [
            task
            for index, task in enumerate(selected)
            if index % num_shards == shard_index
        ]
    return selected


def instance_ids(tasks: list[dict[str, Any]]) -> list[str]:
    ids = [task.get("instance_id") for task in tasks]
    if any(not item for item in ids):
        raise ValueError("Every selected task must contain `instance_id`")
    return [str(item) for item in ids]


def write_task_selection(
    tasks: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    name: str = "swebench_verified",
) -> dict[str, Any]:
    output_root = ensure_dir(output_dir)
    task_jsonl = output_root / f"{name}.jsonl"
    ids_txt = output_root / f"{name}_instance_ids.txt"
    write_jsonl(task_jsonl, tasks)
    ids = instance_ids(tasks)
    ids_txt.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    summary = {
        "name": name,
        "n_tasks": len(tasks),
        "task_jsonl": str(task_jsonl),
        "instance_ids_file": str(ids_txt),
        "instance_ids": ids,
    }
    write_json(output_root / f"{name}_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a SWE-bench task selection file.")
    parser.add_argument("--dataset", default=DEFAULT_SWEBENCH_DATASET)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--split", default=DEFAULT_SWEBENCH_SPLIT)
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--name", default="swebench_verified")
    parser.add_argument("--instance-ids-file")
    parser.add_argument("--instance-id", action="append", dest="instance_ids")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_revision = resolve_swebench_dataset_revision(
        args.dataset,
        args.dataset_revision,
    )
    requested_ids = list(args.instance_ids or [])
    if args.instance_ids_file:
        requested_ids.extend(read_instance_ids_file(args.instance_ids_file))
    tasks = select_tasks(
        load_swebench_tasks(
            args.dataset,
            args.split,
            revision=dataset_revision,
        ),
        instance_ids=requested_ids or None,
        start_index=args.start_index,
        limit=args.limit,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    summary = write_task_selection(tasks, args.output_dir, name=args.name)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
