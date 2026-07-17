"""Prediction-file utilities for SWE-bench reproduction runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from debug_depo.utils import read_jsonl, write_json, write_jsonl


def load_predictions(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def merge_predictions(
    paths: list[str | Path],
    *,
    keep: str = "last",
) -> list[dict[str, Any]]:
    if keep not in {"first", "last"}:
        raise ValueError("keep must be 'first' or 'last'")
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in load_predictions(paths):
        instance_id = str(row.get("instance_id", ""))
        if not instance_id:
            raise ValueError("Prediction row missing instance_id")
        if instance_id not in merged:
            order.append(instance_id)
            merged[instance_id] = row
            continue
        if keep == "last":
            merged[instance_id] = row
    return [merged[instance_id] for instance_id in order]


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_predictions = len(rows)
    n_with_patch = sum(bool(row.get("model_patch")) for row in rows)
    ids = [str(row.get("instance_id", "")) for row in rows]
    return {
        "n_predictions": n_predictions,
        "n_unique_instances": len(set(ids)),
        "n_with_patch": n_with_patch,
        "n_empty_patch": n_predictions - n_with_patch,
        "duplicate_instances": sorted(
            instance_id for instance_id in set(ids) if ids.count(instance_id) > 1
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge and summarize SWE-bench prediction JSONL.")
    parser.add_argument("--input", nargs="+", required=True, dest="inputs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--keep", choices=("first", "last"), default="last")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = merge_predictions(args.inputs, keep=args.keep)
    write_jsonl(args.output, rows)
    summary = summarize_predictions(rows)
    summary["output"] = str(Path(args.output))
    if args.summary_output:
        write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
