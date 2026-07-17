"""Run and summarize the official SWE-bench evaluation harness."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from debug_depo.constants import (
    DEFAULT_SWEBENCH_DATASET,
    DEFAULT_SWEBENCH_SPLIT,
    TARGET_VERIFIED_RESOLVED,
    TARGET_VERIFIED_SCORE,
    TARGET_VERIFIED_TOTAL,
)
from debug_depo.data import read_instance_ids_file
from debug_depo.utils import ensure_dir, load_hf_token_from_file, read_json, write_json


def model_report_name(model: str, run_id: str) -> str:
    return f"{model.replace('/', '__')}.{run_id}.json"


def path_from_cwd(path: str | Path) -> Path:
    output_path = Path(path).expanduser()
    if output_path.is_absolute():
        return output_path
    return (Path.cwd() / output_path).resolve()


def move_harness_report(
    report_dir: str | Path, eval_cwd: str | Path, model: str, run_id: str
) -> Path:
    """Move SWE-bench's aggregate report into our requested report directory."""
    report_path = Path(report_dir) / model_report_name(model, run_id)
    harness_path = Path(eval_cwd) / model_report_name(model, run_id)
    if harness_path.exists() and harness_path.resolve() != report_path.resolve():
        ensure_dir(report_path.parent)
        harness_path.replace(report_path)
    return report_path


def should_build_images_locally(namespace: str | None, auto_namespace: bool) -> bool:
    if namespace is not None:
        return namespace == ""
    return auto_namespace and platform.machine().lower() in {"arm64", "aarch64"}


def build_evaluation_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        args.dataset,
        "--split",
        args.split,
        "--predictions_path",
        args.predictions_path,
        "--max_workers",
        str(args.max_workers),
        "--run_id",
        args.run_id,
        "--timeout",
        str(args.timeout),
        "--cache_level",
        args.cache_level,
        "--clean",
        str(args.clean),
        "--report_dir",
        args.report_dir,
    ]
    if args.force_rebuild:
        command.extend(["--force_rebuild", "True"])
    if should_build_images_locally(args.namespace, args.auto_namespace):
        command.extend(["--namespace", ""])
    elif args.namespace:
        command.extend(["--namespace", args.namespace])
    if args.instance_ids:
        command.append("--instance_ids")
        command.extend(args.instance_ids)
    if args.modal:
        command.append("--modal")
    return command


def load_report(report_dir: str | Path, model: str, run_id: str) -> dict[str, Any] | None:
    path = Path(report_dir) / model_report_name(model, run_id)
    if not path.exists():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def summarize_report(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {
            "status": "missing_report",
            "resolved_instances": 0,
            "submitted_instances": 0,
            "resolution_rate": 0.0,
        }
    submitted = int(report.get("submitted_instances", 0))
    resolved = int(report.get("resolved_instances", 0))
    total = int(report.get("total_instances", submitted))
    denominator = submitted or total or 1
    return {
        "status": "ok",
        "total_instances": total,
        "submitted_instances": submitted,
        "completed_instances": int(report.get("completed_instances", 0)),
        "resolved_instances": resolved,
        "unresolved_instances": int(report.get("unresolved_instances", 0)),
        "empty_patch_instances": int(report.get("empty_patch_instances", 0)),
        "error_instances": int(report.get("error_instances", 0)),
        "resolution_rate": resolved / denominator,
        "target_score": TARGET_VERIFIED_SCORE,
        "target_resolved": TARGET_VERIFIED_RESOLVED,
        "target_total": TARGET_VERIFIED_TOTAL,
        "resolved_delta_vs_target": resolved - TARGET_VERIFIED_RESOLVED
        if total == TARGET_VERIFIED_TOTAL
        else None,
    }


def collect_instance_ids(args: argparse.Namespace) -> list[str]:
    ids = list(args.instance_ids or [])
    if args.instance_ids_file:
        ids.extend(read_instance_ids_file(args.instance_ids_file))
    return ids


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    load_hf_token_from_file()
    args.instance_ids = collect_instance_ids(args)
    args.report_dir = str(path_from_cwd(args.report_dir))
    args.eval_cwd = str(path_from_cwd(args.eval_cwd))
    command = build_evaluation_command(args)
    if args.dry_run:
        summary = {"dry_run": True, "command": command}
        print(json.dumps(summary, indent=2))
        return summary

    completed = subprocess.run(command, cwd=args.eval_cwd, check=False)
    report_path = move_harness_report(args.report_dir, args.eval_cwd, args.model, args.run_id)
    report = load_report(args.report_dir, args.model, args.run_id)
    summary = {
        "command": command,
        "returncode": completed.returncode,
        "report_path": str(report_path),
        **summarize_report(report),
    }
    if args.summary_output:
        write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the official SWE-bench evaluator.")
    parser.add_argument("--dataset", default=DEFAULT_SWEBENCH_DATASET)
    parser.add_argument("--split", default=DEFAULT_SWEBENCH_SPLIT)
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", default="agentforge_sft_verified")
    parser.add_argument("--report-dir", default="results/swebench")
    parser.add_argument("--summary-output")
    parser.add_argument("--instance-ids-file")
    parser.add_argument("--instance-id", action="append", dest="instance_ids")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cache-level", choices=("none", "base", "env", "instance"), default="env")
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--namespace",
        default=None,
        help="Docker image namespace. Use an empty string to force local image builds.",
    )
    parser.add_argument(
        "--auto-namespace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On ARM Macs, pass --namespace '' so SWE-bench builds images locally.",
    )
    parser.add_argument("--modal", action="store_true")
    parser.add_argument("--eval-cwd", default=".")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
