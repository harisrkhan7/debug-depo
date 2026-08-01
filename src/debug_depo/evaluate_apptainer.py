"""Run SWE-bench evaluation with Apptainer SIF images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_INSTANCE,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    TESTS_TIMEOUT,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import get_predictions_from_file

from debug_depo.apptainer_cache import (
    DEFAULT_SWEBENCH_IMAGE_TEMPLATE,
    image_uri_from_template,
    pull_sif_if_missing,
    sif_path_for_image,
)
from debug_depo.constants import DEFAULT_SWEBENCH_DATASET, DEFAULT_SWEBENCH_SPLIT
from debug_depo.data import (
    load_swebench_tasks,
    read_instance_ids_file,
    resolve_swebench_dataset_revision,
)
from debug_depo.evaluate import model_report_name, path_from_cwd, summarize_report
from debug_depo.utils import (
    ensure_dir,
    load_hf_token_from_file,
    package_provenance,
    write_json,
)


DEFAULT_IMAGE_TEMPLATE = DEFAULT_SWEBENCH_IMAGE_TEMPLATE
EVAL_BIND_DIR = "/swebench_eval"
CACHE_KEY_SCHEMA_VERSION = 2
SCORED_STATUSES = frozenset(
    {
        "completed",
        "cached_report",
        "empty_patch",
        "patch_failed",
        "timeout",
    }
)
def collect_instance_ids(args: argparse.Namespace) -> list[str]:
    ids = list(args.instance_ids or [])
    if args.instance_ids_file:
        ids.extend(read_instance_ids_file(args.instance_ids_file))
    return ids


def write_runner_script(log_dir: Path) -> Path:
    runner = log_dir / "run_apptainer_eval.sh"
    runner.write_text(
        f"""#!/usr/bin/env bash
set -uo pipefail
cd /testbed || exit 20
git config --global --add safe.directory /testbed || true
patch_status="{EVAL_BIND_DIR}/patch_status.txt"
apply_log="{EVAL_BIND_DIR}/apply_patch.log"
: > "$apply_log"
applied=0
for git_apply_cmd in \\
  "git apply --verbose" \\
  "git apply --verbose --reject" \\
  "patch --batch --fuzz=5 -p1 -i"
do
  if $git_apply_cmd "{EVAL_BIND_DIR}/patch.diff" >> "$apply_log" 2>&1; then
    echo "{APPLY_PATCH_PASS}" >> "$apply_log"
    applied=1
    break
  fi
done
if [[ "$applied" != "1" ]]; then
  echo "{APPLY_PATCH_FAIL}" >> "$apply_log"
  echo failed > "$patch_status"
  cat "$apply_log" > "{EVAL_BIND_DIR}/{LOG_TEST_OUTPUT}"
  exit 11
fi
echo applied > "$patch_status"
git -c core.fileMode=false diff > "{EVAL_BIND_DIR}/git_diff_before.txt" 2>/dev/null || true
bash "{EVAL_BIND_DIR}/eval.sh" > "{EVAL_BIND_DIR}/{LOG_TEST_OUTPUT}" 2>&1
eval_status=$?
git -c core.fileMode=false diff > "{EVAL_BIND_DIR}/git_diff_after.txt" 2>/dev/null || true
exit "$eval_status"
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    return runner


def build_apptainer_command(
    sif_path: str | Path,
    log_dir: str | Path,
    *,
    extra_args: list[str] | None = None,
) -> list[str]:
    command = [
        "apptainer",
        "exec",
        "--writable-tmpfs",
        "--bind",
        f"{Path(log_dir).resolve()}:{EVAL_BIND_DIR}",
    ]
    command.extend(extra_args or [])
    command.extend([str(sif_path), "/bin/bash", f"{EVAL_BIND_DIR}/run_apptainer_eval.sh"])
    return command


def model_log_name(prediction: dict[str, Any]) -> str:
    return str(prediction.get(KEY_MODEL, "None")).replace("/", "__")


@lru_cache(maxsize=1)
def _swebench_provenance() -> dict[str, str | None]:
    provenance = package_provenance("swebench", "swebench")
    return {
        "swebench_version": provenance["version"],
        "swebench_revision": provenance["revision"],
        "swebench_working_tree_diff_sha256": provenance[
            "working_tree_diff_sha256"
        ],
    }


def _evaluation_cache_key(
    instance: dict[str, Any],
    prediction: dict[str, Any],
    args: argparse.Namespace,
    test_spec: Any,
) -> dict[str, Any]:
    patch = prediction.get(KEY_PREDICTION)
    patch_text = patch if isinstance(patch, str) else ""
    instance_payload = json.dumps(
        instance,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return {
        "schema_version": CACHE_KEY_SCHEMA_VERSION,
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "model_name_or_path": prediction.get(KEY_MODEL, ""),
        "patch_sha256": hashlib.sha256(patch_text.encode("utf-8")).hexdigest(),
        "instance_sha256": hashlib.sha256(instance_payload.encode("utf-8")).hexdigest(),
        "eval_script_sha256": hashlib.sha256(
            str(test_spec.eval_script).encode("utf-8")
        ).hexdigest(),
        "instance_image_key": str(test_spec.instance_image_key),
        "image_template": args.image_template,
        **_swebench_provenance(),
    }


def _load_cached_report(
    report_path: Path,
    cache_key_path: Path,
    expected_key: dict[str, Any],
) -> dict[str, Any] | None:
    if not report_path.is_file() or not cache_key_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        cache_key = json.loads(cache_key_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict) or cache_key != expected_key:
        return None
    return report


def run_instance(
    instance: dict[str, Any],
    prediction: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    test_spec = make_test_spec(instance)
    instance_id = str(instance[KEY_INSTANCE_ID])
    log_dir = ensure_dir(
        Path(args.log_dir) / args.run_id / model_log_name(prediction) / instance_id
    )
    report_path = log_dir / LOG_REPORT
    cache_key_path = log_dir / "cache_key.json"
    cache_key = _evaluation_cache_key(instance, prediction, args, test_spec)
    report = (
        None
        if args.overwrite
        else _load_cached_report(report_path, cache_key_path, cache_key)
    )
    if report is not None:
        return {
            "instance_id": instance_id,
            "status": "cached_report",
            "resolved": bool(report.get(instance_id, {}).get("resolved", False)),
            "report_path": str(report_path),
        }

    patch = prediction.get(KEY_PREDICTION)
    if patch in ("", None):
        if not args.dry_run:
            report_path.unlink(missing_ok=True)
            cache_key_path.unlink(missing_ok=True)
        return {"instance_id": instance_id, "status": "empty_patch"}

    if not args.dry_run:
        report_path.unlink(missing_ok=True)
        cache_key_path.unlink(missing_ok=True)

    image_uri = image_uri_from_template(
        args.image_template, instance_id, image_key=test_spec.instance_image_key
    )
    sif_path = sif_path_for_image(args.sif_dir, image_uri)
    pull_command = pull_sif_if_missing(
        sif_path=sif_path,
        image_uri=image_uri,
        cache_dir=args.apptainer_cache_dir,
        dry_run=args.dry_run,
    )

    (log_dir / "patch.diff").write_text(str(patch), encoding="utf-8")
    (log_dir / "eval.sh").write_text(test_spec.eval_script, encoding="utf-8")
    write_runner_script(log_dir)
    command = build_apptainer_command(
        sif_path,
        log_dir,
        extra_args=args.apptainer_arg,
    )
    metadata = {
        "instance_id": instance_id,
        "image_uri": image_uri,
        "sif_path": str(sif_path),
        "pull_command": pull_command,
        "command": command,
    }
    write_json(log_dir / "apptainer_metadata.json", metadata)
    if args.dry_run:
        return {"instance_id": instance_id, "status": "dry_run", **metadata}

    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        (log_dir / LOG_TEST_OUTPUT).write_text(
            f"{TESTS_TIMEOUT}: {args.timeout} seconds exceeded.\n",
            encoding="utf-8",
        )
        (log_dir / "apptainer_stdout.txt").write_text(exc.stdout or "", encoding="utf-8")
        (log_dir / "apptainer_stderr.txt").write_text(exc.stderr or "", encoding="utf-8")
        return {"instance_id": instance_id, "status": "timeout", "report_path": str(report_path)}

    (log_dir / "apptainer_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (log_dir / "apptainer_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    (log_dir / LOG_INSTANCE).write_text(
        json.dumps({**metadata, "returncode": completed.returncode}, indent=2) + "\n",
        encoding="utf-8",
    )

    patch_status_path = log_dir / "patch_status.txt"
    if not patch_status_path.exists() or patch_status_path.read_text(encoding="utf-8").strip() != "applied":
        return {
            "instance_id": instance_id,
            "status": "patch_failed",
            "returncode": completed.returncode,
        }

    report = get_eval_report(
        test_spec=test_spec,
        prediction=prediction,
        test_log_path=str(log_dir / LOG_TEST_OUTPUT),
        include_tests_status=True,
    )
    write_json(report_path, report)
    write_json(cache_key_path, cache_key)
    resolved = bool(report.get(instance_id, {}).get("resolved", False))
    return {
        "instance_id": instance_id,
        "status": "completed",
        "resolved": resolved,
        "returncode": completed.returncode,
        "report_path": str(report_path),
    }


def aggregate_report(
    dataset: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    *,
    log_dir: str | Path,
    run_id: str,
) -> dict[str, Any]:
    completed_ids: set[str] = set()
    resolved_ids: set[str] = set()
    unresolved_ids: set[str] = set()
    incomplete_ids: set[str] = set()
    empty_patch_ids: set[str] = set()
    error_ids: set[str] = set()
    dataset_ids = {str(instance[KEY_INSTANCE_ID]) for instance in dataset}

    for instance in dataset:
        instance_id = str(instance[KEY_INSTANCE_ID])
        prediction = predictions.get(instance_id)
        if prediction is None:
            incomplete_ids.add(instance_id)
            continue
        if prediction.get(KEY_PREDICTION) in ("", None):
            empty_patch_ids.add(instance_id)
            continue
        report_file = (
            Path(log_dir)
            / run_id
            / model_log_name(prediction)
            / instance_id
            / LOG_REPORT
        )
        if not report_file.exists():
            error_ids.add(instance_id)
            continue
        completed_ids.add(instance_id)
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
            if report[instance_id]["resolved"]:
                resolved_ids.add(instance_id)
            else:
                unresolved_ids.add(instance_id)
        except (json.JSONDecodeError, KeyError):
            error_ids.add(instance_id)

    submitted_ids = sorted(set(predictions) & dataset_ids)
    return {
        "total_instances": len(dataset),
        "submitted_instances": len(submitted_ids),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "completed_ids": sorted(completed_ids),
        "incomplete_ids": sorted(incomplete_ids),
        "empty_patch_ids": sorted(empty_patch_ids),
        "submitted_ids": submitted_ids,
        "resolved_ids": sorted(resolved_ids),
        "unresolved_ids": sorted(unresolved_ids),
        "error_ids": sorted(error_ids),
        "schema_version": 2,
        "runtime": "apptainer",
    }


def run_apptainer_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    load_hf_token_from_file()
    args.report_dir = str(path_from_cwd(args.report_dir))
    args.log_dir = str(path_from_cwd(args.log_dir))
    args.sif_dir = str(path_from_cwd(args.sif_dir))
    if args.apptainer_cache_dir:
        args.apptainer_cache_dir = str(path_from_cwd(args.apptainer_cache_dir))
    args.instance_ids = collect_instance_ids(args)
    args.dataset_revision = resolve_swebench_dataset_revision(
        args.dataset,
        args.dataset_revision,
    )

    dataset = load_swebench_tasks(
        args.dataset,
        args.split,
        revision=args.dataset_revision,
    )
    if args.instance_ids:
        requested_ids = set(args.instance_ids)
        dataset_ids = {str(instance[KEY_INSTANCE_ID]) for instance in dataset}
        missing_ids = sorted(requested_ids - dataset_ids)
        if missing_ids:
            raise ValueError(
                "Instance ids not found: " + ", ".join(missing_ids)
            )
        dataset = [
            instance
            for instance in dataset
            if str(instance[KEY_INSTANCE_ID]) in requested_ids
        ]
    if args.predictions_path == "gold":
        predictions_list = [
            {
                KEY_INSTANCE_ID: instance[KEY_INSTANCE_ID],
                KEY_PREDICTION: instance["patch"],
                KEY_MODEL: "gold",
            }
            for instance in dataset
        ]
    else:
        predictions_list = get_predictions_from_file(
            args.predictions_path,
            args.dataset,
            args.split,
        )
    predictions = {str(pred[KEY_INSTANCE_ID]): pred for pred in predictions_list}
    if args.limit is not None:
        dataset = dataset[: args.limit]
    missing_predictions = [
        str(instance[KEY_INSTANCE_ID])
        for instance in dataset
        if str(instance[KEY_INSTANCE_ID]) not in predictions
    ]
    if missing_predictions:
        print(f"Warning: missing predictions for {len(missing_predictions)} selected instances.")

    missing_results = [
        {
            "instance_id": str(instance[KEY_INSTANCE_ID]),
            "status": "missing_prediction",
        }
        for instance in dataset
        if str(instance[KEY_INSTANCE_ID]) not in predictions
    ]
    runnable = [
        instance
        for instance in dataset
        if str(instance[KEY_INSTANCE_ID]) in predictions
    ]
    results: list[dict[str, Any]] = missing_results

    def run_safely(instance: dict[str, Any]) -> dict[str, Any]:
        instance_id = str(instance[KEY_INSTANCE_ID])
        try:
            return run_instance(instance, predictions[instance_id], args)
        except Exception as exc:
            return {
                "instance_id": instance_id,
                "status": "error",
                "error": repr(exc),
            }

    if args.max_workers <= 1:
        for instance in runnable:
            results.append(run_safely(instance))
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(run_safely, instance): str(instance[KEY_INSTANCE_ID])
                for instance in runnable
            }
            for future in as_completed(futures):
                results.append(future.result())

    report = aggregate_report(
        dataset,
        predictions,
        log_dir=args.log_dir,
        run_id=args.run_id,
    )
    report_path = Path(args.report_dir) / model_report_name(args.model, args.run_id)
    write_json(report_path, report)
    status_ids: dict[str, list[str]] = {}
    for result in results:
        status_ids.setdefault(str(result["status"]), []).append(str(result["instance_id"]))
    summary = {
        "dataset_revision": args.dataset_revision,
        **_swebench_provenance(),
        "report_path": str(report_path),
        "results": results,
        "scored_instances": sum(
            str(result.get("status")) in SCORED_STATUSES for result in results
        ),
        "status_ids": {
            status: sorted(instance_ids)
            for status, instance_ids in sorted(status_ids.items())
        },
        **summarize_report(
            report,
            dataset=args.dataset,
            dataset_revision=args.dataset_revision,
            split=args.split,
            model=args.model,
        ),
    }
    if args.summary_output:
        write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2))
    if args.require_complete:
        failed_statuses = {
            status: len(instance_ids)
            for status, instance_ids in summary["status_ids"].items()
            if status not in SCORED_STATUSES
        }
        if failed_statuses:
            raise RuntimeError(
                "SWE-bench evaluation produced unscored infrastructure outcomes: "
                f"{failed_statuses}"
            )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SWE-bench evaluation with Apptainer.")
    parser.add_argument("--dataset", default=DEFAULT_SWEBENCH_DATASET)
    parser.add_argument(
        "--dataset-revision",
        default=os.getenv("SWEBENCH_DATASET_REVISION"),
        help=(
            "Hugging Face dataset revision. The proven SWE-bench Verified revision "
            "is selected automatically for the default dataset."
        ),
    )
    parser.add_argument("--split", default=DEFAULT_SWEBENCH_SPLIT)
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", default="agentforge_8b_sft_verified")
    parser.add_argument("--report-dir", default="results/swebench")
    parser.add_argument("--summary-output")
    parser.add_argument("--instance-ids-file")
    parser.add_argument("--instance-id", action="append", dest="instance_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--log-dir", default="logs/run_evaluation")
    parser.add_argument(
        "--sif-dir",
        default=os.getenv("SWEBENCH_APPTAINER_SIF_DIR", "data/apptainer/sifs"),
    )
    parser.add_argument(
        "--apptainer-cache-dir",
        default=os.getenv("APPTAINER_CACHEDIR", os.getenv("SWEBENCH_APPTAINER_CACHE_DIR")),
    )
    parser.add_argument(
        "--image-template",
        default=os.getenv("SWEBENCH_APPTAINER_IMAGE_TEMPLATE", DEFAULT_IMAGE_TEMPLATE),
        help="Format string with {instance_id}, {instance_id_lower}, and {image_key}.",
    )
    parser.add_argument(
        "--apptainer-arg",
        action="append",
        default=[],
        help="Extra argument passed to `apptainer exec` before the SIF path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_apptainer_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
