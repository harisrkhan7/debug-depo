"""Evaluate one sample of SWE-smith predictions with Apptainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from debug_depo.apptainer_cache import (
    image_uri,
    pull_sif_if_missing,
    sif_path_for_image,
)
from debug_depo.constants import (
    DEFAULT_SWESMITH_DATASET,
    DEFAULT_SWESMITH_DATASET_REVISION,
    DEFAULT_SWESMITH_SPLIT,
)
from debug_depo.data import load_swebench_tasks, read_instance_ids_file
from debug_depo.utils import (
    ensure_dir,
    load_hf_token_from_file,
    package_provenance,
    read_jsonl,
    write_json,
)


EVAL_BIND_DIR = "/swesmith_eval"
TESTBED_DIR = "/testbed"
CACHE_KEY_SCHEMA_VERSION = 3
PATCH_APPLY_MODES = frozenset({"forward", "reverse"})
SCORED_STATUSES = frozenset(
    {
        "completed",
        "cached_report",
        "empty_patch",
        "patch_failed",
        "timeout",
    }
)
def _load_predictions(path: str | Path) -> list[dict[str, Any]]:
    prediction_path = Path(path)
    if prediction_path.suffix == ".jsonl":
        rows = read_jsonl(prediction_path)
    elif prediction_path.suffix == ".json":
        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        rows = list(payload.values()) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"Expected a prediction list or mapping in {prediction_path}")
        rows = [dict(row) for row in rows if isinstance(row, dict)]
    else:
        raise ValueError("SWE-smith predictions must be JSON or JSONL")

    seen: set[str] = set()
    for row in rows:
        instance_id = str(row.get("instance_id", ""))
        if not instance_id:
            raise ValueError("Prediction row missing instance_id")
        if instance_id in seen:
            raise ValueError(f"Duplicate prediction for instance_id: {instance_id}")
        seen.add(instance_id)
    return rows


def _profile_for(instance: dict[str, Any]) -> Any:
    try:
        from swesmith.profiles import registry
    except ImportError as exc:
        raise RuntimeError(
            "SWE-smith is not installed. Run scripts/install_swesmith.sh first."
        ) from exc
    return registry.get_from_inst(instance)


def _image_uri(image_name: str) -> str:
    return image_uri(image_name)


def _sif_path(sif_dir: str | Path, image_name: str) -> Path:
    return sif_path_for_image(sif_dir, image_name)


def _cache_key(
    prediction: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    patch = prediction.get("model_patch")
    patch_text = patch if isinstance(patch, str) else ""
    return {
        "schema_version": CACHE_KEY_SCHEMA_VERSION,
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "runtime": args.runtime,
        "f2p_only": bool(args.f2p_only),
        "patch_sha256": hashlib.sha256(patch_text.encode("utf-8")).hexdigest(),
        "patch_apply_mode": _patch_apply_mode(prediction),
        "model_name_or_path": prediction.get("model_name_or_path", ""),
        **_swesmith_provenance(),
    }


def _patch_apply_mode(prediction: dict[str, Any]) -> str:
    mode = prediction.get("patch_apply_mode", "forward")
    if mode not in PATCH_APPLY_MODES:
        expected = ", ".join(sorted(PATCH_APPLY_MODES))
        raise ValueError(f"patch_apply_mode must be one of: {expected}")
    return str(mode)


@lru_cache(maxsize=1)
def _swesmith_provenance() -> dict[str, str | None]:
    provenance = package_provenance("swesmith", "swesmith")
    return {
        "swesmith_version": provenance["version"],
        "swesmith_revision": provenance["revision"],
        "swesmith_working_tree_diff_sha256": provenance[
            "working_tree_diff_sha256"
        ],
    }


def _load_cached_report(
    report_path: Path,
    cache_key_path: Path,
    expected_key: dict[str, Any],
) -> dict[str, Any] | None:
    if not report_path.is_file() or not cache_key_path.is_file():
        return None
    try:
        key = json.loads(cache_key_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if key != expected_key or not isinstance(report, dict):
        return None
    return report


def _pull_sif(
    sif_path: Path,
    image_uri: str,
    *,
    cache_dir: str | Path | None,
    dry_run: bool,
) -> list[str]:
    return pull_sif_if_missing(
        sif_path=sif_path,
        image_uri=image_uri,
        cache_dir=cache_dir,
        dry_run=dry_run,
    )


def _write_eval_script(
    log_dir: Path,
    *,
    test_command: str,
) -> Path:
    try:
        from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START
    except ImportError as exc:
        raise RuntimeError(
            "SWE-smith is not installed. Run scripts/install_swesmith.sh first."
        ) from exc

    path = log_dir / "eval.sh"
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -uxo pipefail",
                f"cd {TESTBED_DIR}",
                f": {shlex.quote(TEST_OUTPUT_START)}",
                test_command,
                f": {shlex.quote(TEST_OUTPUT_END)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _write_runner_script(
    log_dir: Path,
    *,
    instance_id: str,
    test_files: list[str],
    patch_apply_mode: str = "forward",
) -> Path:
    if patch_apply_mode not in PATCH_APPLY_MODES:
        expected = ", ".join(sorted(PATCH_APPLY_MODES))
        raise ValueError(f"patch_apply_mode must be one of: {expected}")
    quoted_instance = shlex.quote(instance_id)
    quoted_test_files = " ".join(shlex.quote(str(path)) for path in test_files)
    revert_tests = (
        f"git checkout -- {quoted_test_files}\n" if quoted_test_files else ""
    )
    git_reverse_flag = " --reverse" if patch_apply_mode == "reverse" else ""
    patch_reverse_flag = " -R" if patch_apply_mode == "reverse" else ""
    path = log_dir / "run_apptainer_eval.sh"
    path.write_text(
        f"""#!/usr/bin/env bash
set -uo pipefail
cd {TESTBED_DIR} || exit 20
export GIT_CONFIG_GLOBAL="{EVAL_BIND_DIR}/gitconfig"
git config --global --replace-all safe.directory {TESTBED_DIR} || true
git fetch >/dev/null 2>&1 || true
git checkout --force {quoted_instance} || exit 21
git checkout --force HEAD~1 || exit 22
apply_log="{EVAL_BIND_DIR}/apply_patch.log"
: > "$apply_log"
applied=0
for git_apply_cmd in \
  "git apply --verbose{git_reverse_flag}" \
  "git apply --verbose --reject{git_reverse_flag}" \
  "patch --batch --fuzz=5{patch_reverse_flag} -p1 -i"
do
  if $git_apply_cmd "{EVAL_BIND_DIR}/patch.diff" >> "$apply_log" 2>&1; then
    applied=1
    break
  fi
done
if [[ "$applied" != "1" ]]; then
  echo failed > "{EVAL_BIND_DIR}/patch_status.txt"
  cp "$apply_log" "{EVAL_BIND_DIR}/test_output.txt"
  exit 23
fi
echo applied > "{EVAL_BIND_DIR}/patch_status.txt"
{revert_tests}bash "{EVAL_BIND_DIR}/eval.sh" > "{EVAL_BIND_DIR}/test_output.txt" 2>&1
exit $?
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _apptainer_command(
    sif_path: Path,
    log_dir: Path,
    extra_args: list[str],
) -> list[str]:
    return [
        "apptainer",
        "exec",
        "--writable-tmpfs",
        "--bind",
        f"{log_dir.resolve()}:{EVAL_BIND_DIR}",
        *extra_args,
        str(sif_path),
        "/bin/bash",
        f"{EVAL_BIND_DIR}/run_apptainer_eval.sh",
    ]


def _grade_prediction(
    prediction: dict[str, Any],
    instance: dict[str, Any],
    test_output_path: Path,
    *,
    f2p_only: bool,
) -> dict[str, Any]:
    try:
        from swesmith.harness.grading import get_eval_report
    except ImportError as exc:
        raise RuntimeError(
            "SWE-smith is not installed. Run scripts/install_swesmith.sh first."
        ) from exc
    report = get_eval_report(prediction, instance, test_output_path, f2p_only=f2p_only)
    report["model_name_or_path"] = prediction.get("model_name_or_path", "")
    return report


def _run_mock(
    instance: dict[str, Any],
    prediction: dict[str, Any],
    log_dir: Path,
    cache_key: dict[str, Any],
) -> dict[str, Any]:
    patch = prediction.get("model_patch", "")
    resolved = bool(isinstance(patch, str) and patch.strip())
    report = {
        "instance_id": str(instance["instance_id"]),
        "resolved": resolved,
        "mock": True,
        "model_name_or_path": prediction.get("model_name_or_path", ""),
    }
    write_json(log_dir / "report.json", report)
    write_json(log_dir / "cache_key.json", cache_key)
    return {
        "instance_id": str(instance["instance_id"]),
        "status": "completed",
        "resolved": resolved,
        "report_path": str(log_dir / "report.json"),
    }


def run_instance(
    instance: dict[str, Any],
    prediction: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    instance_id = str(instance["instance_id"])
    log_dir = Path(args.log_dir) / instance_id
    report_path = log_dir / "report.json"
    cache_key_path = log_dir / "cache_key.json"
    cache_key = _cache_key(prediction, args)
    report = (
        None
        if args.overwrite
        else _load_cached_report(report_path, cache_key_path, cache_key)
    )
    if report is not None:
        return {
            "instance_id": instance_id,
            "status": "cached_report",
            "resolved": bool(report.get("resolved", False)),
            "report_path": str(report_path),
        }

    patch = prediction.get("model_patch")
    if not isinstance(patch, str) or not patch.strip():
        if not args.dry_run:
            report_path.unlink(missing_ok=True)
            cache_key_path.unlink(missing_ok=True)
        return {"instance_id": instance_id, "status": "empty_patch", "resolved": False}
    patch_apply_mode = _patch_apply_mode(prediction)
    if args.runtime == "mock":
        if args.dry_run:
            return {
                "instance_id": instance_id,
                "status": "dry_run",
                "resolved": False,
                "runtime": "mock",
                "patch_apply_mode": patch_apply_mode,
            }
        ensure_dir(log_dir)
        report_path.unlink(missing_ok=True)
        cache_key_path.unlink(missing_ok=True)
        return _run_mock(instance, prediction, log_dir, cache_key)

    profile = _profile_for(instance)
    image_name = str(profile.image_name)
    image_uri = _image_uri(image_name)
    sif_path = _sif_path(args.sif_dir, image_name)
    pull_command = _pull_sif(
        sif_path,
        image_uri,
        cache_dir=args.apptainer_cache_dir,
        dry_run=args.dry_run,
    )
    test_command, _ = profile.get_test_cmd(instance, f2p_only=args.f2p_only)
    f2p_files, p2p_files = profile.get_test_files(instance)
    test_files = list(f2p_files) + list(p2p_files)

    command = _apptainer_command(sif_path, log_dir, list(args.apptainer_arg or []))
    metadata = {
        "instance_id": instance_id,
        "image_name": image_name,
        "image_uri": image_uri,
        "sif_path": str(sif_path),
        "pull_command": pull_command,
        "command": command,
        "temperature": prediction.get("temperature"),
        "temperature_run_index": prediction.get("temperature_run_index"),
        "seed": prediction.get("seed"),
        "patch_apply_mode": patch_apply_mode,
    }
    if args.dry_run:
        return {
            "instance_id": instance_id,
            "status": "dry_run",
            "resolved": False,
            **metadata,
        }

    ensure_dir(log_dir)
    report_path.unlink(missing_ok=True)
    cache_key_path.unlink(missing_ok=True)
    (log_dir / "patch.diff").write_text(patch, encoding="utf-8")
    _write_eval_script(log_dir, test_command=str(test_command))
    _write_runner_script(
        log_dir,
        instance_id=instance_id,
        test_files=test_files,
        patch_apply_mode=patch_apply_mode,
    )
    write_json(log_dir / "metadata.json", metadata)
    timeout = args.timeout if args.timeout is not None else int(profile.timeout)
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        (log_dir / "apptainer_stdout.txt").write_text(
            str(exc.stdout or ""), encoding="utf-8"
        )
        (log_dir / "apptainer_stderr.txt").write_text(
            str(exc.stderr or ""), encoding="utf-8"
        )
        return {"instance_id": instance_id, "status": "timeout", "resolved": False}

    (log_dir / "apptainer_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (log_dir / "apptainer_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    metadata["returncode"] = completed.returncode
    write_json(log_dir / "metadata.json", metadata)
    patch_status = log_dir / "patch_status.txt"
    if not patch_status.exists() or patch_status.read_text(encoding="utf-8").strip() != "applied":
        return {
            "instance_id": instance_id,
            "status": "patch_failed",
            "resolved": False,
            "returncode": completed.returncode,
        }
    test_output = log_dir / "test_output.txt"
    if not test_output.exists():
        return {
            "instance_id": instance_id,
            "status": "error",
            "resolved": False,
            "returncode": completed.returncode,
        }

    report = _grade_prediction(
        prediction,
        instance,
        test_output,
        f2p_only=args.f2p_only,
    )
    report["instance_id"] = instance_id
    write_json(report_path, report)
    write_json(cache_key_path, cache_key)
    return {
        "instance_id": instance_id,
        "status": "completed",
        "resolved": bool(report.get("resolved", False)),
        "returncode": completed.returncode,
        "report_path": str(report_path),
    }


def evaluate_swesmith(args: argparse.Namespace) -> dict[str, Any]:
    load_hf_token_from_file()
    predictions_list = _load_predictions(args.predictions_path)
    predictions = {str(row["instance_id"]): row for row in predictions_list}
    requested_ids = list(args.instance_ids or [])
    if args.instance_ids_file:
        requested_ids.extend(read_instance_ids_file(args.instance_ids_file))
    if requested_ids:
        requested = set(requested_ids)
        missing_requested = sorted(requested - set(predictions))
        if missing_requested:
            raise ValueError(
                f"{len(missing_requested)} requested instance IDs are absent from "
                f"the predictions: {missing_requested[:10]}"
            )
        predictions = {
            instance_id: row
            for instance_id, row in predictions.items()
            if instance_id in requested
        }

    dataset_revision = (
        None if Path(args.dataset).is_file() else args.dataset_revision
    )
    args.dataset_revision = dataset_revision
    dataset_rows = load_swebench_tasks(
        args.dataset,
        args.split,
        revision=dataset_revision,
    )
    dataset = {
        str(instance["instance_id"]): instance
        for instance in dataset_rows
        if str(instance.get("instance_id", "")) in predictions
    }
    missing = sorted(set(predictions) - set(dataset))
    if missing:
        raise ValueError(
            f"{len(missing)} prediction instance IDs are absent from the dataset: {missing[:10]}"
        )

    runnable = [(dataset[instance_id], prediction) for instance_id, prediction in predictions.items()]
    results: list[dict[str, Any]] = []

    def run_safely(
        instance: dict[str, Any],
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return run_instance(instance, prediction, args)
        except Exception as exc:
            return {
                "instance_id": str(instance["instance_id"]),
                "status": "error",
                "resolved": False,
                "error": repr(exc),
            }

    if args.max_workers <= 1:
        iterator = tqdm(runnable, desc="SWE-smith evaluation", unit="task")
        for instance, prediction in iterator:
            results.append(run_safely(instance, prediction))
    else:
        progress = tqdm(total=len(runnable), desc="SWE-smith evaluation", unit="task")
        with progress:
            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futures = {
                    pool.submit(run_safely, instance, prediction): str(instance["instance_id"])
                    for instance, prediction in runnable
                }
                for future in as_completed(futures):
                    instance_id = futures[future]
                    result = future.result()
                    results.append(result)
                    progress.set_postfix(instance=instance_id)
                    progress.update(1)

    results.sort(key=lambda row: str(row["instance_id"]))
    by_status: dict[str, list[str]] = {}
    for result in results:
        by_status.setdefault(str(result["status"]), []).append(str(result["instance_id"]))
    resolved_ids = sorted(
        str(result["instance_id"]) for result in results if result.get("resolved")
    )
    unresolved_ids = sorted(
        str(result["instance_id"])
        for result in results
        if result.get("status") in SCORED_STATUSES and not result.get("resolved")
    )
    scored_instances = sum(
        result.get("status") in SCORED_STATUSES for result in results
    )
    summary = {
        "schema_version": 2,
        "runtime": args.runtime,
        "dataset": args.dataset,
        "dataset_revision": dataset_revision,
        "split": args.split,
        **_swesmith_provenance(),
        "predictions_path": str(args.predictions_path),
        "submitted_instances": len(predictions),
        "completed_instances": sum(
            result.get("status") in {"completed", "cached_report"} for result in results
        ),
        "scored_instances": scored_instances,
        "resolved_instances": len(resolved_ids),
        "resolution_rate": (
            len(resolved_ids) / scored_instances if scored_instances else None
        ),
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "status_ids": {status: sorted(ids) for status, ids in sorted(by_status.items())},
        "results": results,
    }
    if not args.dry_run:
        write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2))
    if args.require_complete:
        failed = [
            result
            for result in results
            if str(result.get("status")) not in SCORED_STATUSES
        ]
        if failed:
            counts = {
                status: len(ids)
                for status, ids in summary["status_ids"].items()
                if status not in SCORED_STATUSES
            }
            raise RuntimeError(
                "SWE-smith evaluation produced unscored infrastructure outcomes: "
                f"{counts}"
            )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one SWE-smith prediction sample.")
    parser.add_argument("--dataset", default=DEFAULT_SWESMITH_DATASET)
    parser.add_argument("--split", default=DEFAULT_SWESMITH_SPLIT)
    parser.add_argument(
        "--dataset-revision",
        default=os.getenv(
            "SWESMITH_DATASET_REVISION",
            DEFAULT_SWESMITH_DATASET_REVISION,
        ),
    )
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--sif-dir", default="data/apptainer/swesmith-sifs")
    parser.add_argument("--apptainer-cache-dir")
    parser.add_argument("--runtime", choices=("apptainer", "mock"), default="apptainer")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--instance-ids-file")
    parser.add_argument("--instance-id", action="append", dest="instance_ids")
    parser.add_argument("--f2p-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--apptainer-arg", action="append")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1")
    if args.runtime == "apptainer" and not args.dry_run and shutil.which("apptainer") is None:
        raise SystemExit("apptainer is not available on PATH")
    evaluate_swesmith(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
