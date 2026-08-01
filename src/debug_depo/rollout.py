"""Collect AgentForge trajectories and SWE-bench prediction JSONL files."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from debug_depo.agentforge import (
    AgentForgeConfig,
    REDACTED_SECRET,
    miniswe_failure,
    miniswe_result_status,
    prediction_record,
    run_agentforge_instance,
)
from debug_depo.constants import (
    DEFAULT_AGENTFORGE_MODEL,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_STEPS,
    DEFAULT_SWEBENCH_DATASET,
    DEFAULT_SWEBENCH_SPLIT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from debug_depo.data import (
    load_swebench_tasks,
    read_instance_ids_file,
    resolve_swebench_dataset_revision,
    select_tasks,
    task_rows_sha256,
)
from debug_depo.utils import (
    ensure_dir,
    package_provenance,
    read_json,
    slugify,
    utc_now,
    write_json,
    write_jsonl,
)


FINISHED_ROLLOUT_STATUSES = frozenset({"completed", "mocked", "model_terminated"})
RESUME_CONFIG_KEYS = (
    "dataset",
    "dataset_revision",
    "split",
    "model",
    "mini_swe_version",
    "mini_swe_revision",
    "mini_swe_working_tree_diff_sha256",
    "max_steps",
    "context_length",
    "temperature",
    "top_p",
    "timeout_seconds",
    "num_shards",
    "shard_index",
    "n_tasks",
    "mock",
    "mock_patch",
    "harness",
    "agentforge_command",
    "mini_model",
    "mini_config",
    "mini_runner",
    "mini_environment_class",
    "mini_image_template",
    "task_instance_ids",
    "task_rows_sha256",
)


def selected_instance_ids(args: argparse.Namespace) -> list[str] | None:
    ids = list(args.instance_ids or [])
    if args.instance_ids_file:
        ids.extend(read_instance_ids_file(args.instance_ids_file))
    return ids or None


def trajectory_path(output_dir: str | Path, instance_id: str) -> Path:
    return Path(output_dir) / "trajectories" / slugify(instance_id) / "trajectory.json"


def result_from_existing(output_dir: str | Path, instance: dict[str, Any]) -> dict[str, Any] | None:
    path = trajectory_path(output_dir, str(instance["instance_id"]))
    if not path.exists():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict):
        return None
    patch = payload.get("patch")
    if not isinstance(patch, str):
        patch = ""
    mini_failure = miniswe_failure(path.parent)
    status = str(payload.get("status", "completed"))
    if mini_failure is not None:
        status = miniswe_result_status(mini_failure[0])
        patch = ""
    result = {
        "instance_id": str(instance["instance_id"]),
        "status": status,
        "patch": patch,
        "patch_source": payload.get("patch_source"),
        "trajectory_path": str(path),
    }
    if mini_failure is not None:
        result["mini_swe_exit_status"] = mini_failure[0]
        result["mini_swe_exit_status_path"] = mini_failure[1]
    return result


def _resume_mismatches(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> list[str]:
    return [
        key
        for key in RESUME_CONFIG_KEYS
        if previous.get(key) != current.get(key)
    ]


def collect_rollouts(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = ensure_dir(args.output_dir)
    if args.rollout_workers < 1:
        raise ValueError("rollout_workers must be at least 1")
    ids = selected_instance_ids(args)
    dataset_revision = resolve_swebench_dataset_revision(
        args.dataset,
        args.dataset_revision,
    )
    args.dataset_revision = dataset_revision
    tasks = select_tasks(
        load_swebench_tasks(
            args.dataset,
            args.split,
            revision=dataset_revision,
        ),
        instance_ids=ids,
        start_index=args.start_index,
        limit=args.limit,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if not tasks:
        raise ValueError("No SWE-bench tasks selected")

    config = AgentForgeConfig(
        model=args.model,
        dataset=args.dataset,
        dataset_revision=dataset_revision,
        split=args.split,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        harness=args.harness,
        command=args.agentforge_command,
        cwd=args.agentforge_cwd,
        mini_model=args.mini_model,
        mini_config=args.mini_config,
        mini_runner=args.mini_runner,
        mini_environment_class=args.mini_environment_class,
        mini_image_template=args.mini_image_template,
        mini_workers=args.mini_workers,
        mini_docker_start_concurrency=args.mini_docker_start_concurrency,
        max_steps=args.max_steps,
        context_length=args.context_length,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_seconds=args.timeout_seconds,
        mock=args.mock,
        mock_patch=args.mock_patch,
        stream_output=args.stream_output,
    )

    persisted_agentforge_command = args.agentforge_command
    if args.llm_api_key and persisted_agentforge_command:
        persisted_agentforge_command = persisted_agentforge_command.replace(
            args.llm_api_key,
            REDACTED_SECRET,
        )
    mini_swe_provenance = (
        package_provenance("mini-swe-agent", "minisweagent")
        if args.harness == "mini-swe-agent-plus"
        else {
            "version": None,
            "revision": None,
            "working_tree_diff_sha256": None,
        }
    )
    run_at = utc_now()
    run_config: dict[str, Any] = {
        "schema_version": 2,
        "created_at": run_at,
        "updated_at": run_at,
        "dataset": args.dataset,
        "dataset_revision": dataset_revision,
        "split": args.split,
        "model": args.model,
        "mini_swe_version": mini_swe_provenance["version"],
        "mini_swe_revision": mini_swe_provenance["revision"],
        "mini_swe_working_tree_diff_sha256": mini_swe_provenance[
            "working_tree_diff_sha256"
        ],
        "max_steps": args.max_steps,
        "context_length": args.context_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "timeout_seconds": args.timeout_seconds,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "n_tasks": len(tasks),
        "mock": args.mock,
        "mock_patch": args.mock_patch,
        "harness": args.harness,
        "agentforge_command": persisted_agentforge_command,
        "rollout_workers": args.rollout_workers,
        "mini_model": args.mini_model,
        "mini_config": args.mini_config,
        "mini_runner": args.mini_runner,
        "mini_environment_class": args.mini_environment_class,
        "mini_image_template": args.mini_image_template,
        "mini_workers": args.mini_workers,
        "mini_docker_start_concurrency": args.mini_docker_start_concurrency,
        "task_instance_ids": [str(task["instance_id"]) for task in tasks],
        "task_rows_sha256": task_rows_sha256(tasks),
    }
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists() and not args.overwrite:
        previous_config = read_json(run_config_path)
        if not isinstance(previous_config, dict):
            raise ValueError(f"Invalid existing run configuration: {run_config_path}")
        mismatches = _resume_mismatches(previous_config, run_config)
        if mismatches:
            raise ValueError(
                "Existing collection is incompatible with this configuration "
                f"({', '.join(mismatches)}). Use a new output directory or --overwrite."
            )
        run_config["created_at"] = previous_config.get("created_at", run_at)
    write_json(run_config_path, run_config)

    def run_one(index: int, instance: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        instance_id = str(instance["instance_id"])
        try:
            result = None if args.overwrite else result_from_existing(output_dir, instance)
            if result is None or result.get("status") == "error":
                result = run_agentforge_instance(instance, output_dir, config)
        except Exception as exc:
            if args.stop_on_error:
                raise
            result = {
                "instance_id": instance_id,
                "status": "error",
                "error": repr(exc),
                "patch": "",
                "patch_source": None,
            }
        return index, result

    results_by_index: list[dict[str, Any] | None] = [None] * len(tasks)
    if args.rollout_workers == 1:
        progress = tqdm(
            enumerate(tasks),
            total=len(tasks),
            desc="AgentForge SWE-bench rollouts",
            unit="task",
            disable=not args.progress,
        )
        for index, instance in progress:
            progress.set_postfix(instance=str(instance["instance_id"]))
            result_index, result = run_one(index, instance)
            results_by_index[result_index] = result
    else:
        progress = tqdm(
            total=len(tasks),
            desc="AgentForge SWE-bench rollouts",
            unit="task",
            disable=not args.progress,
        )
        with progress:
            with ThreadPoolExecutor(max_workers=args.rollout_workers) as pool:
                futures = {
                    pool.submit(run_one, index, instance): (index, str(instance["instance_id"]))
                    for index, instance in enumerate(tasks)
                }
                for future in as_completed(futures):
                    index, instance_id = futures[future]
                    result_index, result = future.result()
                    results_by_index[result_index] = result
                    progress.set_postfix(instance=instance_id)
                    progress.update(1)

    if any(result is None for result in results_by_index):
        raise RuntimeError("Rollout collection ended before every selected task produced a result")

    results = [result for result in results_by_index if result is not None]
    predictions = [
        prediction_record(instance, args.model, str(result.get("patch", "")))
        for instance, result in zip(tasks, results)
    ]

    predictions_path = Path(args.predictions_file) if args.predictions_file else output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    summary = {
        **run_config,
        "predictions_path": str(predictions_path),
        "n_completed": sum(result.get("status") in {"completed", "mocked"} for result in results),
        "n_model_terminated": sum(
            result.get("status") == "model_terminated" for result in results
        ),
        "n_finished": sum(
            result.get("status") in FINISHED_ROLLOUT_STATUSES for result in results
        ),
        "n_errors": sum(result.get("status") == "error" for result in results),
        "n_with_patch": sum(bool(result.get("patch")) for result in results),
        "results": [
            {key: value for key, value in result.items() if key != "patch"}
            for result in results
        ],
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if args.require_complete and summary["n_finished"] != summary["n_tasks"]:
        raise RuntimeError(
            "SWE-bench collection did not finish every rollout: "
            f"{summary['n_finished']}/{summary['n_tasks']} finished, "
            f"{summary['n_model_terminated']} model-terminated, "
            f"{summary['n_errors']} errors"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AgentForge on SWE-bench tasks and write prediction JSONL."
    )
    parser.add_argument("--dataset", default=os.getenv("SWEBENCH_DATASET", DEFAULT_SWEBENCH_DATASET))
    parser.add_argument(
        "--dataset-revision",
        default=os.getenv("SWEBENCH_DATASET_REVISION"),
        help=(
            "Hugging Face dataset revision. The proven SWE-bench Verified revision "
            "is selected automatically for the default dataset."
        ),
    )
    parser.add_argument("--split", default=os.getenv("SWEBENCH_SPLIT", DEFAULT_SWEBENCH_SPLIT))
    parser.add_argument(
        "--output-dir",
        default="data/processed/agentforge_swebench_verified",
    )
    parser.add_argument("--predictions-file")
    parser.add_argument("--instance-ids-file")
    parser.add_argument("--instance-id", action="append", dest="instance_ids")
    parser.add_argument("--start-index", type=int, default=int(os.getenv("START_INDEX", "0")))
    parser.add_argument("--limit", type=int, default=os.getenv("LIMIT"))
    parser.add_argument("--num-shards", type=int, default=int(os.getenv("NUM_SHARDS", "1")))
    parser.add_argument("--shard-index", type=int, default=int(os.getenv("SHARD_INDEX", "0")))
    parser.add_argument("--model", default=os.getenv("AGENTFORGE_MODEL", DEFAULT_AGENTFORGE_MODEL))
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL"))
    parser.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY"))
    parser.add_argument(
        "--harness",
        choices=("command", "mini-swe-agent-plus"),
        default=os.getenv("HARNESS", "command"),
        help="Use a custom command template or the official mini-swe-agent-plus harness.",
    )
    parser.add_argument("--agentforge-command", default=os.getenv("AGENTFORGE_COMMAND"))
    parser.add_argument("--agentforge-cwd", default=os.getenv("AGENTFORGE_REPO"))
    parser.add_argument(
        "--mini-model",
        default=os.getenv("MINI_SWE_MODEL"),
        help="Model string passed to mini-swe-agent-plus, e.g. hosted_vllm/model_name.",
    )
    parser.add_argument(
        "--mini-config",
        default=os.getenv("MINI_SWE_CONFIG"),
        help="Optional mini-swe-agent-plus config path.",
    )
    parser.add_argument(
        "--mini-image-template",
        default=os.getenv("MINI_SWE_IMAGE_TEMPLATE"),
        help=(
            "Optional image template injected into local mini-swe task rows. "
            "Use the evaluator's template to share prebuilt SIFs."
        ),
    )
    parser.add_argument(
        "--mini-runner",
        choices=("pool_way", "swebench", "singularity"),
        default=os.getenv("MINI_SWE_RUNNER", "pool_way"),
        help=(
            "mini-swe-agent-plus SWE-bench runner. pool_way is the current Docker-only "
            "pool runner; swebench is the standard mini-swe runner; singularity is a "
            "shortcut for swebench with --environment-class singularity."
        ),
    )
    parser.add_argument(
        "--mini-environment-class",
        choices=("docker", "singularity"),
        default=os.getenv("MINI_SWE_ENVIRONMENT_CLASS"),
        help="Task container backend for the standard mini-swe SWE-bench runner.",
    )
    parser.add_argument("--mini-workers", type=int, default=int(os.getenv("MINI_SWE_WORKERS", "1")))
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=int(os.getenv("ROLLOUT_WORKERS", "1")),
        help="Number of SWE-bench instances to run concurrently in this collector process.",
    )
    parser.add_argument(
        "--mini-docker-start-concurrency",
        type=int,
        default=int(os.getenv("MINI_SWE_DOCKER_START_CONCURRENCY", "1")),
    )
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", DEFAULT_MAX_STEPS)))
    parser.add_argument(
        "--context-length",
        type=int,
        default=int(os.getenv("CONTEXT_LENGTH", DEFAULT_CONTEXT_LENGTH)),
    )
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("TOP_P", DEFAULT_TOP_P)))
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.getenv("TIMEOUT_SECONDS", "7200")),
    )
    parser.add_argument("--mock", action="store_true", help="Write smoke-test trajectories only.")
    parser.add_argument("--mock-patch", choices=("empty", "gold"), default="empty")
    parser.add_argument(
        "--stream-output",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("STREAM_OUTPUT", "0") == "1",
        help="Stream harness stdout/stderr live while also writing log files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bars.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if isinstance(args.limit, str):
        args.limit = int(args.limit) if args.limit else None
    if not args.mock and args.harness == "command" and not args.agentforge_command:
        parser.error("Provide --agentforge-command for a real run, or use --mock for a smoke run.")
    collect_rollouts(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
