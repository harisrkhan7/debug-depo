"""Collect repeated, temperature-swept trajectories on SWE-smith tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from debug_depo.agentforge import AgentForgeConfig, prediction_record, run_agentforge_instance
from debug_depo.constants import (
    DEFAULT_AGENTFORGE_MODEL,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_STEPS,
    DEFAULT_SWESMITH_DATASET,
    DEFAULT_SWESMITH_DATASET_REVISION,
    DEFAULT_SWESMITH_SPLIT,
    DEFAULT_TOP_P,
)
from debug_depo.data import load_swebench_tasks, read_instance_ids_file, select_tasks
from debug_depo.rollout import result_from_existing
from debug_depo.utils import (
    ensure_dir,
    package_provenance,
    read_json,
    utc_now,
    write_json,
    write_jsonl,
)


DEFAULT_TEMPERATURES = (0.6, 0.7)
DEFAULT_RUNS_PER_TEMPERATURE = 4
DEFAULT_BASE_SEED = 42
FINISHED_ROLLOUT_STATUSES = frozenset({"completed", "mocked", "model_terminated"})
RESUME_MANIFEST_KEYS = (
    "dataset",
    "dataset_revision",
    "split",
    "model",
    "mini_swe_version",
    "mini_swe_revision",
    "mini_swe_working_tree_diff_sha256",
    "swesmith_version",
    "swesmith_revision",
    "swesmith_working_tree_diff_sha256",
    "num_shards",
    "shard_index",
    "expected_tasks",
    "n_tasks",
    "runs_per_temperature",
    "total_samples_per_task",
    "temperatures",
    "base_seed",
    "max_steps",
    "context_length",
    "top_p",
    "timeout_seconds",
    "mini_model",
    "mini_config",
    "mini_runner",
    "mini_environment_class",
    "task_initialization",
    "mock",
    "mock_patch",
    "task_instance_ids",
)


def parse_temperatures(spec: str) -> list[float]:
    """Parse a comma-, colon-, or whitespace-separated temperature list."""

    values = [part for part in re.split(r"[,:\s]+", spec.strip()) if part]
    if not values:
        raise ValueError("at least one temperature is required")
    temperatures = [float(value) for value in values]
    if any(not math.isfinite(temperature) or temperature < 0 for temperature in temperatures):
        raise ValueError("temperatures must be finite and non-negative")
    if len(set(temperatures)) != len(temperatures):
        raise ValueError("temperatures must be unique")
    return [round(temperature, 6) for temperature in temperatures]


def temperature_schedule(
    temperatures: list[float],
    runs_per_temperature: int,
) -> list[tuple[float, int]]:
    """Return ``(temperature, within-temperature run index)`` sample slots."""

    if runs_per_temperature < 1:
        raise ValueError("runs_per_temperature must be at least 1")
    return [
        (temperature, run_index)
        for temperature in temperatures
        for run_index in range(runs_per_temperature)
    ]


def rollout_seed(base_seed: int, instance_id: str, sample_index: int) -> int:
    """Derive a stable vLLM/LiteLLM-compatible seed for one trajectory."""

    digest = hashlib.sha256(
        f"{base_seed}\0{instance_id}\0{sample_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def selected_instance_ids(args: argparse.Namespace) -> list[str] | None:
    ids = list(args.instance_ids or [])
    if args.instance_ids_file:
        ids.extend(read_instance_ids_file(args.instance_ids_file))
    if len(ids) != len(set(ids)):
        raise ValueError("SWE-smith instance IDs must be unique")
    return ids or None


def _base_agent_config(
    args: argparse.Namespace,
    *,
    dataset_revision: str | None,
) -> AgentForgeConfig:
    return AgentForgeConfig(
        model=args.model,
        dataset=args.dataset,
        dataset_revision=dataset_revision,
        split=args.split,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        harness="mini-swe-agent-plus",
        mini_model=args.mini_model,
        mini_config=args.mini_config,
        mini_runner=args.mini_runner,
        mini_environment_class=args.mini_environment_class,
        mini_workers=args.mini_workers,
        mini_docker_start_concurrency=args.mini_docker_start_concurrency,
        initialize_swesmith_task=True,
        max_steps=args.max_steps,
        context_length=args.context_length,
        top_p=args.top_p,
        timeout_seconds=args.timeout_seconds,
        mock=args.mock,
        mock_patch=args.mock_patch,
        stream_output=args.stream_output,
    )


def _sample_dir(output_dir: Path, sample_index: int) -> Path:
    return output_dir / "samples" / f"sample-{sample_index}"


def _resume_mismatches(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> list[str]:
    return [
        key
        for key in RESUME_MANIFEST_KEYS
        if previous.get(key) != current.get(key)
    ]


def collect_swesmith(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = ensure_dir(args.output_dir)
    if args.rollout_workers < 1:
        raise ValueError("rollout_workers must be at least 1")
    if args.mini_runner == "pool_way" and not args.mock:
        raise ValueError(
            "SWE-smith collection cannot use mini-swe-agent-plus pool_way because "
            "that runner ignores task startup commands. Use the swebench runner "
            "with Docker, or use the singularity runner."
        )

    dataset_revision = (
        None if Path(args.dataset).is_file() else args.dataset_revision
    )
    selected_tasks = select_tasks(
        load_swebench_tasks(
            args.dataset,
            args.split,
            revision=dataset_revision,
        ),
        instance_ids=selected_instance_ids(args),
        start_index=args.start_index,
        limit=args.limit,
    )
    if args.expected_tasks is not None and len(selected_tasks) != args.expected_tasks:
        raise ValueError(
            "Expected "
            f"{args.expected_tasks} selected SWE-smith tasks, found {len(selected_tasks)}"
        )
    tasks = select_tasks(
        selected_tasks,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if not tasks:
        raise ValueError("No SWE-smith tasks selected")

    temperatures = parse_temperatures(args.temperatures)
    schedule = temperature_schedule(
        temperatures,
        args.runs_per_temperature,
    )
    total_samples = len(schedule)
    base_config = _base_agent_config(
        args,
        dataset_revision=dataset_revision,
    )
    mini_swe_provenance = package_provenance("mini-swe-agent", "minisweagent")
    swesmith_provenance = package_provenance("swesmith", "swesmith")
    run_at = utc_now()
    manifest: dict[str, Any] = {
        "schema_version": 3,
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
        "swesmith_version": swesmith_provenance["version"],
        "swesmith_revision": swesmith_provenance["revision"],
        "swesmith_working_tree_diff_sha256": swesmith_provenance[
            "working_tree_diff_sha256"
        ],
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "expected_tasks": args.expected_tasks,
        "n_tasks": len(tasks),
        "runs_per_temperature": args.runs_per_temperature,
        "total_samples_per_task": total_samples,
        "n_rollouts": len(tasks) * total_samples,
        "temperatures": temperatures,
        "base_seed": args.base_seed,
        "rollout_workers": args.rollout_workers,
        "max_steps": args.max_steps,
        "context_length": args.context_length,
        "top_p": args.top_p,
        "timeout_seconds": args.timeout_seconds,
        "mini_model": args.mini_model,
        "mini_config": args.mini_config,
        "mini_runner": args.mini_runner,
        "mini_environment_class": args.mini_environment_class,
        "task_initialization": "checkout_instance_branch",
        "mock": args.mock,
        "mock_patch": args.mock_patch,
        "task_instance_ids": [str(task["instance_id"]) for task in tasks],
    }
    manifest_path = output_dir / "collection_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        previous_manifest = read_json(manifest_path)
        if not isinstance(previous_manifest, dict):
            raise ValueError(f"Invalid existing collection manifest: {manifest_path}")
        mismatches = _resume_mismatches(previous_manifest, manifest)
        if mismatches:
            mismatch_list = ", ".join(mismatches)
            raise ValueError(
                "Existing collection is incompatible with this configuration "
                f"({mismatch_list}). Use a new output directory or --overwrite."
            )
        manifest["created_at"] = previous_manifest.get("created_at", run_at)
    write_json(manifest_path, manifest)

    collector_id = f"{os.getpid()}-{run_at}"
    rollout_events_path = output_dir / "rollout_events.jsonl"
    active_rollouts_path = output_dir / "active_rollouts.json"
    diagnostic_lock = threading.Lock()
    active_rollouts: dict[str, dict[str, Any]] = {}

    def append_diagnostic_event(payload: dict[str, Any]) -> None:
        with rollout_events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()

    def write_active_rollouts(updated_at: str) -> None:
        write_json(
            active_rollouts_path,
            {
                "schema_version": 1,
                "collector_id": collector_id,
                "updated_at": updated_at,
                "active": list(active_rollouts.values()),
            },
        )

    with diagnostic_lock:
        diagnostic_at = utc_now()
        append_diagnostic_event(
            {
                "schema_version": 1,
                "event": "collector_started",
                "at": diagnostic_at,
                "collector_id": collector_id,
                "pid": os.getpid(),
            }
        )
        write_active_rollouts(diagnostic_at)

    def record_rollout_event(
        event: str,
        rollout: dict[str, Any],
        *,
        status: str | None = None,
        error: str | None = None,
    ) -> None:
        event_at = utc_now()
        key = f"{rollout['task_index']}:{rollout['sample_index']}"
        event_payload = {
            "schema_version": 1,
            "event": event,
            "at": event_at,
            "collector_id": collector_id,
            **rollout,
        }
        if status is not None:
            event_payload["status"] = status
        if error is not None:
            event_payload["error"] = error

        with diagnostic_lock:
            if event == "rollout_started":
                active_rollouts[key] = {
                    **rollout,
                    "started_at": event_at,
                }
            else:
                active_rollouts.pop(key, None)
            append_diagnostic_event(event_payload)
            write_active_rollouts(event_at)

    jobs = [
        (task_index, task, sample_index, temperature, temperature_run_index)
        for task_index, task in enumerate(tasks)
        for sample_index, (temperature, temperature_run_index) in enumerate(schedule)
    ]

    def run_one(
        task_index: int,
        task: dict[str, Any],
        sample_index: int,
        temperature: float,
        temperature_run_index: int,
    ) -> tuple[int, int, dict[str, Any]]:
        instance_id = str(task["instance_id"])
        sample_dir = ensure_dir(_sample_dir(output_dir, sample_index))
        seed = rollout_seed(args.base_seed, instance_id, sample_index)
        config = replace(base_config, temperature=temperature, seed=seed)
        rollout_diagnostic = {
            "instance_id": instance_id,
            "task_index": task_index,
            "sample_index": sample_index,
            "temperature": temperature,
            "temperature_run_index": temperature_run_index,
            "seed": seed,
        }
        rollout_started = False
        diagnostic_error: str | None = None
        result: dict[str, Any] | None = None
        try:
            result = None if args.overwrite else result_from_existing(sample_dir, task)
            if result is None or result.get("status") == "error":
                record_rollout_event("rollout_started", rollout_diagnostic)
                rollout_started = True
                result = run_agentforge_instance(task, sample_dir, config)
        except Exception as exc:
            diagnostic_error = repr(exc)
            if args.stop_on_error:
                raise
            result = {
                "instance_id": instance_id,
                "status": "error",
                "error": repr(exc),
                "patch": "",
                "patch_source": None,
            }
        finally:
            if rollout_started:
                record_rollout_event(
                    "rollout_finished",
                    rollout_diagnostic,
                    status=(
                        str(result.get("status", "unknown"))
                        if result is not None
                        else "interrupted"
                    ),
                    error=diagnostic_error,
                )
        if result is None:
            raise RuntimeError(f"Rollout did not produce a result: {instance_id}")
        return task_index, sample_index, {
            **result,
            "sample_index": sample_index,
            "temperature": temperature,
            "temperature_run_index": temperature_run_index,
            "seed": seed,
        }

    results: dict[tuple[int, int], dict[str, Any]] = {}
    progress = tqdm(
        total=len(jobs),
        desc="SWE-smith trajectories",
        unit="rollout",
        disable=not args.progress,
    )
    with progress:
        if args.rollout_workers == 1:
            for job in jobs:
                task_index, sample_index, result = run_one(*job)
                results[(task_index, sample_index)] = result
                progress.set_postfix(
                    instance=result["instance_id"],
                    sample=sample_index,
                    temperature=result["temperature"],
                )
                progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=args.rollout_workers) as pool:
                futures = {pool.submit(run_one, *job): job for job in jobs}
                for future in as_completed(futures):
                    task_index, sample_index, result = future.result()
                    results[(task_index, sample_index)] = result
                    progress.set_postfix(
                        instance=result["instance_id"],
                        sample=sample_index,
                        temperature=result["temperature"],
                    )
                    progress.update(1)

    if len(results) != len(jobs):
        raise RuntimeError("Collection ended before every SWE-smith rollout produced a result")

    all_results: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []
    for sample_index, (temperature, temperature_run_index) in enumerate(schedule):
        sample_dir = ensure_dir(_sample_dir(output_dir, sample_index))
        sample_results = [
            results[(task_index, sample_index)] for task_index in range(len(tasks))
        ]
        predictions: list[dict[str, Any]] = []
        for task, result in zip(tasks, sample_results):
            row: dict[str, Any] = prediction_record(
                task,
                args.model,
                str(result.get("patch", "")),
            )
            row.update(
                {
                    "sample_index": sample_index,
                    "temperature": temperature,
                    "temperature_run_index": temperature_run_index,
                    "seed": result["seed"],
                }
            )
            if result.get("patch_source") == "gold":
                row["patch_apply_mode"] = "reverse"
            predictions.append(row)
        predictions_path = sample_dir / "predictions.jsonl"
        write_jsonl(predictions_path, predictions)
        sample_summary = {
            "schema_version": 1,
            "created_at": run_at,
            "sample_index": sample_index,
            "temperature": temperature,
            "temperature_run_index": temperature_run_index,
            "predictions_path": str(predictions_path),
            "n_tasks": len(tasks),
            "n_completed": sum(
                result.get("status") in {"completed", "mocked"} for result in sample_results
            ),
            "n_model_terminated": sum(
                result.get("status") == "model_terminated"
                for result in sample_results
            ),
            "n_finished": sum(
                result.get("status") in FINISHED_ROLLOUT_STATUSES
                for result in sample_results
            ),
            "n_errors": sum(result.get("status") == "error" for result in sample_results),
            "n_with_patch": sum(bool(result.get("patch")) for result in sample_results),
            "results": [
                {key: value for key, value in result.items() if key != "patch"}
                for result in sample_results
            ],
        }
        write_json(sample_dir / "summary.json", sample_summary)
        sample_summaries.append(sample_summary)
        all_results.extend(sample_results)

    summary = {
        **manifest,
        "n_completed": sum(
            result.get("status") in {"completed", "mocked"} for result in all_results
        ),
        "n_model_terminated": sum(
            result.get("status") == "model_terminated" for result in all_results
        ),
        "n_finished": sum(
            result.get("status") in FINISHED_ROLLOUT_STATUSES
            for result in all_results
        ),
        "n_errors": sum(result.get("status") == "error" for result in all_results),
        "n_with_patch": sum(bool(result.get("patch")) for result in all_results),
        "samples": [
            {
                key: value
                for key, value in sample.items()
                if key not in {"results", "created_at", "schema_version"}
            }
            for sample in sample_summaries
        ],
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if args.require_complete and summary["n_finished"] != summary["n_rollouts"]:
        raise RuntimeError(
            "SWE-smith collection did not finish every rollout: "
            f"{summary['n_finished']}/{summary['n_rollouts']} finished, "
            f"{summary['n_model_terminated']} model-terminated, "
            f"{summary['n_errors']} errors"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect repeated mini-swe-agent trajectories on SWE-smith."
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("SWESMITH_DATASET", DEFAULT_SWESMITH_DATASET),
    )
    parser.add_argument(
        "--split",
        default=os.getenv("SWESMITH_SPLIT", DEFAULT_SWESMITH_SPLIT),
    )
    parser.add_argument(
        "--dataset-revision",
        default=os.getenv(
            "SWESMITH_DATASET_REVISION",
            DEFAULT_SWESMITH_DATASET_REVISION,
        ),
    )
    parser.add_argument("--output-dir", default="data/processed/swesmith_collection")
    parser.add_argument("--instance-ids-file")
    parser.add_argument("--instance-id", action="append", dest="instance_ids")
    parser.add_argument("--start-index", type=int, default=int(os.getenv("START_INDEX", "0")))
    limit_default = os.getenv("LIMIT")
    parser.add_argument("--limit", type=int, default=limit_default or None)
    parser.add_argument("--expected-tasks", type=int)
    parser.add_argument("--num-shards", type=int, default=int(os.getenv("NUM_SHARDS", "1")))
    parser.add_argument("--shard-index", type=int, default=int(os.getenv("SHARD_INDEX", "0")))
    parser.add_argument(
        "--runs-per-temperature",
        type=int,
        default=int(
            os.getenv(
                "RUNS_PER_TEMPERATURE",
                str(DEFAULT_RUNS_PER_TEMPERATURE),
            )
        ),
    )
    parser.add_argument(
        "--temperatures",
        default=os.getenv(
            "TEMPERATURES",
            ":".join(str(temperature) for temperature in DEFAULT_TEMPERATURES),
        ),
        help="Comma-, colon-, or whitespace-separated temperature values.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=int(os.getenv("BASE_SEED", str(DEFAULT_BASE_SEED))),
    )
    parser.add_argument("--model", default=os.getenv("AGENTFORGE_MODEL", DEFAULT_AGENTFORGE_MODEL))
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL"))
    parser.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY"))
    parser.add_argument("--mini-model", default=os.getenv("MINI_SWE_MODEL"))
    parser.add_argument("--mini-config", default=os.getenv("MINI_SWE_CONFIG"))
    parser.add_argument(
        "--mini-runner",
        choices=("pool_way", "swebench", "singularity"),
        default=os.getenv("MINI_SWE_RUNNER", "singularity"),
    )
    parser.add_argument(
        "--mini-environment-class",
        choices=("docker", "singularity"),
        default=os.getenv("MINI_SWE_ENVIRONMENT_CLASS", "singularity"),
    )
    parser.add_argument("--mini-workers", type=int, default=int(os.getenv("MINI_SWE_WORKERS", "1")))
    parser.add_argument(
        "--mini-docker-start-concurrency",
        type=int,
        default=int(os.getenv("MINI_SWE_DOCKER_START_CONCURRENCY", "1")),
    )
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=int(os.getenv("ROLLOUT_WORKERS", "4")),
    )
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", DEFAULT_MAX_STEPS)))
    parser.add_argument(
        "--context-length",
        type=int,
        default=int(os.getenv("CONTEXT_LENGTH", DEFAULT_CONTEXT_LENGTH)),
    )
    parser.add_argument("--top-p", type=float, default=float(os.getenv("TOP_P", DEFAULT_TOP_P)))
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.getenv("TIMEOUT_SECONDS", "21600")),
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-patch", choices=("empty", "gold"), default="empty")
    parser.add_argument(
        "--stream-output",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("STREAM_OUTPUT", "0") == "1",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    collect_swesmith(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
