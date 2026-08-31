"""Collect repeated, temperature-swept trajectories on SWE-smith tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from tqdm.auto import tqdm

from debug_depo.agentforge import (
    AgentForgeConfig,
    active_subprocess_count,
    prediction_record,
    run_agentforge_instance,
    terminate_active_subprocesses,
)
from debug_depo.constants import (
    DEFAULT_AGENTFORGE_MODEL,
    DEFAULT_MAX_STEPS,
    DEFAULT_SWESMITH_CONTEXT_LENGTH,
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

    digest = hashlib.sha256(f"{base_seed}\0{instance_id}\0{sample_index}".encode("utf-8")).digest()
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
    return [key for key in RESUME_MANIFEST_KEYS if previous.get(key) != current.get(key)]


def _validate_collection_args(args: argparse.Namespace) -> bool:
    if args.rollout_workers < 1:
        raise ValueError("rollout_workers must be at least 1")
    if args.recovery_replicas < 1:
        raise ValueError("recovery_replicas must be at least 1")

    recovery_mode = args.recovery_replica_index is not None
    if recovery_mode:
        if not 0 <= args.recovery_replica_index < args.recovery_replicas:
            raise ValueError("recovery_replica_index must satisfy 0 <= index < recovery_replicas")
        if not args.recovery_run_id or not re.fullmatch(r"[A-Za-z0-9._-]+", args.recovery_run_id):
            raise ValueError(
                "recovery_run_id is required in recovery mode and may contain "
                "only letters, numbers, dots, underscores, and dashes"
            )
        if args.overwrite:
            raise ValueError("Recovery mode cannot be combined with --overwrite")
    elif args.recovery_replicas != 1 or args.recovery_run_id is not None:
        raise ValueError("recovery_replicas and recovery_run_id require --recovery-replica-index")

    if args.mini_runner == "pool_way" and not args.mock:
        raise ValueError(
            "SWE-smith collection cannot use mini-swe-agent-plus pool_way because "
            "that runner ignores task startup commands. Use the swebench runner "
            "with Docker, or use the singularity runner."
        )
    return recovery_mode


def _select_collection_tasks(
    args: argparse.Namespace,
    dataset_revision: str | None,
) -> list[dict[str, Any]]:
    selected_tasks = select_tasks(
        load_swebench_tasks(args.dataset, args.split, revision=dataset_revision),
        instance_ids=selected_instance_ids(args),
        start_index=args.start_index,
        limit=args.limit,
    )
    if args.expected_tasks is not None and len(selected_tasks) != args.expected_tasks:
        raise ValueError(
            f"Expected {args.expected_tasks} selected SWE-smith tasks, found {len(selected_tasks)}"
        )
    tasks = select_tasks(
        selected_tasks,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if not tasks:
        raise ValueError("No SWE-smith tasks selected")
    return tasks


def _collection_manifest(
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
    temperatures: list[float],
    total_samples: int,
    dataset_revision: str | None,
) -> tuple[dict[str, Any], str]:
    mini_swe_provenance = package_provenance("mini-swe-agent", "minisweagent")
    swesmith_provenance = package_provenance("swesmith", "swesmith")
    run_at = utc_now()
    return {
        "schema_version": 3,
        "created_at": run_at,
        "updated_at": run_at,
        "dataset": args.dataset,
        "dataset_revision": dataset_revision,
        "split": args.split,
        "model": args.model,
        "mini_swe_version": mini_swe_provenance["version"],
        "mini_swe_revision": mini_swe_provenance["revision"],
        "mini_swe_working_tree_diff_sha256": mini_swe_provenance["working_tree_diff_sha256"],
        "swesmith_version": swesmith_provenance["version"],
        "swesmith_revision": swesmith_provenance["revision"],
        "swesmith_working_tree_diff_sha256": swesmith_provenance["working_tree_diff_sha256"],
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
    }, run_at


def _prepare_collection_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    manifest: dict[str, Any],
    recovery_mode: bool,
    run_at: str,
) -> None:
    manifest_path = output_dir / "collection_manifest.json"
    if recovery_mode and not manifest_path.exists():
        raise ValueError(f"Recovery requires an existing collection manifest: {manifest_path}")
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
    if not recovery_mode:
        write_json(manifest_path, manifest)


def _rollout_event_recorder(
    args: argparse.Namespace,
    output_dir: Path,
    recovery_mode: bool,
    run_at: str,
) -> Callable[..., None]:
    collector_id = f"{os.getpid()}-{run_at}"
    if recovery_mode:
        diagnostic_suffix = f"recovery-{args.recovery_run_id}-replica-{args.recovery_replica_index}"
        rollout_events_path = output_dir / f"rollout_events.{diagnostic_suffix}.jsonl"
        active_rollouts_path = output_dir / f"active_rollouts.{diagnostic_suffix}.json"
        diagnostic_context: dict[str, Any] = {
            "recovery_run_id": args.recovery_run_id,
            "recovery_replicas": args.recovery_replicas,
            "recovery_replica_index": args.recovery_replica_index,
        }
    else:
        rollout_events_path = output_dir / "rollout_events.jsonl"
        active_rollouts_path = output_dir / "active_rollouts.json"
        diagnostic_context = {}

    diagnostic_lock = threading.Lock()
    active_rollouts: dict[str, dict[str, Any]] = {}

    def append_event(payload: dict[str, Any]) -> None:
        with rollout_events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()

    def write_active(updated_at: str) -> None:
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
        append_event(
            {
                "schema_version": 1,
                "event": "collector_started",
                "at": diagnostic_at,
                "collector_id": collector_id,
                "pid": os.getpid(),
                **diagnostic_context,
            }
        )
        write_active(diagnostic_at)

    def record(
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
            **diagnostic_context,
            **rollout,
        }
        if status is not None:
            event_payload["status"] = status
        if error is not None:
            event_payload["error"] = error

        with diagnostic_lock:
            if event == "rollout_started":
                active_rollouts[key] = {**rollout, "started_at": event_at}
            else:
                active_rollouts.pop(key, None)
            append_event(event_payload)
            write_active(event_at)

    return record


def _rollout_jobs(
    tasks: list[dict[str, Any]],
    schedule: list[tuple[float, int]],
    args: argparse.Namespace,
    recovery_mode: bool,
) -> list[tuple[int, dict[str, Any], int, float, int]]:
    jobs = [
        (task_index, task, sample_index, temperature, temperature_run_index)
        for task_index, task in enumerate(tasks)
        for sample_index, (temperature, temperature_run_index) in enumerate(schedule)
    ]
    if not recovery_mode:
        return jobs
    return [
        job
        for job_index, job in enumerate(jobs)
        if job_index % args.recovery_replicas == args.recovery_replica_index
    ]


def _run_rollout(
    job: tuple[int, dict[str, Any], int, float, int],
    *,
    args: argparse.Namespace,
    output_dir: Path,
    base_config: AgentForgeConfig,
    record_event: Callable[..., None],
) -> tuple[int, int, dict[str, Any]]:
    task_index, task, sample_index, temperature, temperature_run_index = job
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
            record_event("rollout_started", rollout_diagnostic)
            rollout_started = True
            result = run_agentforge_instance(task, sample_dir, config)
    except Exception as exc:
        diagnostic_error = repr(exc)
        if args.stop_on_error:
            raise
        result = {
            "instance_id": instance_id,
            "status": "error",
            "error": diagnostic_error,
            "patch": "",
            "patch_source": None,
        }
    finally:
        if rollout_started:
            record_event(
                "rollout_finished",
                rollout_diagnostic,
                status=(
                    str(result.get("status", "unknown")) if result is not None else "interrupted"
                ),
                error=diagnostic_error,
            )
    if result is None:
        raise RuntimeError(f"Rollout did not produce a result: {instance_id}")
    return (
        task_index,
        sample_index,
        {
            **result,
            "sample_index": sample_index,
            "temperature": temperature,
            "temperature_run_index": temperature_run_index,
            "seed": seed,
        },
    )


def _execute_rollouts(
    jobs: list[tuple[int, dict[str, Any], int, float, int]],
    *,
    args: argparse.Namespace,
    output_dir: Path,
    base_config: AgentForgeConfig,
    record_event: Callable[..., None],
    recovery_mode: bool,
) -> dict[tuple[int, int], dict[str, Any]]:
    results: dict[tuple[int, int], dict[str, Any]] = {}
    progress = tqdm(
        total=len(jobs),
        desc=(
            f"SWE-smith recovery {args.recovery_replica_index + 1}/{args.recovery_replicas}"
            if recovery_mode
            else "SWE-smith trajectories"
        ),
        unit="rollout",
        disable=not args.progress,
    )

    def run(job: tuple[int, dict[str, Any], int, float, int]) -> tuple[int, int, dict[str, Any]]:
        return _run_rollout(
            job,
            args=args,
            output_dir=output_dir,
            base_config=base_config,
            record_event=record_event,
        )

    def save(result_tuple: tuple[int, int, dict[str, Any]]) -> None:
        task_index, sample_index, result = result_tuple
        results[(task_index, sample_index)] = result
        progress.set_postfix(
            instance=result["instance_id"],
            sample=sample_index,
            temperature=result["temperature"],
        )
        progress.update(1)

    with progress:
        if args.rollout_workers == 1:
            for job in jobs:
                save(run(job))
        else:
            with ThreadPoolExecutor(max_workers=args.rollout_workers) as pool:
                futures = [pool.submit(run, job) for job in jobs]
                try:
                    for future in as_completed(futures):
                        save(future.result())
                except BaseException as exc:
                    print(
                        f"Collector executor stopping after {type(exc).__name__}: {exc}; "
                        f"active rollout subprocesses={active_subprocess_count()}",
                        file=sys.stderr,
                        flush=True,
                    )
                    for future in futures:
                        future.cancel()
                    terminate_active_subprocesses()
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise

    if len(results) != len(jobs):
        raise RuntimeError("Collection ended before every SWE-smith rollout produced a result")
    return results


def _result_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "n_completed": sum(result.get("status") in {"completed", "mocked"} for result in results),
        "n_model_terminated": sum(result.get("status") == "model_terminated" for result in results),
        "n_finished": sum(result.get("status") in FINISHED_ROLLOUT_STATUSES for result in results),
        "n_errors": sum(result.get("status") == "error" for result in results),
        "n_with_patch": sum(bool(result.get("patch")) for result in results),
    }


def collect_swesmith(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = ensure_dir(args.output_dir)
    recovery_mode = _validate_collection_args(args)
    dataset_revision = None if Path(args.dataset).is_file() else args.dataset_revision
    tasks = _select_collection_tasks(args, dataset_revision)
    temperatures = parse_temperatures(args.temperatures)
    schedule = temperature_schedule(temperatures, args.runs_per_temperature)
    base_config = _base_agent_config(args, dataset_revision=dataset_revision)
    manifest, run_at = _collection_manifest(
        args,
        tasks,
        temperatures,
        len(schedule),
        dataset_revision,
    )
    _prepare_collection_manifest(args, output_dir, manifest, recovery_mode, run_at)
    jobs = _rollout_jobs(tasks, schedule, args, recovery_mode)
    results = _execute_rollouts(
        jobs,
        args=args,
        output_dir=output_dir,
        base_config=base_config,
        record_event=_rollout_event_recorder(args, output_dir, recovery_mode, run_at),
        recovery_mode=recovery_mode,
    )

    if recovery_mode:
        recovery_results = list(results.values())
        counts = _result_counts(recovery_results)
        recovery_summary = {
            "schema_version": 1,
            "created_at": run_at,
            "mode": "shard_recovery_replica",
            "recovery_run_id": args.recovery_run_id,
            "recovery_replicas": args.recovery_replicas,
            "recovery_replica_index": args.recovery_replica_index,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "n_assigned_rollouts": len(jobs),
            "n_finished": counts["n_finished"],
            "n_errors": counts["n_errors"],
        }
        recovery_summary_path = output_dir / (
            f"recovery-{args.recovery_run_id}-replica-{args.recovery_replica_index}.json"
        )
        write_json(recovery_summary_path, recovery_summary)
        print(json.dumps(recovery_summary, indent=2))
        if args.require_complete and recovery_summary["n_finished"] != len(recovery_results):
            raise RuntimeError(
                "SWE-smith recovery replica did not finish every assigned rollout: "
                f"{recovery_summary['n_finished']}/{len(recovery_results)} finished, "
                f"{recovery_summary['n_errors']} errors"
            )
        return recovery_summary

    all_results: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []
    for sample_index, (temperature, temperature_run_index) in enumerate(schedule):
        sample_dir = ensure_dir(_sample_dir(output_dir, sample_index))
        sample_results = [results[(task_index, sample_index)] for task_index in range(len(tasks))]
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
            **_result_counts(sample_results),
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
        **_result_counts(all_results),
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
    parser.add_argument("--recovery-replicas", type=int, default=1)
    parser.add_argument("--recovery-replica-index", type=int)
    parser.add_argument("--recovery-run-id")
    parser.add_argument(
        "--max-steps", type=int, default=int(os.getenv("MAX_STEPS", DEFAULT_MAX_STEPS))
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=int(os.getenv("CONTEXT_LENGTH", DEFAULT_SWESMITH_CONTEXT_LENGTH)),
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
    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    previous_handlers = {signum: signal.getsignal(signum) for signum in handled_signals}

    def handle_termination(signum: int, _frame: object) -> None:
        for handled_signal in handled_signals:
            signal.signal(handled_signal, signal.SIG_IGN)
        signal_name = signal.Signals(signum).name
        try:
            print(
                f"Collector received {signal_name} ({signum}); "
                f"active rollout subprocesses={active_subprocess_count()}; cleaning up",
                file=sys.stderr,
                flush=True,
            )
        finally:
            terminate_active_subprocesses()
        raise SystemExit(128 + signum)

    for signum in handled_signals:
        signal.signal(signum, handle_termination)
    try:
        collect_swesmith(args)
        return 0
    finally:
        terminate_active_subprocesses()
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
