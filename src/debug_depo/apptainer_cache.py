"""Persistent, process-safe Apptainer SIF image caching and prebuilding."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from tqdm.auto import tqdm

from debug_depo.constants import (
    DEFAULT_SWEBENCH_DATASET,
    DEFAULT_SWEBENCH_DATASET_REVISION,
    DEFAULT_SWEBENCH_SPLIT,
    DEFAULT_SWESMITH_DATASET,
    DEFAULT_SWESMITH_DATASET_REVISION,
    DEFAULT_SWESMITH_SPLIT,
    TARGET_VERIFIED_TOTAL,
)
from debug_depo.data import (
    load_swebench_tasks,
    read_instance_ids_file,
    resolve_swebench_dataset_revision,
    select_tasks,
)
from debug_depo.utils import ensure_dir, slugify, write_json

APPTAINER_IMAGE_SCHEMES = ("docker://", "oras://", "library://")
DEFAULT_SWEBENCH_IMAGE_TEMPLATE = (
    "docker://ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest"
)
_IMAGE_TEMPLATE_TAG_IN_FIELD = re.compile(
    r"\{(instance_id|instance_id_lower|image_key):([^{}]+)\}"
)


@dataclass(frozen=True)
class CacheImage:
    """One unique image required by one or more selected tasks."""

    family: str
    image_uri: str
    sif_path: Path
    instance_ids: tuple[str, ...]


def image_uri(image_name: str) -> str:
    """Return an Apptainer-compatible URI for an OCI image name."""

    if image_name.startswith(APPTAINER_IMAGE_SCHEMES):
        return image_name
    return f"docker://{image_name}"


def normalize_image_template(template: str) -> str:
    """Accept the common ``{instance_id:tag}`` Docker-template typo."""

    return _IMAGE_TEMPLATE_TAG_IN_FIELD.sub(r"{\1}:\2", template)


def image_uri_from_template(
    template: str,
    instance_id: str,
    image_key: str = "",
) -> str:
    """Render a SWE-bench image URI from a task and an image template."""

    normalized_template = normalize_image_template(template)
    try:
        return image_uri(
            normalized_template.format(
                instance_id=instance_id,
                instance_id_lower=instance_id.lower(),
                image_key=image_key,
            )
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "Invalid Apptainer image template "
            f"{template!r}. Use placeholders like '{{instance_id}}' and put Docker "
            "tags after the closing brace, e.g. "
            "'docker://...swe-bench.eval.x86_64.{instance_id}:latest'."
        ) from exc


def sif_path_for_image(sif_dir: str | Path, image_name: str) -> Path:
    """Return the stable shared SIF path for an image name or URI."""

    return Path(sif_dir) / f"{slugify(image_uri(image_name))}.sif"


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold an advisory lock that is visible to other processes and PBS jobs."""

    ensure_dir(path.parent)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _temporary_sif_path(sif_path: Path) -> Path:
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    return sif_path.with_name(f".{sif_path.stem}.{token}.sif")


def pull_sif_if_missing(
    *,
    sif_path: str | Path,
    image_uri: str,
    cache_dir: str | Path | None,
    executable: str = "apptainer",
    dry_run: bool = False,
    retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Pull one SIF atomically and reuse it across processes.

    The returned command names the final SIF path for provenance. The actual
    pull writes a unique temporary SIF, which is atomically renamed only after
    Apptainer exits successfully.
    """

    destination = Path(sif_path)
    command = [executable, "pull", str(destination), image_uri]
    if destination.is_file() or dry_run:
        return command
    if retries < 1:
        raise ValueError("retries must be at least 1")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds must be non-negative")

    ensure_dir(destination.parent)
    lock_path = destination.with_name(f"{destination.name}.lock")
    with _exclusive_file_lock(lock_path):
        if destination.is_file():
            return command

        pull_env = dict(os.environ if env is None else env)
        if cache_dir:
            pull_env["APPTAINER_CACHEDIR"] = str(ensure_dir(cache_dir))

        for attempt in range(retries):
            temporary = _temporary_sif_path(destination)
            pull_command = [executable, "pull", str(temporary), image_uri]
            try:
                subprocess.run(pull_command, env=pull_env, check=True)
                if not temporary.is_file():
                    raise RuntimeError(
                        f"Apptainer reported success without creating {temporary}"
                    )
                os.replace(temporary, destination)
                break
            except (OSError, RuntimeError, subprocess.CalledProcessError):
                temporary.unlink(missing_ok=True)
                if attempt == retries - 1:
                    raise
                time.sleep(retry_backoff_seconds * (2**attempt))

    return command


def _group_cache_images(
    *,
    family: str,
    sif_dir: str | Path,
    task_images: list[tuple[str, str]],
) -> list[CacheImage]:
    instance_ids_by_uri: dict[str, list[str]] = {}
    for instance_id, image_name in task_images:
        uri = image_uri(image_name)
        instance_ids_by_uri.setdefault(uri, []).append(instance_id)
    return [
        CacheImage(
            family=family,
            image_uri=uri,
            sif_path=sif_path_for_image(sif_dir, uri),
            instance_ids=tuple(instance_ids),
        )
        for uri, instance_ids in instance_ids_by_uri.items()
    ]


def swebench_cache_images(
    tasks: list[dict[str, Any]],
    *,
    sif_dir: str | Path,
    image_template: str = DEFAULT_SWEBENCH_IMAGE_TEMPLATE,
) -> list[CacheImage]:
    """Return the per-instance images used by Verified collection/evaluation."""

    task_images = []
    for task in tasks:
        instance_id = str(task.get("instance_id") or "")
        if not instance_id:
            raise ValueError("Every SWE-bench task must contain instance_id")
        task_images.append(
            (
                instance_id,
                image_uri_from_template(
                    image_template,
                    instance_id,
                    image_key=str(task.get("image_name") or ""),
                ),
            )
        )
    return _group_cache_images(
        family="swebench",
        sif_dir=sif_dir,
        task_images=task_images,
    )


def _swesmith_image_name(task: dict[str, Any]) -> str:
    image_name = task.get("image_name")
    if isinstance(image_name, str) and image_name.strip():
        return image_name.strip()
    try:
        from swesmith.profiles import registry
    except ImportError as exc:
        raise RuntimeError(
            "A SWE-smith task has no image_name and SWE-smith is not installed. "
            "Run scripts/install_swesmith.sh before prebuilding this cache."
        ) from exc
    return str(registry.get_from_inst(task).image_name)


def swesmith_cache_images(
    tasks: list[dict[str, Any]],
    *,
    sif_dir: str | Path,
) -> list[CacheImage]:
    """Return deduplicated repository images for selected SWE-smith tasks."""

    task_images = []
    for task in tasks:
        instance_id = str(task.get("instance_id") or "")
        if not instance_id:
            raise ValueError("Every SWE-smith task must contain instance_id")
        task_images.append((instance_id, _swesmith_image_name(task)))
    return _group_cache_images(
        family="swesmith",
        sif_dir=sif_dir,
        task_images=task_images,
    )


def _selected_cache_tasks(
    *,
    dataset: str,
    split: str,
    dataset_revision: str | None,
    instance_ids_file: str | Path | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    requested_ids = (
        read_instance_ids_file(instance_ids_file)
        if instance_ids_file
        else None
    )
    if instance_ids_file and not requested_ids:
        raise ValueError(f"No instance IDs found in {instance_ids_file}")
    if requested_ids and len(requested_ids) != len(set(requested_ids)):
        raise ValueError(f"Duplicate instance IDs in {instance_ids_file}")
    selected = select_tasks(
        load_swebench_tasks(dataset, split, revision=dataset_revision),
        instance_ids=requested_ids,
        limit=limit,
    )
    if not selected:
        raise ValueError(f"No tasks selected from {dataset} ({split})")
    return selected


def _build_one_image(
    image: CacheImage,
    *,
    cache_dir: str | Path | None,
    executable: str,
    dry_run: bool,
    retries: int,
    retry_backoff_seconds: float,
) -> dict[str, Any]:
    existed = image.sif_path.is_file()
    try:
        command = pull_sif_if_missing(
            sif_path=image.sif_path,
            image_uri=image.image_uri,
            cache_dir=cache_dir,
            executable=executable,
            dry_run=dry_run,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
    except Exception as exc:
        return {
            "family": image.family,
            "image_uri": image.image_uri,
            "sif_path": str(image.sif_path),
            "instance_id_examples": list(image.instance_ids[:10]),
            "n_tasks": len(image.instance_ids),
            "status": "failed",
            "error": repr(exc),
        }
    return {
        "family": image.family,
        "image_uri": image.image_uri,
        "sif_path": str(image.sif_path),
        "instance_id_examples": list(image.instance_ids[:10]),
        "n_tasks": len(image.instance_ids),
        "status": "dry_run" if dry_run else ("cached" if existed else "pulled"),
        "pull_command": command,
    }


def prebuild_apptainer_cache(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve selected task images, build missing SIFs, and return a summary."""

    if args.max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if args.mode == "smoke":
        swebench_limit = (
            args.swebench_limit if args.swebench_limit is not None else 1
        )
        swesmith_limit = (
            args.swesmith_limit if args.swesmith_limit is not None else 1
        )
    else:
        swebench_limit = args.swebench_limit
        swesmith_limit = args.swesmith_limit

    jobs: list[tuple[CacheImage, str | Path | None]] = []
    family_summary: dict[str, dict[str, Any]] = {}
    if args.datasets in {"both", "swebench"}:
        swebench_revision = resolve_swebench_dataset_revision(
            args.swebench_dataset,
            args.swebench_dataset_revision,
        )
        swebench_tasks = _selected_cache_tasks(
            dataset=args.swebench_dataset,
            split=args.swebench_split,
            dataset_revision=swebench_revision,
            instance_ids_file=args.swebench_task_ids_file,
            limit=swebench_limit,
        )
        if (
            args.mode == "full"
            and args.expected_swebench_tasks is not None
            and len(swebench_tasks) != args.expected_swebench_tasks
        ):
            raise ValueError(
                "Expected "
                f"{args.expected_swebench_tasks} SWE-bench tasks, "
                f"selected {len(swebench_tasks)}"
            )
        swebench_images = swebench_cache_images(
            swebench_tasks,
            sif_dir=args.swebench_sif_dir,
            image_template=args.swebench_image_template,
        )
        jobs.extend(
            (image, args.swebench_apptainer_cache_dir)
            for image in swebench_images
        )
        family_summary["swebench"] = {
            "dataset": args.swebench_dataset,
            "dataset_revision": swebench_revision,
            "split": args.swebench_split,
            "task_ids_file": args.swebench_task_ids_file,
            "selected_tasks": len(swebench_tasks),
            "unique_images": len(swebench_images),
            "sif_dir": str(args.swebench_sif_dir),
            "apptainer_cache_dir": args.swebench_apptainer_cache_dir,
            "image_template": args.swebench_image_template,
        }

    if args.datasets in {"both", "swesmith"}:
        swesmith_revision = (
            None
            if Path(args.swesmith_dataset).is_file()
            else args.swesmith_dataset_revision
        )
        swesmith_tasks = _selected_cache_tasks(
            dataset=args.swesmith_dataset,
            split=args.swesmith_split,
            dataset_revision=swesmith_revision,
            instance_ids_file=args.swesmith_task_ids_file,
            limit=swesmith_limit,
        )
        swesmith_images = swesmith_cache_images(
            swesmith_tasks,
            sif_dir=args.swesmith_sif_dir,
        )
        jobs.extend(
            (image, args.swesmith_apptainer_cache_dir)
            for image in swesmith_images
        )
        family_summary["swesmith"] = {
            "dataset": args.swesmith_dataset,
            "dataset_revision": swesmith_revision,
            "split": args.swesmith_split,
            "task_ids_file": args.swesmith_task_ids_file,
            "selected_tasks": len(swesmith_tasks),
            "unique_images": len(swesmith_images),
            "sif_dir": str(args.swesmith_sif_dir),
            "apptainer_cache_dir": args.swesmith_apptainer_cache_dir,
        }

    results: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(jobs),
        desc="Apptainer cache",
        unit="image",
        disable=not args.progress,
    )
    with progress:
        if args.max_workers == 1:
            for image, cache_dir in jobs:
                progress.set_postfix(family=image.family)
                results.append(
                    _build_one_image(
                        image,
                        cache_dir=cache_dir,
                        executable=args.executable,
                        dry_run=args.dry_run,
                        retries=args.retries,
                        retry_backoff_seconds=args.retry_backoff_seconds,
                    )
                )
                progress.update(1)
        else:
            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futures = {
                    pool.submit(
                        _build_one_image,
                        image,
                        cache_dir=cache_dir,
                        executable=args.executable,
                        dry_run=args.dry_run,
                        retries=args.retries,
                        retry_backoff_seconds=args.retry_backoff_seconds,
                    ): image
                    for image, cache_dir in jobs
                }
                for future in as_completed(futures):
                    image = futures[future]
                    progress.set_postfix(family=image.family)
                    results.append(future.result())
                    progress.update(1)

    results.sort(key=lambda row: (str(row["family"]), str(row["image_uri"])))
    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "datasets": args.datasets,
        "dry_run": bool(args.dry_run),
        "families": family_summary,
        "total_tasks": sum(
            int(item["selected_tasks"]) for item in family_summary.values()
        ),
        "total_unique_images": len(jobs),
        "status_counts": dict(sorted(status_counts.items())),
        "results": results,
    }
    if args.summary_output:
        write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prebuild persistent SWE-bench Verified and SWE-smith Apptainer SIFs."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default=os.getenv("CACHE_BUILD_MODE", "smoke"),
        help="Smoke defaults to one task per family; full uses the complete selections.",
    )
    parser.add_argument(
        "--datasets",
        choices=("both", "swebench", "swesmith"),
        default=os.getenv("CACHE_BUILD_DATASETS", "both"),
    )
    parser.add_argument("--swebench-dataset", default=DEFAULT_SWEBENCH_DATASET)
    parser.add_argument(
        "--swebench-dataset-revision",
        default=os.getenv(
            "SWEBENCH_DATASET_REVISION",
            DEFAULT_SWEBENCH_DATASET_REVISION,
        ),
    )
    parser.add_argument("--swebench-split", default=DEFAULT_SWEBENCH_SPLIT)
    parser.add_argument("--swebench-task-ids-file")
    parser.add_argument("--swebench-limit", type=int)
    parser.add_argument(
        "--expected-swebench-tasks",
        type=int,
        default=TARGET_VERIFIED_TOTAL,
    )
    parser.add_argument(
        "--swebench-image-template",
        default=os.getenv(
            "SWEBENCH_APPTAINER_IMAGE_TEMPLATE",
            DEFAULT_SWEBENCH_IMAGE_TEMPLATE,
        ),
    )
    parser.add_argument(
        "--swebench-sif-dir",
        default=os.getenv(
            "SWEBENCH_APPTAINER_SIF_DIR",
            "data/apptainer/swebench-sifs",
        ),
    )
    parser.add_argument(
        "--swebench-apptainer-cache-dir",
        default=os.getenv(
            "SWEBENCH_APPTAINER_CACHE_DIR",
            os.getenv("APPTAINER_CACHEDIR"),
        ),
    )
    parser.add_argument("--swesmith-dataset", default=DEFAULT_SWESMITH_DATASET)
    parser.add_argument(
        "--swesmith-dataset-revision",
        default=os.getenv(
            "SWESMITH_DATASET_REVISION",
            DEFAULT_SWESMITH_DATASET_REVISION,
        ),
    )
    parser.add_argument("--swesmith-split", default=DEFAULT_SWESMITH_SPLIT)
    parser.add_argument(
        "--swesmith-task-ids-file",
        default=os.getenv(
            "SWESMITH_TASK_IDS_FILE",
            "data/splits/train_instance_ids.txt",
        ),
    )
    parser.add_argument("--swesmith-limit", type=int)
    parser.add_argument(
        "--swesmith-sif-dir",
        default=os.getenv(
            "SWESMITH_APPTAINER_SIF_DIR",
            "data/apptainer/swesmith-sifs",
        ),
    )
    parser.add_argument(
        "--swesmith-apptainer-cache-dir",
        default=os.getenv(
            "SWESMITH_APPTAINER_CACHE_DIR",
            os.getenv("APPTAINER_CACHEDIR"),
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=os.getenv("CACHE_BUILD_SUMMARY_OUTPUT"),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("CACHE_BUILD_MAX_WORKERS", "4")),
    )
    parser.add_argument(
        "--executable",
        default=os.getenv("MSWEA_SINGULARITY_EXECUTABLE", "apptainer"),
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run and shutil.which(args.executable) is None:
        raise SystemExit(f"{args.executable!r} is not available on PATH")
    summary = prebuild_apptainer_cache(args)
    return 1 if summary["status_counts"].get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
