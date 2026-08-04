"""Create reproducible, repository-disjoint SWE-smith train/validation ID files."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from debug_depo.constants import (
    DEFAULT_SWESMITH_DATASET,
    DEFAULT_SWESMITH_DATASET_REVISION,
    DEFAULT_SWESMITH_SPLIT,
)
from debug_depo.data import load_swebench_tasks, read_instance_ids_file
from debug_depo.utils import ensure_dir, read_json, write_json


DEFAULT_VALIDATION_FRACTION = 0.10
DEFAULT_SPLIT_SEED = 42
DEFAULT_TRAJECTORY_SUBSET_SIZE = 5_000
DEFAULT_VALIDATION_SUBSET_SIZE = 500
DEFAULT_VALIDATION_SCREENING_SIZES = (100, 200)
SPLIT_MANIFEST_SCHEMA_VERSION = 4
TRAJECTORY_SUBSET_FILENAME = "swesmith_train_5000_instance_ids.txt"
VALIDATION_SUBSET_FILENAME = "swesmith_validation_500_instance_ids.txt"
CACHE_SUBSET_FILENAME = "swesmith_cache_5500_instance_ids.txt"


def _ordered_ids_sha256(instance_ids: list[str]) -> str:
    payload = "".join(f"{instance_id}\n" for instance_id in instance_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _repository_snapshot(instance_id: str) -> str:
    repository, separator, _mutation = instance_id.rpartition(".")
    if not separator or not repository:
        raise ValueError(
            f"SWE-smith instance IDs must end with a dot-separated mutation key: {instance_id!r}"
        )
    return repository


def _normalize_repository_selector(selector: str) -> str:
    normalized = selector.strip().removeprefix("swesmith/")
    if not normalized:
        raise ValueError("Excluded repository selectors must not be empty")
    return normalized


def _matches_repository_selector(repository: str, selector: str) -> bool:
    """Match either one snapshot or every snapshot of an owner/repository."""

    return repository == selector or repository.startswith(f"{selector}.")


def repository_covering_subset(
    instance_ids: list[str],
    *,
    size: int,
    seed: int,
    namespace: str,
) -> list[str]:
    """Select a proportional, deterministic subset covering every repository."""

    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("Source instance IDs must be unique")
    if not 1 <= size <= len(instance_ids):
        raise ValueError(f"Subset size must be between 1 and {len(instance_ids)}, got {size}")

    by_repository: dict[str, list[str]] = defaultdict(list)
    for instance_id in instance_ids:
        by_repository[_repository_snapshot(instance_id)].append(instance_id)
    if size < len(by_repository):
        raise ValueError(f"Subset size {size} cannot cover {len(by_repository)} repositories")

    def rank(instance_id: str, stage: str) -> str:
        return hashlib.sha256(
            f"{seed}\0{namespace}\0{stage}\0{instance_id}".encode("utf-8")
        ).hexdigest()

    # Reserve one slot per repository, then award each remaining slot to the
    # repository furthest below its proportional ideal. Integer arithmetic
    # keeps the allocation stable across Python versions.
    repository_names = sorted(by_repository)
    quotas = {repository: 1 for repository in repository_names}
    slots_remaining = size - len(repository_names)
    total_size = len(instance_ids)
    allocation_heap: list[tuple[int, str, str]] = []
    for repository in repository_names:
        if len(by_repository[repository]) == 1:
            continue
        deficit = size * len(by_repository[repository]) - total_size
        tie_breaker = hashlib.sha256(
            f"{seed}\0{namespace}\0quota\0{repository}".encode("utf-8")
        ).hexdigest()
        heapq.heappush(
            allocation_heap,
            (-deficit, tie_breaker, repository),
        )
    for _ in range(slots_remaining):
        _negative_deficit, tie_breaker, repository = heapq.heappop(allocation_heap)
        quotas[repository] += 1
        if quotas[repository] < len(by_repository[repository]):
            deficit = size * len(by_repository[repository]) - quotas[repository] * total_size
            heapq.heappush(
                allocation_heap,
                (-deficit, tie_breaker, repository),
            )

    selected = {
        instance_id
        for repository, repository_ids in by_repository.items()
        for instance_id in sorted(
            repository_ids,
            key=lambda item: rank(item, "sample"),
        )[: quotas[repository]]
    }
    return sorted(selected, key=lambda item: rank(item, "order"))


def write_task_subsets(
    *,
    train_ids: list[str],
    validation_ids: list[str],
    output_dir: str | Path,
    trajectory_subset_size: int = DEFAULT_TRAJECTORY_SUBSET_SIZE,
    validation_subset_size: int = DEFAULT_VALIDATION_SUBSET_SIZE,
    validation_screening_sizes: tuple[int, ...] | None = None,
    excluded_validation_repositories: tuple[str, ...] = (),
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    """Write training, validation, and cache-union task-ID files."""

    if set(train_ids) & set(validation_ids):
        raise ValueError("Source train and validation instance IDs must be disjoint")
    train_repositories = {_repository_snapshot(item) for item in train_ids}
    validation_repositories = {_repository_snapshot(item) for item in validation_ids}
    if train_repositories & validation_repositories:
        raise ValueError("Source train and validation memberships must be repository-disjoint")

    exclusion_selectors = tuple(
        dict.fromkeys(
            _normalize_repository_selector(selector)
            for selector in excluded_validation_repositories
        )
    )
    matched_excluded_repositories = sorted(
        repository
        for repository in validation_repositories
        if any(
            _matches_repository_selector(repository, selector)
            for selector in exclusion_selectors
        )
    )
    unmatched_selectors = [
        selector
        for selector in exclusion_selectors
        if not any(
            _matches_repository_selector(repository, selector)
            for repository in validation_repositories
        )
    ]
    if unmatched_selectors:
        raise ValueError(
            "Excluded validation repositories did not match the validation parent: "
            + ", ".join(unmatched_selectors)
        )
    matched_excluded_repository_set = set(matched_excluded_repositories)
    eligible_validation_ids = [
        instance_id
        for instance_id in validation_ids
        if _repository_snapshot(instance_id) not in matched_excluded_repository_set
    ]
    if not eligible_validation_ids:
        raise ValueError("Repository exclusions removed every validation instance")

    trajectory_ids = repository_covering_subset(
        train_ids,
        size=trajectory_subset_size,
        seed=seed,
        namespace="trajectory",
    )
    selected_validation_ids = repository_covering_subset(
        eligible_validation_ids,
        size=validation_subset_size,
        seed=seed,
        namespace="validation",
    )
    if validation_screening_sizes is None:
        validation_screening_sizes = (
            DEFAULT_VALIDATION_SCREENING_SIZES
            if validation_subset_size == DEFAULT_VALIDATION_SUBSET_SIZE
            else ()
        )
    screening_sizes = sorted(set(validation_screening_sizes))
    if any(not 1 <= size < validation_subset_size for size in screening_sizes):
        raise ValueError(
            "Validation screening sizes must be unique positive budgets smaller "
            "than the main validation subset"
        )

    screening_subsets: dict[int, list[str]] = {}
    previous_ids: set[str] = set()
    for size in screening_sizes:
        screening_ids = repository_covering_subset(
            eligible_validation_ids,
            size=size,
            seed=seed,
            namespace="validation",
        )
        screening_id_set = set(screening_ids)
        if previous_ids and not previous_ids <= screening_id_set:
            raise ValueError(
                f"Validation screening memberships are not nested at budget {size}"
            )
        if not screening_id_set <= set(selected_validation_ids):
            raise ValueError(
                f"Validation screening budget {size} is not contained in the "
                f"{validation_subset_size}-task validation subset"
            )
        screening_subsets[size] = screening_ids
        previous_ids = screening_id_set
    cache_ids = trajectory_ids + selected_validation_ids

    output_root = ensure_dir(output_dir)
    trajectory_filename = (
        TRAJECTORY_SUBSET_FILENAME
        if trajectory_subset_size == DEFAULT_TRAJECTORY_SUBSET_SIZE
        else f"swesmith_train_{trajectory_subset_size}_instance_ids.txt"
    )
    validation_filename = (
        VALIDATION_SUBSET_FILENAME
        if validation_subset_size == DEFAULT_VALIDATION_SUBSET_SIZE
        else f"swesmith_validation_{validation_subset_size}_instance_ids.txt"
    )
    cache_filename = (
        CACHE_SUBSET_FILENAME
        if (
            trajectory_subset_size == DEFAULT_TRAJECTORY_SUBSET_SIZE
            and validation_subset_size == DEFAULT_VALIDATION_SUBSET_SIZE
        )
        else f"swesmith_cache_{len(cache_ids)}_instance_ids.txt"
    )
    trajectory_path = output_root / trajectory_filename
    validation_path = output_root / validation_filename
    cache_path = output_root / cache_filename
    for path, instance_ids in (
        (trajectory_path, trajectory_ids),
        (validation_path, selected_validation_ids),
        (cache_path, cache_ids),
    ):
        path.write_text(
            "".join(f"{instance_id}\n" for instance_id in instance_ids),
            encoding="utf-8",
        )

    screening_metadata: dict[str, dict[str, Any]] = {}
    for size, screening_ids in screening_subsets.items():
        filename = f"swesmith_validation_{size}_instance_ids.txt"
        path = output_root / filename
        path.write_text(
            "".join(f"{instance_id}\n" for instance_id in screening_ids),
            encoding="utf-8",
        )
        screening_metadata[str(size)] = {
            "source": "validation_instance_ids.txt",
            "file": str(path),
            "n_tasks": len(screening_ids),
            "n_repositories": len(
                {_repository_snapshot(item) for item in screening_ids}
            ),
            "sha256": _ordered_ids_sha256(screening_ids),
        }

    return {
        "strategy": "repository_covering_proportional_hash_sample",
        "strategy_version": 1,
        "group_field": "instance_id_without_final_dot_component",
        "allocation": ("one_per_repository_then_greatest_proportional_deficit"),
        "selection": "sha256_hash_rank_without_replacement",
        "ordering": "sha256_hash_order",
        "seed": seed,
        "validation_exclusions": {
            "selectors": list(exclusion_selectors),
            "matched_repositories": matched_excluded_repositories,
            "n_excluded_tasks": len(validation_ids) - len(eligible_validation_ids),
            "n_eligible_tasks": len(eligible_validation_ids),
        },
        "trajectory": {
            "source": "train_instance_ids.txt",
            "file": str(trajectory_path),
            "n_tasks": len(trajectory_ids),
            "n_repositories": len({_repository_snapshot(item) for item in trajectory_ids}),
            "sha256": _ordered_ids_sha256(trajectory_ids),
        },
        "validation": {
            "source": "validation_instance_ids.txt",
            "file": str(validation_path),
            "n_tasks": len(selected_validation_ids),
            "n_repositories": len({_repository_snapshot(item) for item in selected_validation_ids}),
            "sha256": _ordered_ids_sha256(selected_validation_ids),
        },
        "validation_screening": screening_metadata,
        "cache": {
            "sources": [
                trajectory_filename,
                validation_filename,
            ],
            "file": str(cache_path),
            "n_tasks": len(cache_ids),
            "n_repositories": len({_repository_snapshot(item) for item in cache_ids}),
            "sha256": _ordered_ids_sha256(cache_ids),
        },
    }


def repository_disjoint_split(
    tasks: list[dict[str, Any]],
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[str], list[str], list[str]]:
    """Split task IDs near the requested ratio without sharing repositories."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not tasks:
        raise ValueError("Cannot split an empty dataset")

    instance_ids = [str(task.get("instance_id") or "") for task in tasks]
    repositories = [str(task.get("repo") or "") for task in tasks]
    if any(not instance_id for instance_id in instance_ids):
        raise ValueError("Every task must contain instance_id")
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("Task instance IDs must be unique")
    if any(not repository for repository in repositories):
        raise ValueError("Every task must contain repo")

    repository_counts = Counter(repositories)
    if len(repository_counts) < 2:
        raise ValueError("At least two repositories are required for a disjoint split")

    target = round(len(tasks) * validation_fraction)
    ranked_repositories = sorted(
        repository_counts,
        key=lambda repository: hashlib.sha256(f"{seed}\0{repository}".encode("utf-8")).hexdigest(),
    )
    validation_repositories: list[str] = []
    validation_count = 0
    for repository in ranked_repositories:
        candidate_count = validation_count + repository_counts[repository]
        if abs(candidate_count - target) < abs(validation_count - target):
            validation_repositories.append(repository)
            validation_count = candidate_count

    if not validation_repositories:
        validation_repositories.append(ranked_repositories[0])
    if len(validation_repositories) == len(repository_counts):
        validation_repositories.pop()

    validation_repository_set = set(validation_repositories)
    train_ids = [
        instance_id
        for instance_id, repository in zip(instance_ids, repositories)
        if repository not in validation_repository_set
    ]
    validation_ids = [
        instance_id
        for instance_id, repository in zip(instance_ids, repositories)
        if repository in validation_repository_set
    ]
    return train_ids, validation_ids, sorted(validation_repositories)


def prepare_swesmith_splits(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_swebench_tasks(
        args.dataset,
        args.source_split,
        revision=args.dataset_revision,
    )
    train_ids, validation_ids, validation_repositories = repository_disjoint_split(
        tasks,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    output_dir = ensure_dir(args.output_dir)
    train_path = output_dir / "train_instance_ids.txt"
    validation_path = output_dir / "validation_instance_ids.txt"
    manifest_path = output_dir / "swesmith_py_split_manifest.json"
    train_path.write_text("".join(f"{item}\n" for item in train_ids), encoding="utf-8")
    validation_path.write_text(
        "".join(f"{item}\n" for item in validation_ids),
        encoding="utf-8",
    )
    task_subsets = write_task_subsets(
        train_ids=train_ids,
        validation_ids=validation_ids,
        output_dir=output_dir,
        trajectory_subset_size=getattr(
            args,
            "trajectory_subset_size",
            DEFAULT_TRAJECTORY_SUBSET_SIZE,
        ),
        validation_subset_size=getattr(
            args,
            "validation_subset_size",
            DEFAULT_VALIDATION_SUBSET_SIZE,
        ),
        excluded_validation_repositories=tuple(
            getattr(args, "exclude_repository", ())
        ),
        seed=args.seed,
    )

    all_ids = [str(task["instance_id"]) for task in tasks]
    train_id_set = set(train_ids)
    train_repositories = {
        str(task["repo"]) for task in tasks if str(task["instance_id"]) in train_id_set
    }
    manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "source_split": args.source_split,
        "strategy": "repository_disjoint_hash_ranked_greedy",
        "group_field": "repo",
        "seed": args.seed,
        "requested_validation_fraction": args.validation_fraction,
        "actual_validation_fraction": len(validation_ids) / len(tasks),
        "n_total": len(tasks),
        "n_train": len(train_ids),
        "n_validation": len(validation_ids),
        "n_train_repositories": len(train_repositories),
        "n_validation_repositories": len(validation_repositories),
        "validation_repositories": validation_repositories,
        "dataset_instance_ids_sha256": _ordered_ids_sha256(all_ids),
        "train_instance_ids_sha256": _ordered_ids_sha256(train_ids),
        "validation_instance_ids_sha256": _ordered_ids_sha256(validation_ids),
        "train_instance_ids_file": str(train_path),
        "validation_instance_ids_file": str(validation_path),
        "task_subsets": task_subsets,
    }
    write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic, repository-disjoint SWE-smith train and "
            "validation instance-ID files."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_SWESMITH_DATASET)
    parser.add_argument(
        "--dataset-revision",
        default=DEFAULT_SWESMITH_DATASET_REVISION,
    )
    parser.add_argument("--source-split", default=DEFAULT_SWESMITH_SPLIT)
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--trajectory-subset-size",
        type=int,
        default=DEFAULT_TRAJECTORY_SUBSET_SIZE,
    )
    parser.add_argument(
        "--validation-subset-size",
        type=int,
        default=DEFAULT_VALIDATION_SUBSET_SIZE,
    )
    parser.add_argument(
        "--subsets-only",
        action="store_true",
        help=(
            "Regenerate the curated subset files from existing train/validation "
            "instance-ID files without loading the Hugging Face dataset."
        ),
    )
    parser.add_argument(
        "--train-instance-ids-file",
        default="data/splits/train_instance_ids.txt",
    )
    parser.add_argument(
        "--validation-instance-ids-file",
        default="data/splits/validation_instance_ids.txt",
    )
    parser.add_argument(
        "--exclude-repository",
        action="append",
        default=[],
        metavar="REPOSITORY",
        help=(
            "Exclude a repository from derived validation subsets. Repeat for "
            "multiple repositories. Accepts an exact owner__repo.commit snapshot "
            "or owner__repo to exclude all of its snapshots; an optional "
            "'swesmith/' prefix is ignored."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.subsets_only:
        manifest = write_task_subsets(
            train_ids=read_instance_ids_file(args.train_instance_ids_file),
            validation_ids=read_instance_ids_file(args.validation_instance_ids_file),
            output_dir=args.output_dir,
            trajectory_subset_size=args.trajectory_subset_size,
            validation_subset_size=args.validation_subset_size,
            excluded_validation_repositories=tuple(args.exclude_repository),
            seed=args.seed,
        )
        manifest_path = Path(args.output_dir) / "swesmith_py_split_manifest.json"
        if manifest_path.is_file():
            split_manifest = read_json(manifest_path)
            if not isinstance(split_manifest, dict):
                raise ValueError(f"Invalid split manifest: {manifest_path}")
            split_manifest["schema_version"] = SPLIT_MANIFEST_SCHEMA_VERSION
            split_manifest["task_subsets"] = manifest
            write_json(manifest_path, split_manifest)
    else:
        manifest = prepare_swesmith_splits(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
