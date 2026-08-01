import argparse

import pytest

import debug_depo.prepare_swesmith_splits as split_module
from debug_depo.prepare_swesmith_splits import (
    prepare_swesmith_splits,
    repository_covering_subset,
    repository_disjoint_split,
    write_task_subsets,
)
from debug_depo.utils import read_json


def task(instance_id, repo):
    return {"instance_id": instance_id, "repo": repo}


def test_repository_disjoint_split_is_stable_and_complete():
    tasks = [
        task("a-1", "repo-a"),
        task("a-2", "repo-a"),
        task("b-1", "repo-b"),
        task("b-2", "repo-b"),
        task("c-1", "repo-c"),
        task("d-1", "repo-d"),
    ]

    first = repository_disjoint_split(tasks, validation_fraction=0.33, seed=42)
    second = repository_disjoint_split(tasks, validation_fraction=0.33, seed=42)
    train_ids, validation_ids, validation_repositories = first

    assert first == second
    assert set(train_ids).isdisjoint(validation_ids)
    assert set(train_ids) | set(validation_ids) == {task["instance_id"] for task in tasks}
    train_repositories = {task["repo"] for task in tasks if task["instance_id"] in train_ids}
    assert train_repositories.isdisjoint(validation_repositories)


def test_repository_disjoint_split_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        repository_disjoint_split([task("duplicate", "repo-a"), task("duplicate", "repo-b")])


def test_prepare_swesmith_splits_writes_ids_and_manifest(tmp_path, monkeypatch):
    tasks = [
        task("owner__a.aaaa1111.mutation-1", "repo-a"),
        task("owner__a.aaaa1111.mutation-2", "repo-a"),
        task("owner__b.bbbb2222.mutation-1", "repo-b"),
        task("owner__c.cccc3333.mutation-1", "repo-c"),
    ]
    monkeypatch.setattr(
        split_module,
        "load_swebench_tasks",
        lambda *_args, **_kwargs: tasks,
    )
    args = argparse.Namespace(
        dataset="dataset",
        dataset_revision="revision",
        source_split="train",
        output_dir=str(tmp_path),
        validation_fraction=0.25,
        seed=42,
        trajectory_subset_size=2,
        validation_subset_size=1,
    )

    manifest = prepare_swesmith_splits(args)

    train_ids = (tmp_path / "train_instance_ids.txt").read_text().splitlines()
    validation_ids = (tmp_path / "validation_instance_ids.txt").read_text().splitlines()
    assert set(train_ids).isdisjoint(validation_ids)
    assert set(train_ids) | set(validation_ids) == {task["instance_id"] for task in tasks}
    assert manifest == read_json(tmp_path / "swesmith_py_split_manifest.json")
    assert manifest["dataset_revision"] == "revision"
    assert manifest["schema_version"] == 3
    assert manifest["task_subsets"]["trajectory"]["n_tasks"] == 2
    assert manifest["task_subsets"]["validation"]["n_tasks"] == 1
    assert manifest["task_subsets"]["cache"]["n_tasks"] == 3


def _snapshot_ids(name, count):
    return [f"owner__{name}.{name * 4}.mutation-{index}" for index in range(count)]


def test_repository_covering_subset_is_stable_proportional_and_covering():
    source_ids = _snapshot_ids("a", 10) + _snapshot_ids("b", 5) + _snapshot_ids("c", 1)

    first = repository_covering_subset(
        source_ids,
        size=8,
        seed=42,
        namespace="test",
    )
    second = repository_covering_subset(
        list(reversed(source_ids)),
        size=8,
        seed=42,
        namespace="test",
    )

    assert first == second
    assert len(first) == len(set(first)) == 8
    assert set(first) <= set(source_ids)
    assert {item.rpartition(".")[0] for item in first} == {
        "owner__a.aaaa",
        "owner__b.bbbb",
        "owner__c.cccc",
    }
    counts = {
        repository: sum(item.startswith(repository) for item in first)
        for repository in ("owner__a.aaaa", "owner__b.bbbb", "owner__c.cccc")
    }
    assert counts == {
        "owner__a.aaaa": 5,
        "owner__b.bbbb": 2,
        "owner__c.cccc": 1,
    }


def test_repository_covering_subset_rejects_too_few_slots():
    source_ids = _snapshot_ids("a", 2) + _snapshot_ids("b", 2) + _snapshot_ids("c", 2)

    with pytest.raises(ValueError, match="cannot cover 3 repositories"):
        repository_covering_subset(
            source_ids,
            size=2,
            seed=42,
            namespace="test",
        )


def test_write_task_subsets_keeps_parent_membership_and_cache_union(tmp_path):
    train_ids = _snapshot_ids("a", 6) + _snapshot_ids("b", 3) + _snapshot_ids("c", 1)
    validation_ids = _snapshot_ids("d", 4) + _snapshot_ids("e", 2)

    metadata = write_task_subsets(
        train_ids=train_ids,
        validation_ids=validation_ids,
        output_dir=tmp_path,
        trajectory_subset_size=6,
        validation_subset_size=3,
        seed=42,
    )

    trajectory_ids = (tmp_path / "swesmith_train_6_instance_ids.txt").read_text().splitlines()
    selected_validation_ids = (
        (tmp_path / "swesmith_validation_3_instance_ids.txt").read_text().splitlines()
    )
    cache_ids = (tmp_path / "swesmith_cache_9_instance_ids.txt").read_text().splitlines()

    assert len(trajectory_ids) == 6
    assert len(selected_validation_ids) == 3
    assert set(trajectory_ids) <= set(train_ids)
    assert set(selected_validation_ids) <= set(validation_ids)
    assert set(trajectory_ids).isdisjoint(selected_validation_ids)
    assert cache_ids == trajectory_ids + selected_validation_ids
    assert metadata["trajectory"]["n_repositories"] == 3
    assert metadata["validation"]["n_repositories"] == 2
    assert metadata["cache"]["n_tasks"] == 9


def test_default_validation_screening_subsets_are_nested_and_recorded(tmp_path):
    train_ids = _snapshot_ids("a", 5_000)
    validation_ids = _snapshot_ids("d", 500)

    metadata = write_task_subsets(
        train_ids=train_ids,
        validation_ids=validation_ids,
        output_dir=tmp_path,
    )

    memberships = {
        size: set(
            (tmp_path / f"swesmith_validation_{size}_instance_ids.txt")
            .read_text()
            .splitlines()
        )
        for size in (100, 200, 500)
    }
    assert memberships[100] <= memberships[200] <= memberships[500]
    assert metadata["validation_screening"]["100"]["n_tasks"] == 100
    assert metadata["validation_screening"]["200"]["n_tasks"] == 200
    assert len(metadata["validation_screening"]["100"]["sha256"]) == 64
