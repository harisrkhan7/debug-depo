import subprocess
import threading
import time
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import debug_depo.apptainer_cache as apptainer_cache
from debug_depo.apptainer_cache import (
    DEFAULT_SWEBENCH_IMAGE_TEMPLATE,
    build_parser,
    image_uri,
    image_uri_from_template,
    prebuild_apptainer_cache,
    pull_sif_if_missing,
    sif_path_for_image,
    swebench_cache_images,
    swesmith_cache_images,
)


def test_image_uri_and_sif_path_are_stable(tmp_path):
    image_name = "swebench/swesmith.x86_64.repo_1776_project:latest"

    assert image_uri(image_name) == f"docker://{image_name}"
    assert image_uri(f"oras://{image_name}") == f"oras://{image_name}"
    expected = (
        tmp_path / "docker_swebench_swesmith.x86_64.repo_1776_project_latest.sif"
    )
    assert sif_path_for_image(tmp_path, image_name) == expected
    assert sif_path_for_image(tmp_path, f"docker://{image_name}") == expected


def test_pull_is_atomic_and_reuses_the_finished_sif(tmp_path, monkeypatch):
    sif_path = tmp_path / "sifs/image.sif"
    cache_dir = tmp_path / "cache"
    calls = []

    def fake_run(command, *, env, check):
        calls.append((command, env, check))
        Path(command[2]).write_bytes(b"sif")

    monkeypatch.setattr(apptainer_cache.subprocess, "run", fake_run)

    command = pull_sif_if_missing(
        sif_path=sif_path,
        image_uri="docker://example/image:latest",
        cache_dir=cache_dir,
    )
    second_command = pull_sif_if_missing(
        sif_path=sif_path,
        image_uri="docker://example/image:latest",
        cache_dir=cache_dir,
    )

    expected = [
        "apptainer",
        "pull",
        str(sif_path),
        "docker://example/image:latest",
    ]
    assert command == expected
    assert second_command == expected
    assert sif_path.read_bytes() == b"sif"
    assert len(calls) == 1
    pull_command, pull_env, check = calls[0]
    assert pull_command[:2] == ["apptainer", "pull"]
    assert pull_command[2] != str(sif_path)
    assert pull_command[2].endswith(".sif")
    assert pull_env["APPTAINER_CACHEDIR"] == str(cache_dir)
    assert check is True


def test_process_lock_allows_only_one_concurrent_pull(tmp_path, monkeypatch):
    sif_path = tmp_path / "image.sif"
    calls = 0
    calls_guard = threading.Lock()

    def fake_run(command, *, env, check):
        nonlocal calls
        with calls_guard:
            calls += 1
        time.sleep(0.05)
        Path(command[2]).write_bytes(b"sif")

    monkeypatch.setattr(apptainer_cache.subprocess, "run", fake_run)

    def pull():
        return pull_sif_if_missing(
            sif_path=sif_path,
            image_uri="docker://example/image:latest",
            cache_dir=None,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        commands = list(pool.map(lambda _index: pull(), range(2)))

    assert calls == 1
    assert commands[0] == commands[1]
    assert sif_path.read_bytes() == b"sif"


def test_failed_pull_removes_temporary_sif(tmp_path, monkeypatch):
    sif_path = tmp_path / "image.sif"

    def fail_pull(command, *, env, check):
        Path(command[2]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(255, command)

    monkeypatch.setattr(apptainer_cache.subprocess, "run", fail_pull)
    monkeypatch.setattr(apptainer_cache.time, "sleep", lambda _seconds: None)

    with pytest.raises(subprocess.CalledProcessError):
        pull_sif_if_missing(
            sif_path=sif_path,
            image_uri="docker://example/image:latest",
            cache_dir=None,
        )

    assert not sif_path.exists()
    assert list(tmp_path.glob(".*.sif")) == []


def test_dry_run_does_not_create_cache_directories(tmp_path):
    sif_path = tmp_path / "sifs/image.sif"

    command = pull_sif_if_missing(
        sif_path=sif_path,
        image_uri="docker://example/image:latest",
        cache_dir=tmp_path / "cache",
        dry_run=True,
    )

    assert command == [
        "apptainer",
        "pull",
        str(sif_path),
        "docker://example/image:latest",
    ]
    assert not tmp_path.joinpath("sifs").exists()
    assert not tmp_path.joinpath("cache").exists()


def test_dataset_image_resolution_matches_runtime_cache_paths(tmp_path):
    swebench = swebench_cache_images(
        [
            {"instance_id": "repo__project-1"},
            {"instance_id": "repo__project-2"},
        ],
        sif_dir=tmp_path / "swebench",
    )
    assert len(swebench) == 2
    assert swebench[0].image_uri == image_uri_from_template(
        DEFAULT_SWEBENCH_IMAGE_TEMPLATE,
        "repo__project-1",
    )
    assert swebench[0].sif_path == sif_path_for_image(
        tmp_path / "swebench",
        swebench[0].image_uri,
    )

    swesmith = swesmith_cache_images(
        [
            {"instance_id": "repo.task-1", "image_name": "org/repo-image:latest"},
            {"instance_id": "repo.task-2", "image_name": "org/repo-image:latest"},
            {"instance_id": "other.task-1", "image_name": "org/other-image:latest"},
        ],
        sif_dir=tmp_path / "swesmith",
    )
    assert len(swesmith) == 2
    assert swesmith[0].instance_ids == ("repo.task-1", "repo.task-2")
    assert swesmith[0].sif_path == sif_path_for_image(
        tmp_path / "swesmith",
        "org/repo-image:latest",
    )


def test_full_dry_run_uses_verified_and_task_file_swesmith_selections(tmp_path):
    swebench_dataset = tmp_path / "verified.jsonl"
    swebench_dataset.write_text(
        "".join(
            json.dumps({"instance_id": instance_id}) + "\n"
            for instance_id in ("repo__project-1", "repo__project-2")
        ),
        encoding="utf-8",
    )
    swesmith_dataset = tmp_path / "swesmith.jsonl"
    swesmith_dataset.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "instance_id": "repo.task-1",
                    "image_name": "swebench/repo-image:latest",
                },
                {
                    "instance_id": "repo.task-2",
                    "image_name": "swebench/repo-image:latest",
                },
                {
                    "instance_id": "other.task-1",
                    "image_name": "swebench/other-image:latest",
                },
            )
        ),
        encoding="utf-8",
    )
    task_ids = tmp_path / "smith-ids.txt"
    task_ids.write_text("repo.task-2\nother.task-1\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    args = build_parser().parse_args(
        [
            "--mode",
            "full",
            "--swebench-dataset",
            str(swebench_dataset),
            "--expected-swebench-tasks",
            "2",
            "--swebench-sif-dir",
            str(tmp_path / "verified-sifs"),
            "--swesmith-dataset",
            str(swesmith_dataset),
            "--swesmith-task-ids-file",
            str(task_ids),
            "--swesmith-sif-dir",
            str(tmp_path / "smith-sifs"),
            "--summary-output",
            str(summary_path),
            "--max-workers",
            "2",
            "--dry-run",
            "--no-progress",
        ]
    )

    summary = prebuild_apptainer_cache(args)

    assert summary["families"]["swebench"]["selected_tasks"] == 2
    assert summary["families"]["swebench"]["unique_images"] == 2
    assert summary["families"]["swesmith"]["selected_tasks"] == 2
    assert summary["families"]["swesmith"]["unique_images"] == 2
    assert summary["total_unique_images"] == 4
    assert summary["status_counts"] == {"dry_run": 4}
    assert summary_path.is_file()
    assert not (tmp_path / "verified-sifs").exists()
    assert not (tmp_path / "smith-sifs").exists()


def test_explicit_empty_task_file_does_not_expand_to_the_full_dataset(tmp_path):
    dataset = tmp_path / "swesmith.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "instance_id": "repo.task-1",
                "image_name": "swebench/repo-image:latest",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    task_ids = tmp_path / "empty.txt"
    task_ids.write_text("", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--datasets",
            "swesmith",
            "--swesmith-dataset",
            str(dataset),
            "--swesmith-task-ids-file",
            str(task_ids),
            "--dry-run",
            "--no-progress",
        ]
    )

    with pytest.raises(ValueError, match="No instance IDs found"):
        prebuild_apptainer_cache(args)
