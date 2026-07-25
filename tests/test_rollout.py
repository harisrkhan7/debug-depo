import json

import pytest

from debug_depo.rollout import build_parser, collect_rollouts, result_from_existing
from debug_depo.utils import read_json, read_jsonl


def write_dataset(path):
    rows = [
        {
            "instance_id": "repo__repo-1",
            "repo": "repo/repo",
            "problem_statement": "Fix one",
            "patch": "diff --git a/a.py b/a.py\n",
        },
        {
            "instance_id": "repo__repo-2",
            "repo": "repo/repo",
            "problem_statement": "Fix two",
            "patch": "diff --git a/b.py b/b.py\n",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_mock_collection_writes_predictions_and_summary(tmp_path):
    dataset = tmp_path / "tasks.jsonl"
    output_dir = tmp_path / "out"
    write_dataset(dataset)
    args = build_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--mock",
            "--mock-patch",
            "gold",
            "--limit",
            "2",
            "--no-progress",
        ]
    )

    summary = collect_rollouts(args)
    predictions = read_jsonl(output_dir / "predictions.jsonl")

    assert summary["n_tasks"] == 2
    assert summary["n_with_patch"] == 2
    assert [row["instance_id"] for row in predictions] == ["repo__repo-1", "repo__repo-2"]
    run_config = read_json(output_dir / "run_config.json")
    assert run_config["schema_version"] == 2
    assert run_config["dataset_revision"] is None
    assert run_config["max_steps"] == 200
    assert run_config["timeout_seconds"] == 7200
    assert run_config["task_instance_ids"] == ["repo__repo-1", "repo__repo-2"]
    assert len(run_config["task_rows_sha256"]) == 64


def test_mock_collection_supports_rollout_workers(tmp_path):
    dataset = tmp_path / "tasks.jsonl"
    output_dir = tmp_path / "out"
    write_dataset(dataset)
    args = build_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--mock",
            "--mock-patch",
            "gold",
            "--limit",
            "2",
            "--rollout-workers",
            "2",
            "--no-progress",
        ]
    )

    summary = collect_rollouts(args)
    predictions = read_jsonl(output_dir / "predictions.jsonl")

    assert summary["rollout_workers"] == 2
    assert read_json(output_dir / "run_config.json")["rollout_workers"] == 2
    assert [row["instance_id"] for row in predictions] == ["repo__repo-1", "repo__repo-2"]


def test_run_config_redacts_a_literal_api_key_from_custom_command(tmp_path):
    dataset = tmp_path / "tasks.jsonl"
    output_dir = tmp_path / "out"
    write_dataset(dataset)
    args = build_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
            "--mock",
            "--llm-api-key",
            "real-secret-key",
            "--agentforge-command",
            "run --api-key real-secret-key",
            "--no-progress",
        ]
    )

    collect_rollouts(args)

    run_config_path = output_dir / "run_config.json"
    run_config = read_json(run_config_path)
    assert run_config["agentforge_command"] == "run --api-key <redacted>"
    assert "real-secret-key" not in run_config_path.read_text(encoding="utf-8")


def test_existing_miniswe_failure_is_not_reused_as_completed(tmp_path):
    output_dir = tmp_path / "out"
    trajectory_dir = output_dir / "trajectories" / "repo__repo-1"
    trajectory_dir.mkdir(parents=True)
    (trajectory_dir / "trajectory.json").write_text(
        json.dumps({"status": "completed", "patch": "UndefinedError traceback", "patch_source": "preds.json"})
    )
    (trajectory_dir / "exit_statuses_1.yaml").write_text(
        "instances_by_exit_status:\n"
        "    Uncaught NameError:\n"
        "    - repo__repo-1\n"
    )

    result = result_from_existing(output_dir, {"instance_id": "repo__repo-1"})

    assert result["status"] == "error"
    assert result["patch"] == ""
    assert result["mini_swe_exit_status"] == "Uncaught NameError"


def test_existing_miniswe_model_termination_is_reused_as_finished(tmp_path):
    output_dir = tmp_path / "out"
    trajectory_dir = output_dir / "trajectories" / "repo__repo-1"
    trajectory_dir.mkdir(parents=True)
    (trajectory_dir / "trajectory.json").write_text(
        json.dumps({"status": "error", "patch": "", "patch_source": None})
    )
    (trajectory_dir / "exit_statuses_1.yaml").write_text(
        "instances_by_exit_status:\n"
        "    LimitsExceeded:\n"
        "    - repo__repo-1\n"
    )

    result = result_from_existing(output_dir, {"instance_id": "repo__repo-1"})

    assert result["status"] == "model_terminated"
    assert result["patch"] == ""
    assert result["mini_swe_exit_status"] == "LimitsExceeded"


def test_require_complete_rejects_collection_errors(tmp_path, monkeypatch):
    dataset = tmp_path / "tasks.jsonl"
    output_dir = tmp_path / "out"
    write_dataset(dataset)
    args = build_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
            "--require-complete",
            "--no-progress",
        ]
    )

    def fail_rollout(*_args, **_kwargs):
        raise RuntimeError("agent environment unavailable")

    monkeypatch.setattr(
        "debug_depo.rollout.run_agentforge_instance",
        fail_rollout,
    )

    with pytest.raises(RuntimeError, match="did not finish every rollout"):
        collect_rollouts(args)

    summary = read_json(output_dir / "summary.json")
    assert summary["n_finished"] == 0
    assert summary["n_errors"] == 1


def test_collection_resume_retries_an_existing_error(tmp_path, monkeypatch):
    dataset = tmp_path / "tasks.jsonl"
    output_dir = tmp_path / "out"
    write_dataset(dataset)
    trajectory_dir = output_dir / "trajectories" / "repo__repo-1"
    trajectory_dir.mkdir(parents=True)
    (trajectory_dir / "trajectory.json").write_text(
        json.dumps({"status": "error", "patch": "", "patch_source": None}),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
            "--require-complete",
            "--no-progress",
        ]
    )
    retried = []

    def complete_rollout(task, _output_dir, _config):
        retried.append(str(task["instance_id"]))
        return {
            "instance_id": str(task["instance_id"]),
            "status": "completed",
            "patch": "",
            "patch_source": None,
        }

    monkeypatch.setattr(
        "debug_depo.rollout.run_agentforge_instance",
        complete_rollout,
    )

    summary = collect_rollouts(args)

    assert retried == ["repo__repo-1"]
    assert summary["n_finished"] == 1
    assert summary["n_errors"] == 0


def test_collection_records_miniswe_provenance(tmp_path, monkeypatch):
    dataset = tmp_path / "tasks.jsonl"
    output_dir = tmp_path / "out"
    write_dataset(dataset)
    monkeypatch.setattr(
        "debug_depo.rollout.package_provenance",
        lambda *_args: {
            "version": "1.14.4",
            "revision": "mini-commit",
            "working_tree_diff_sha256": "patch-hash",
        },
    )
    args = build_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--harness",
            "mini-swe-agent-plus",
            "--mock",
            "--limit",
            "1",
            "--no-progress",
        ]
    )

    collect_rollouts(args)

    run_config = read_json(output_dir / "run_config.json")
    assert run_config["mini_swe_version"] == "1.14.4"
    assert run_config["mini_swe_revision"] == "mini-commit"
    assert run_config["mini_swe_working_tree_diff_sha256"] == "patch-hash"


def test_collection_refuses_resume_when_selected_task_content_changes(tmp_path):
    dataset = tmp_path / "tasks.jsonl"
    output_dir = tmp_path / "out"
    write_dataset(dataset)
    args = build_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--mock",
            "--limit",
            "1",
            "--no-progress",
        ]
    )
    collect_rollouts(args)
    rows = [
        {
            "instance_id": "repo__repo-1",
            "repo": "repo/repo",
            "problem_statement": "Changed after the first run",
            "patch": "diff --git a/a.py b/a.py\n",
        }
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="task_rows_sha256"):
        collect_rollouts(args)
