import json

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
    assert read_json(output_dir / "run_config.json")["max_steps"] == 200


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
