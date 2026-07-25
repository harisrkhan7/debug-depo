import json
from pathlib import Path

from debug_depo.swesmith_progress import (
    inspect_collection,
    inspect_evaluation,
    render_progress,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_manifest(
    run_root: Path,
    shard_index: int,
    task_ids: list[str],
) -> None:
    write_json(
        run_root / "collection" / f"shard-{shard_index}" / "collection_manifest.json",
        {
            "expected_tasks": 3,
            "num_shards": 2,
            "shard_index": shard_index,
            "n_tasks": len(task_ids),
            "total_samples_per_task": 2,
            "task_instance_ids": task_ids,
        },
    )


def write_trajectory(
    run_root: Path,
    shard_index: int,
    sample_index: int,
    instance_id: str,
    status: str,
) -> None:
    write_json(
        run_root
        / "collection"
        / f"shard-{shard_index}"
        / "samples"
        / f"sample-{sample_index}"
        / "trajectories"
        / instance_id
        / "trajectory.json",
        {"instance_id": instance_id, "status": status},
    )


def test_reports_per_shard_and_overall_finished_task_counts(tmp_path):
    write_manifest(tmp_path, 0, ["repo.task-0", "repo.task-2"])
    write_manifest(tmp_path, 1, ["repo.task-1"])

    write_trajectory(tmp_path, 0, 0, "repo.task-0", "completed")
    write_trajectory(tmp_path, 0, 1, "repo.task-0", "model_terminated")
    write_trajectory(tmp_path, 0, 0, "repo.task-2", "error")
    write_trajectory(tmp_path, 1, 0, "repo.task-1", "mocked")
    write_trajectory(tmp_path, 1, 1, "repo.task-1", "completed")

    progress = inspect_collection(tmp_path)

    assert [len(shard.collected_task_ids) for shard in progress.shards] == [1, 1]
    assert progress.collected_tasks == 2
    assert progress.expected_tasks == 3
    assert progress.finished_rollouts == 4
    assert progress.expected_rollouts == 6
    assert progress.error_rollouts == 1

    evaluation = inspect_evaluation(tmp_path, progress)
    output = render_progress(progress, evaluation, bar_width=10)
    assert "shard-0" in output
    assert "shard-1" in output
    assert "OVERALL" in output
    assert "2/3" in output
    assert "4/6" in output


def test_missing_manifest_is_shown_as_waiting_shard(tmp_path):
    write_manifest(tmp_path, 0, ["repo.task-0", "repo.task-2"])

    progress = inspect_collection(tmp_path)

    assert len(progress.shards) == 2
    assert progress.shards[1].expected_tasks == 1
    assert not progress.shards[1].manifest_present
    evaluation = inspect_evaluation(tmp_path, progress)
    output = render_progress(progress, evaluation, bar_width=10)
    assert "shard-1" in output
    assert "not started" in output


def test_reports_live_and_summarized_evaluation_progress(tmp_path):
    write_manifest(tmp_path, 0, ["repo.task-0", "repo.task-2"])
    write_manifest(tmp_path, 1, ["repo.task-1"])
    collection = inspect_collection(tmp_path)

    predictions = tmp_path / "merged" / "sample-0" / "predictions.jsonl"
    predictions.parent.mkdir(parents=True)
    predictions.write_text(
        "\n".join(
            [
                json.dumps({"instance_id": "repo.task-0", "model_patch": "patch"}),
                json.dumps({"instance_id": "repo.task-1", "model_patch": "patch"}),
                json.dumps({"instance_id": "repo.task-2", "model_patch": ""}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sample_0 = tmp_path / "evaluation" / "sample-0"
    (sample_0 / "logs").mkdir(parents=True)
    write_json(sample_0 / "logs" / "repo.task-0" / "report.json", {"resolved": True})
    write_json(sample_0 / "logs" / "repo.task-0" / "cache_key.json", {"key": 1})
    (sample_0 / "logs" / "repo.task-1").mkdir(parents=True)
    (sample_0 / "logs" / "repo.task-1" / "patch_status.txt").write_text(
        "failed\n",
        encoding="utf-8",
    )

    write_json(
        tmp_path / "evaluation" / "sample-1" / "summary.json",
        {
            "status_ids": {
                "completed": ["repo.task-0"],
                "empty_patch": ["repo.task-1"],
                "timeout": ["repo.task-2"],
            }
        },
    )

    evaluation = inspect_evaluation(tmp_path, collection)

    assert [len(sample.evaluated_task_ids) for sample in evaluation.samples] == [3, 3]
    assert evaluation.evaluated == 6
    assert evaluation.fully_evaluated_task_ids == {
        "repo.task-0",
        "repo.task-1",
        "repo.task-2",
    }
    output = render_progress(collection, evaluation, bar_width=10)
    assert "EVALUATION" in output
    assert "sample-0" in output
    assert "sample-1" in output
    assert "Fully evaluated tasks" in output
