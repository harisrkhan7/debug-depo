import csv
import json
import os

from debug_depo.analyze_run import analyze_run


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_run_writes_one_row_per_prediction_and_classifies_failures(tmp_path):
    run_root = tmp_path / "run"
    predictions = [
        {
            "instance_id": "repo__project-1",
            "model_name_or_path": "org/model",
            "model_patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n",
        },
        {
            "instance_id": "repo__project-2",
            "model_name_or_path": "org/model",
            "model_patch": "",
        },
    ]
    merged = run_root / "merged" / "predictions.jsonl"
    merged.parent.mkdir(parents=True)
    merged.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")

    trajectory_one = run_root / "rollouts/shard-0/trajectories/repo__project-1/trajectory.json"
    task_one = trajectory_one.parent / "task.json"
    write_json(task_one, {"instance_id": "repo__project-1", "repo": "repo/project"})
    write_json(
        trajectory_one,
        {
            "instance_id": "repo__project-1",
            "status": "completed",
            "returncode": 0,
            "patch_source": "preds.json",
            "config": {"max_steps": 200, "context_length": 100},
        },
    )
    os.utime(task_one, (1000, 1000))
    os.utime(trajectory_one, (1123, 1123))
    write_json(
        trajectory_one.parent / "repo__project-1/repo__project-1.traj.json",
        {
            "instance_id": "repo__project-1",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": "action\n```bash\npytest -q\n```",
                    "extra": {
                        "response": {
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 2,
                                "total_tokens": 12,
                            },
                            "choices": [{"finish_reason": "stop"}],
                        }
                    },
                },
                {"role": "user", "content": "<returncode>1</returncode>"},
            ],
            "info": {
                "exit_status": "Submitted",
                "model_stats": {"api_calls": 1},
                "config": {"agent": {"step_limit": 10}},
            },
        },
    )

    trajectory_two = run_root / "rollouts/shard-1/trajectories/repo__project-2/trajectory.json"
    write_json(
        trajectory_two,
        {
            "instance_id": "repo__project-2",
            "status": "error",
            "returncode": 0,
            "mini_swe_exit_status": "LimitsExceeded",
        },
    )
    (trajectory_two.parent / "stderr.txt").write_text("step limit exceeded\n", encoding="utf-8")

    report_path = (
        run_root
        / "evaluation/logs/run/org__model/repo__project-1/report.json"
    )
    write_json(
        report_path,
        {
            "repo__project-1": {
                "resolved": False,
                "tests_status": {
                    "FAIL_TO_PASS": {"success": ["a"], "failure": ["b"]},
                    "PASS_TO_PASS": {"success": ["c", "d"], "failure": []},
                },
            }
        },
    )

    output_csv = run_root / "analysis/instances.csv"
    summary_path = run_root / "analysis/summary.json"
    summary = analyze_run(run_root, output_csv, summary_path, expected_count=2)

    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["instances"] == 2
    assert summary["rollouts_with_patch"] == 1
    assert summary["evaluations_resolved"] == 0
    assert rows[0]["repo"] == "repo/project"
    assert rows[0]["evaluation_status"] == "unresolved"
    assert rows[0]["trajectory_steps"] == "2"
    assert rows[0]["trajectory_messages"] == "4"
    assert rows[0]["agent_action_steps"] == "1"
    assert rows[0]["model_api_calls"] == "1"
    assert rows[0]["prompt_tokens_total"] == "10"
    assert rows[0]["completion_tokens_total"] == "2"
    assert rows[0]["max_prompt_tokens"] == "10"
    assert rows[0]["step_limit_utilization"] == "0.1"
    assert rows[0]["context_limit_utilization"] == "0.1"
    assert rows[0]["commands_executed"] == "1"
    assert rows[0]["commands_failed"] == "1"
    assert rows[0]["test_command_steps"] == "1"
    assert rows[0]["submission_step"] == "1"
    assert rows[0]["seconds_per_action"] == "123.0"
    assert rows[0]["duration_seconds"] == "123.0"
    assert rows[0]["fail_to_pass_failure"] == "1"
    assert rows[1]["evaluation_status"] == "empty_patch"
    assert rows[1]["failure_category"] == "step_limit"
    assert "LimitsExceeded" in rows[1]["failure_reason"]


def test_analyze_run_reports_expected_count_mismatch(tmp_path):
    run_root = tmp_path / "run"
    predictions = run_root / "merged/predictions.jsonl"
    predictions.parent.mkdir(parents=True)
    predictions.write_text(
        json.dumps({"instance_id": "repo__project-1", "model_patch": ""}) + "\n",
        encoding="utf-8",
    )

    summary = analyze_run(
        run_root,
        run_root / "analysis/instances.csv",
        run_root / "analysis/summary.json",
        expected_count=500,
    )

    assert summary["matches_expected_count"] is False
    assert summary["instances"] == 1


def test_analyze_run_smoke_samples_every_prediction_shard(tmp_path):
    run_root = tmp_path / "run"
    for shard in range(3):
        predictions = run_root / f"rollouts/shard-{shard}/predictions.jsonl"
        predictions.parent.mkdir(parents=True)
        rows = [
            {
                "instance_id": f"repo__project-{shard}-{index}",
                "model_patch": "diff --git a/a b/a\n+new\n" if index == 0 else "",
            }
            for index in range(2)
        ]
        predictions.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    output_csv = run_root / "analysis-smoke/instances.csv"
    summary = analyze_run(
        run_root,
        output_csv,
        run_root / "analysis-smoke/summary.json",
        expected_count=6,
        sample_per_shard=1,
    )
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["source_instances"] == 6
    assert summary["instances"] == 3
    assert summary["sampled_shards"] == {"shard-0": 1, "shard-1": 1, "shard-2": 1}
    assert {row["shard"] for row in rows} == {"shard-0", "shard-1", "shard-2"}
    assert all(row["patch_present"] == "True" for row in rows)
