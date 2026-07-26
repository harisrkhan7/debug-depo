import csv
import json
from pathlib import Path

import pytest

from debug_depo.swesmith_analyze import analyze_swesmith
from debug_depo.swesmith_collect import (
    build_parser as build_collect_parser,
    collect_swesmith,
    parse_temperatures,
    rollout_seed,
    temperature_schedule,
)
from debug_depo.swesmith_evaluate import (
    _write_runner_script,
    build_parser as build_evaluate_parser,
    evaluate_swesmith,
    run_instance,
)
from debug_depo.utils import read_jsonl


def write_tasks(path):
    rows = [
        {
            "instance_id": "repo__project.task-1",
            "repo": "repo/project",
            "problem_statement": "Fix task one",
            "patch": "diff --git a/a.py b/a.py\n+change\n",
        },
        {
            "instance_id": "repo__project.task-2",
            "repo": "repo/project",
            "problem_statement": "Fix task two",
            "patch": "diff --git a/b.py b/b.py\n+change\n",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_temperature_schedule_and_seeds_are_stable():
    temperatures = parse_temperatures("0.6:0.7")
    assert temperatures == [0.6, 0.7]
    assert temperature_schedule(temperatures, 4) == [
        (0.6, 0),
        (0.6, 1),
        (0.6, 2),
        (0.6, 3),
        (0.7, 0),
        (0.7, 1),
        (0.7, 2),
        (0.7, 3),
    ]
    seeds = [rollout_seed(42, "repo__task-1", index) for index in range(8)]
    assert seeds == [rollout_seed(42, "repo__task-1", index) for index in range(8)]
    assert len(set(seeds)) == 8


def test_collect_parser_normalizes_limit_environment(monkeypatch):
    monkeypatch.setenv("LIMIT", "")
    assert build_collect_parser().parse_args([]).limit is None

    monkeypatch.setenv("LIMIT", "5000")
    assert build_collect_parser().parse_args([]).limit == 5000


@pytest.mark.parametrize("spec", ["", "-0.1", "nan", "inf", "0.6:0.6"])
def test_temperature_spec_rejects_invalid_values(spec):
    with pytest.raises(ValueError):
        parse_temperatures(spec)


def test_require_complete_rejects_collection_errors(tmp_path, monkeypatch):
    dataset = tmp_path / "tasks.jsonl"
    output_dir = tmp_path / "collection"
    write_tasks(dataset)
    args = build_collect_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--temperatures",
            "0.6",
            "--runs-per-temperature",
            "1",
            "--rollout-workers",
            "1",
            "--require-complete",
            "--no-progress",
        ]
    )

    def fail_rollout(*_args, **_kwargs):
        raise RuntimeError("agent environment unavailable")

    monkeypatch.setattr(
        "debug_depo.swesmith_collect.run_agentforge_instance",
        fail_rollout,
    )

    with pytest.raises(RuntimeError, match="did not finish every rollout"):
        collect_swesmith(args)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_completed"] == 0
    assert summary["n_errors"] == 2


def test_collection_resume_retries_only_error_slots(tmp_path, monkeypatch):
    dataset = tmp_path / "tasks.jsonl"
    output_dir = tmp_path / "collection"
    write_tasks(dataset)
    args = build_collect_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--temperatures",
            "0.6",
            "--runs-per-temperature",
            "1",
            "--rollout-workers",
            "1",
            "--require-complete",
            "--no-progress",
        ]
    )

    def write_result(task, sample_dir, status):
        instance_id = str(task["instance_id"])
        trajectory_dir = Path(sample_dir) / "trajectories" / instance_id
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        (trajectory_dir / "trajectory.json").write_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "status": status,
                    "patch": "",
                    "patch_source": None,
                }
            ),
            encoding="utf-8",
        )
        return {
            "instance_id": instance_id,
            "status": status,
            "patch": "",
            "patch_source": None,
        }

    def first_attempt(task, sample_dir, _config):
        status = (
            "completed"
            if task["instance_id"] == "repo__project.task-1"
            else "error"
        )
        return write_result(task, sample_dir, status)

    monkeypatch.setattr(
        "debug_depo.swesmith_collect.run_agentforge_instance",
        first_attempt,
    )
    with pytest.raises(RuntimeError, match="did not finish every rollout"):
        collect_swesmith(args)

    retried = []

    def retry_attempt(task, sample_dir, _config):
        retried.append(str(task["instance_id"]))
        return write_result(task, sample_dir, "completed")

    monkeypatch.setattr(
        "debug_depo.swesmith_collect.run_agentforge_instance",
        retry_attempt,
    )
    summary = collect_swesmith(args)

    assert retried == ["repo__project.task-2"]
    assert summary["n_finished"] == 2
    assert summary["n_errors"] == 0


def test_require_complete_accepts_model_terminations(tmp_path, monkeypatch):
    dataset = tmp_path / "tasks.jsonl"
    write_tasks(dataset)
    args = build_collect_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(tmp_path / "collection"),
            "--temperatures",
            "0.6",
            "--runs-per-temperature",
            "1",
            "--rollout-workers",
            "1",
            "--require-complete",
            "--no-progress",
        ]
    )

    def model_terminated(task, _sample_dir, _config):
        return {
            "instance_id": str(task["instance_id"]),
            "status": "model_terminated",
            "patch": "",
            "patch_source": None,
        }

    monkeypatch.setattr(
        "debug_depo.swesmith_collect.run_agentforge_instance",
        model_terminated,
    )
    summary = collect_swesmith(args)

    assert summary["n_completed"] == 0
    assert summary["n_model_terminated"] == 2
    assert summary["n_finished"] == 2
    assert summary["n_errors"] == 0


def test_collection_rejects_an_unexpected_selected_task_count(tmp_path):
    dataset = tmp_path / "tasks.jsonl"
    write_tasks(dataset)
    args = build_collect_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(tmp_path / "collection"),
            "--expected-tasks",
            "3",
            "--mock",
            "--no-progress",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Expected 3 selected SWE-smith tasks, found 2",
    ):
        collect_swesmith(args)


def test_collection_rejects_pool_way_task_initialization(tmp_path):
    dataset = tmp_path / "tasks.jsonl"
    write_tasks(dataset)
    args = build_collect_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(tmp_path / "collection"),
            "--mini-runner",
            "pool_way",
            "--mini-environment-class",
            "docker",
            "--no-progress",
        ]
    )

    with pytest.raises(ValueError, match="pool_way.*ignores task startup commands"):
        collect_swesmith(args)


def test_mock_collect_evaluate_analyze_pipeline(tmp_path):
    dataset = tmp_path / "tasks.jsonl"
    run_root = tmp_path / "run"
    collection_root = run_root / "collection/shard-0"
    write_tasks(dataset)

    collect_args = build_collect_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(collection_root),
            "--runs-per-temperature",
            "4",
            "--temperatures",
            "0.6:0.7",
            "--rollout-workers",
            "2",
            "--mock",
            "--mock-patch",
            "gold",
            "--no-progress",
        ]
    )
    collection = collect_swesmith(collect_args)

    assert collection["n_tasks"] == 2
    assert collection["n_rollouts"] == 16
    assert collection["n_with_patch"] == 16
    assert collection["temperatures"] == [0.6, 0.7]
    assert collection["runs_per_temperature"] == 4
    assert collection["total_samples_per_task"] == 8
    assert collection["task_initialization"] == "checkout_instance_branch"
    for sample_index in range(8):
        rows = read_jsonl(
            collection_root / f"samples/sample-{sample_index}/predictions.jsonl"
        )
        assert len(rows) == 2
        assert {row["sample_index"] for row in rows} == {sample_index}
        assert {row["patch_apply_mode"] for row in rows} == {"reverse"}
        merged = run_root / f"merged/sample-{sample_index}/predictions.jsonl"
        merged.parent.mkdir(parents=True)
        merged.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        evaluation_root = run_root / f"evaluation/sample-{sample_index}"
        evaluate_args = build_evaluate_parser().parse_args(
            [
                "--dataset",
                str(dataset),
                "--predictions-path",
                str(merged),
                "--summary-output",
                str(evaluation_root / "summary.json"),
                "--log-dir",
                str(evaluation_root / "logs"),
                "--runtime",
                "mock",
                "--max-workers",
                "2",
            ]
        )
        evaluation = evaluate_swesmith(evaluate_args)
        assert evaluation["resolved_instances"] == 2

    summary = analyze_swesmith(
        run_root,
        rollouts_csv=run_root / "analysis/rollouts.csv",
        tasks_csv=run_root / "analysis/tasks.csv",
        summary_output=run_root / "analysis/summary.json",
        total_samples=8,
        runs_per_temperature=4,
        expected_tasks=2,
    )
    with (run_root / "analysis/tasks.csv").open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle))

    assert summary["rollouts"] == 16
    assert summary["resolved_rollouts"] == 16
    assert summary["efficiency"]["trajectories"] == 16
    assert summary["efficiency"]["resolved_trajectories"] == 16
    assert summary["efficiency"]["resolution_rate"] == 1.0
    assert summary["efficiency"]["all"]["total_tokens"]["available"] == 0
    assert summary["efficiency"]["total_tokens_per_resolved_task"] is None
    assert summary["mixed_temperature_pass_at_k"] == {
        str(index): 1.0 for index in range(1, 9)
    }
    assert {
        temperature: metrics["pass_at_k"]
        for temperature, metrics in summary["temperatures"].items()
    } == {
        temperature: {str(index): 1.0 for index in range(1, 5)}
        for temperature in ("0.6", "0.7")
    }
    assert len(tasks) == 2
    assert all(row["runs_resolved"] == "8" for row in tasks)
    assert all(row["mixed_temperature_pass_at_4"] == "1.0" for row in tasks)
    assert all(row["resolved_at_least_once"] == "True" for row in tasks)

    collect_args.temperatures = "0.5:0.7"
    with pytest.raises(ValueError, match="incompatible.*temperatures"):
        collect_swesmith(collect_args)


def test_analysis_rejects_an_incoherent_sample_layout(tmp_path):
    with pytest.raises(ValueError, match="divisible"):
        analyze_swesmith(
            tmp_path,
            rollouts_csv=tmp_path / "rollouts.csv",
            tasks_csv=tmp_path / "tasks.csv",
            summary_output=tmp_path / "summary.json",
            total_samples=10,
            runs_per_temperature=4,
        )


def test_evaluation_rejects_requested_ids_missing_from_predictions(tmp_path):
    dataset = tmp_path / "tasks.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_tasks(dataset)
    predictions.write_text(
        json.dumps(
            {
                "instance_id": "repo__project.task-1",
                "model_name_or_path": "model",
                "model_patch": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = build_evaluate_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--predictions-path",
            str(predictions),
            "--summary-output",
            str(tmp_path / "summary.json"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--runtime",
            "mock",
            "--instance-id",
            "repo__project.task-2",
        ]
    )

    with pytest.raises(ValueError, match="absent from the predictions"):
        evaluate_swesmith(args)


def test_evaluation_cache_is_keyed_by_prediction_content(tmp_path):
    args = build_evaluate_parser().parse_args(
        [
            "--dataset",
            "dataset",
            "--predictions-path",
            "predictions.jsonl",
            "--summary-output",
            str(tmp_path / "summary.json"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--runtime",
            "mock",
        ]
    )
    instance = {"instance_id": "repo__project.task-1"}
    prediction = {
        "instance_id": instance["instance_id"],
        "model_name_or_path": "model",
        "model_patch": "first patch",
    }

    assert run_instance(instance, prediction, args)["status"] == "completed"
    assert run_instance(instance, prediction, args)["status"] == "cached_report"

    reverse_prediction = {**prediction, "patch_apply_mode": "reverse"}
    assert run_instance(instance, reverse_prediction, args)["status"] == "completed"
    assert run_instance(instance, reverse_prediction, args)["status"] == "cached_report"

    changed_prediction = {**prediction, "model_patch": "different patch"}
    assert run_instance(instance, changed_prediction, args)["status"] == "completed"

    args.dataset_revision = "new-dataset-revision"
    assert run_instance(instance, changed_prediction, args)["status"] == "completed"


def test_evaluation_dry_run_does_not_mutate_cached_reports(tmp_path):
    args = build_evaluate_parser().parse_args(
        [
            "--dataset",
            "dataset",
            "--predictions-path",
            "predictions.jsonl",
            "--summary-output",
            str(tmp_path / "summary.json"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--runtime",
            "mock",
            "--overwrite",
            "--dry-run",
        ]
    )
    instance = {"instance_id": "repo__project.task-1"}
    prediction = {
        "instance_id": instance["instance_id"],
        "model_name_or_path": "model",
        "model_patch": "patch",
    }
    log_dir = tmp_path / "logs" / instance["instance_id"]
    log_dir.mkdir(parents=True)
    report_path = log_dir / "report.json"
    cache_key_path = log_dir / "cache_key.json"
    report_path.write_text('{"resolved": true}\n', encoding="utf-8")
    cache_key_path.write_text('{"old": "cache"}\n', encoding="utf-8")
    before = {
        path.name: path.read_bytes()
        for path in log_dir.iterdir()
    }

    result = run_instance(instance, prediction, args)

    assert result["status"] == "dry_run"
    assert {
        path.name: path.read_bytes()
        for path in log_dir.iterdir()
    } == before


def test_evaluation_dry_run_does_not_write_new_artifacts(tmp_path):
    dataset = tmp_path / "tasks.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_tasks(dataset)
    predictions.write_text(
        json.dumps(
            {
                "instance_id": "repo__project.task-1",
                "model_name_or_path": "model",
                "model_patch": "patch",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = tmp_path / "evaluation" / "summary.json"
    log_dir = tmp_path / "evaluation" / "logs"
    args = build_evaluate_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--predictions-path",
            str(predictions),
            "--summary-output",
            str(summary_path),
            "--log-dir",
            str(log_dir),
            "--runtime",
            "mock",
            "--dry-run",
        ]
    )

    summary = evaluate_swesmith(args)

    assert summary["status_ids"] == {"dry_run": ["repo__project.task-1"]}
    assert not summary_path.exists()
    assert not log_dir.exists()


def test_require_complete_rejects_evaluation_infrastructure_errors(
    tmp_path,
    monkeypatch,
):
    dataset = tmp_path / "tasks.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_tasks(dataset)
    predictions.write_text(
        json.dumps(
            {
                "instance_id": "repo__project.task-1",
                "model_name_or_path": "model",
                "model_patch": "patch",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = build_evaluate_parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--predictions-path",
            str(predictions),
            "--summary-output",
            str(tmp_path / "summary.json"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--runtime",
            "mock",
            "--require-complete",
        ]
    )

    def fail_instance(*_args, **_kwargs):
        raise RuntimeError("container unavailable")

    monkeypatch.setattr(
        "debug_depo.swesmith_evaluate.run_instance",
        fail_instance,
    )

    with pytest.raises(RuntimeError, match="infrastructure outcomes"):
        evaluate_swesmith(args)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["scored_instances"] == 0
    assert summary["status_ids"] == {"error": ["repo__project.task-1"]}


def test_analysis_excludes_dry_runs_from_scored_metrics(tmp_path):
    run_root = tmp_path / "run"
    predictions = run_root / "merged/sample-0/predictions.jsonl"
    predictions.parent.mkdir(parents=True)
    predictions.write_text(
        json.dumps(
            {
                "instance_id": "repo__project.task-1",
                "model_patch": "patch",
                "sample_index": 0,
                "temperature": 0.6,
                "temperature_run_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evaluation_summary = run_root / "evaluation/sample-0/summary.json"
    evaluation_summary.parent.mkdir(parents=True)
    evaluation_summary.write_text(
        json.dumps(
            {
                "status_ids": {"dry_run": ["repo__project.task-1"]},
                "resolved_ids": [],
            }
        ),
        encoding="utf-8",
    )
    stale_report = (
        run_root
        / "evaluation/sample-0/logs/repo__project.task-1/report.json"
    )
    stale_report.parent.mkdir(parents=True)
    stale_report.write_text(
        json.dumps(
            {
                "instance_id": "repo__project.task-1",
                "resolved": True,
            }
        ),
        encoding="utf-8",
    )

    summary = analyze_swesmith(
        run_root,
        rollouts_csv=run_root / "analysis/rollouts.csv",
        tasks_csv=run_root / "analysis/tasks.csv",
        summary_output=run_root / "analysis/summary.json",
        total_samples=1,
        runs_per_temperature=1,
    )

    assert summary["evaluated_rollouts"] == 0
    assert summary["fully_evaluated_tasks"] == 0
    assert summary["mixed_temperature_pass_at_k"] == {"1": None}
    assert summary["unscored_evaluation_status_counts"] == {"dry_run": 1}


def test_apptainer_runner_uses_an_isolated_git_config(tmp_path):
    path = _write_runner_script(
        tmp_path,
        instance_id="repo__project.task-1",
        test_files=[],
    )

    script = path.read_text(encoding="utf-8")
    assert 'export GIT_CONFIG_GLOBAL="/swesmith_eval/gitconfig"' in script
    assert "git config --global --replace-all safe.directory /testbed" in script
    assert "git apply --verbose --reverse" not in script
    assert "patch --batch --fuzz=5 -R" not in script


def test_apptainer_runner_can_reverse_swesmith_gold_patch(tmp_path):
    path = _write_runner_script(
        tmp_path,
        instance_id="repo__project.task-1",
        test_files=[],
        patch_apply_mode="reverse",
    )

    script = path.read_text(encoding="utf-8")
    assert '"git apply --verbose --reverse"' in script
    assert '"git apply --verbose --reject --reverse"' in script
    assert '"patch --batch --fuzz=5 -R -p1 -i"' in script
