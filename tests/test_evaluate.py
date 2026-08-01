import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from debug_depo import evaluate
from debug_depo.evaluate import build_evaluation_command, model_report_name, summarize_report


def args(**overrides):
    values = {
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "dataset_revision": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
        "split": "test",
        "predictions_path": "predictions.jsonl",
        "max_workers": 2,
        "run_id": "run",
        "timeout": 1800,
        "cache_level": "env",
        "clean": False,
        "report_dir": "results",
        "force_rebuild": False,
        "namespace": "",
        "auto_namespace": False,
        "instance_ids": ["a", "b"],
        "instance_ids_file": None,
        "modal": False,
        "model": "org/model",
        "summary_output": None,
        "eval_cwd": ".",
        "dry_run": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_build_evaluation_command_includes_empty_namespace_for_local_builds():
    command = build_evaluation_command(args())

    namespace_index = command.index("--namespace")
    assert command[namespace_index + 1] == ""
    assert command[-3:] == ["--instance_ids", "a", "b"]


def test_summarize_report_compares_full_verified_target():
    summary = summarize_report(
        {
            "total_instances": 500,
            "submitted_instances": 500,
            "completed_instances": 500,
            "resolved_instances": 191,
            "unresolved_instances": 309,
            "empty_patch_instances": 0,
            "error_instances": 0,
        }
    )

    assert summary["resolution_rate"] == 191 / 500
    assert summary["target_name"] == "klear-agentforge-8b-sft-swe-bench-verified"
    assert summary["resolved_delta_vs_target"] == 0
    assert model_report_name("org/model", "run") == "org__model.run.json"


def test_summarize_report_omits_verified_target_for_other_evaluation_setups():
    report = {
        "total_instances": 100,
        "submitted_instances": 100,
        "completed_instances": 100,
        "resolved_instances": 40,
        "unresolved_instances": 60,
        "empty_patch_instances": 0,
        "error_instances": 0,
    }

    summary = summarize_report(
        report,
        dataset="SWE-bench/SWE-smith-py",
        split="validation",
        model="org/trained-model",
    )

    assert summary["dataset"] == "SWE-bench/SWE-smith-py"
    assert summary["split"] == "validation"
    assert summary["model"] == "org/trained-model"
    assert summary["resolution_rate"] == 0.4
    assert summary["target_name"] is None
    assert summary["target_score"] is None
    assert summary["target_resolved"] is None
    assert summary["target_total"] is None
    assert summary["resolved_delta_vs_target"] is None


def test_summarize_report_omits_full_target_for_verified_subset():
    summary = summarize_report(
        {
            "total_instances": 500,
            "submitted_instances": 5,
            "completed_instances": 5,
            "resolved_instances": 2,
            "unresolved_instances": 3,
            "empty_patch_instances": 0,
            "error_instances": 0,
        }
    )

    assert summary["target_name"] is None
    assert summary["resolved_delta_vs_target"] is None


def test_summarize_report_target_requires_matching_dataset_split_and_model():
    report = {
        "total_instances": 500,
        "submitted_instances": 500,
        "completed_instances": 500,
        "resolved_instances": 191,
        "unresolved_instances": 309,
        "empty_patch_instances": 0,
        "error_instances": 0,
    }

    for overrides in (
        {"dataset": "org/other-dataset"},
        {"dataset_revision": "different-revision"},
        {"split": "validation"},
        {"model": "org/other-model"},
    ):
        summary = summarize_report(report, **overrides)
        assert summary["target_name"] is None
        assert summary["resolved_delta_vs_target"] is None


def test_run_evaluation_moves_harness_report_from_eval_cwd(tmp_path, monkeypatch):
    eval_cwd = tmp_path / "eval"
    report_dir = tmp_path / "reports"
    eval_cwd.mkdir()
    report_payload = {
        "total_instances": 500,
        "submitted_instances": 1,
        "completed_instances": 1,
        "resolved_instances": 1,
        "unresolved_instances": 0,
        "empty_patch_instances": 0,
        "error_instances": 0,
    }

    def fake_run(command, cwd, check):
        harness_report = Path(cwd) / model_report_name("org/model", "run")
        harness_report.write_text(json.dumps(report_payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(evaluate.subprocess, "run", fake_run)
    monkeypatch.setattr(
        evaluate,
        "validate_evaluator_dataset_snapshot",
        lambda _args: {
            "dataset_revision": _args.dataset_revision,
            "task_rows_sha256": "a" * 64,
            "selected_task_count": 500,
        },
    )

    summary = evaluate.run_evaluation(
        args(
            eval_cwd=str(eval_cwd),
            report_dir=str(report_dir),
            run_id="run",
            instance_ids=[],
        )
    )

    expected_report = report_dir / model_report_name("org/model", "run")
    assert expected_report.is_file()
    assert not (eval_cwd / model_report_name("org/model", "run")).exists()
    assert summary["report_path"] == str(expected_report)
    assert summary["status"] == "ok"
    assert summary["resolved_instances"] == 1
    assert summary["task_rows_sha256"] == "a" * 64


def test_run_evaluation_rejects_a_failed_harness(tmp_path, monkeypatch):
    eval_cwd = tmp_path / "eval"
    eval_cwd.mkdir()

    monkeypatch.setattr(
        evaluate.subprocess,
        "run",
        lambda command, cwd, check: subprocess.CompletedProcess(command, 9),
    )
    monkeypatch.setattr(
        evaluate,
        "validate_evaluator_dataset_snapshot",
        lambda _args: {
            "dataset_revision": _args.dataset_revision,
            "task_rows_sha256": "a" * 64,
            "selected_task_count": 500,
        },
    )

    with pytest.raises(RuntimeError, match="exited with status 9"):
        evaluate.run_evaluation(
            args(
                eval_cwd=str(eval_cwd),
                report_dir=str(tmp_path / "reports"),
                instance_ids=[],
            )
        )


def test_snapshot_validation_rejects_current_rows_that_differ_from_pin(monkeypatch):
    pinned = [{"instance_id": "a", "problem_statement": "pinned"}]
    current = [{"instance_id": "a", "problem_statement": "current"}]

    def fake_load(_dataset, _split, *, revision):
        return pinned if revision else current

    monkeypatch.setattr(evaluate, "load_swebench_tasks", fake_load)

    with pytest.raises(RuntimeError, match="do not match the requested revision"):
        evaluate.validate_evaluator_dataset_snapshot(
            args(instance_ids=["a"], instance_ids_file=None)
        )


def test_snapshot_validation_records_matching_pinned_rows(monkeypatch):
    tasks = [{"instance_id": "a", "problem_statement": "same"}]
    monkeypatch.setattr(
        evaluate,
        "load_swebench_tasks",
        lambda _dataset, _split, *, revision: list(tasks),
    )

    snapshot = evaluate.validate_evaluator_dataset_snapshot(
        args(instance_ids=["a"], instance_ids_file=None)
    )

    assert snapshot["dataset_revision"] == args().dataset_revision
    assert snapshot["selected_task_count"] == 1
    assert len(snapshot["task_rows_sha256"]) == 64
