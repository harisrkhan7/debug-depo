import json
from types import SimpleNamespace

import pytest

import debug_depo.evaluate_apptainer as evaluate_apptainer
from debug_depo.evaluate_apptainer import (
    DEFAULT_IMAGE_TEMPLATE,
    EVAL_BIND_DIR,
    _evaluation_cache_key,
    build_parser,
    build_apptainer_command,
    image_uri_from_template,
    run_apptainer_evaluation,
    run_instance,
)
from debug_depo.apptainer_cache import pull_sif_if_missing, sif_path_for_image
from debug_depo.utils import write_json


def test_default_epoch_image_template_uses_instance_id():
    image_uri = image_uri_from_template(
        DEFAULT_IMAGE_TEMPLATE,
        "astropy__astropy-12907",
        image_key="sweb.eval.x86_64.astropy__astropy-12907:latest",
    )

    assert image_uri == (
        "docker://ghcr.io/epoch-research/"
        "swe-bench.eval.x86_64.astropy__astropy-12907:latest"
    )


def test_image_template_accepts_tag_inside_placeholder():
    image_uri = image_uri_from_template(
        "docker://ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id:latest}",
        "astropy__astropy-12907",
    )

    assert image_uri == (
        "docker://ghcr.io/epoch-research/"
        "swe-bench.eval.x86_64.astropy__astropy-12907:latest"
    )


def test_sif_path_uses_stable_image_filename(tmp_path):
    image_uri = (
        "docker://ghcr.io/epoch-research/"
        "swe-bench.eval.x86_64.astropy__astropy-12907:latest"
    )

    assert sif_path_for_image(tmp_path, image_uri) == (
        tmp_path
        / "docker_ghcr.io_epoch-research_swe-bench.eval.x86_64."
        "astropy__astropy-12907_latest.sif"
    )


def test_apptainer_exec_command_binds_log_dir(tmp_path):
    command = build_apptainer_command(
        tmp_path / "image.sif",
        tmp_path / "logs",
        extra_args=["--containall"],
    )

    assert command[:3] == ["apptainer", "exec", "--writable-tmpfs"]
    assert "--containall" in command
    assert f"{(tmp_path / 'logs').resolve()}:{EVAL_BIND_DIR}" in command
    assert command[-3:] == [
        str(tmp_path / "image.sif"),
        "/bin/bash",
        f"{EVAL_BIND_DIR}/run_apptainer_eval.sh",
    ]


def test_pull_sif_if_missing_dry_run_returns_pull_command(tmp_path):
    sif_path = tmp_path / "astropy__astropy-12907.sif"

    command = pull_sif_if_missing(
        sif_path=sif_path,
        image_uri="docker://example/image:latest",
        cache_dir=tmp_path / "cache",
        dry_run=True,
    )

    assert command == ["apptainer", "pull", str(sif_path), "docker://example/image:latest"]
    assert not sif_path.exists()


def test_cached_report_is_keyed_by_prediction_and_instance_content(
    tmp_path,
    monkeypatch,
):
    args = build_parser().parse_args(
        [
            "--dataset",
            "dataset",
            "--predictions-path",
            "predictions.jsonl",
            "--model",
            "model",
            "--run-id",
            "run",
            "--report-dir",
            str(tmp_path / "reports"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--sif-dir",
            str(tmp_path / "sifs"),
            "--dry-run",
        ]
    )
    args.dataset_revision = "dataset-commit"
    instance = {"instance_id": "repo__repo-1", "problem_statement": "first"}
    prediction = {
        "instance_id": "repo__repo-1",
        "model_name_or_path": "model",
        "model_patch": "first patch",
    }
    test_spec = SimpleNamespace(
        instance_image_key="image-key",
        eval_script="run tests",
        get_test_cmd=lambda _instance, f2p_only: ("run tests", None),
        get_test_files=lambda _instance: ([], []),
    )
    monkeypatch.setattr(
        evaluate_apptainer,
        "make_test_spec",
        lambda _instance: test_spec,
    )
    log_dir = tmp_path / "logs" / "run" / "model" / "repo__repo-1"
    log_dir.mkdir(parents=True)
    report_path = log_dir / "report.json"
    cache_key_path = log_dir / "cache_key.json"
    write_json(report_path, {"repo__repo-1": {"resolved": True}})
    write_json(
        cache_key_path,
        _evaluation_cache_key(instance, prediction, args, test_spec),
    )
    assert json.loads(cache_key_path.read_text())["dataset_revision"] == (
        "dataset-commit"
    )

    assert run_instance(instance, prediction, args)["status"] == "cached_report"
    assert run_instance(
        instance,
        {**prediction, "model_patch": "changed patch"},
        args,
    )["status"] == "dry_run"
    assert run_instance(
        {**instance, "problem_statement": "changed"},
        prediction,
        args,
    )["status"] == "dry_run"

    args.dry_run = False
    assert run_instance(
        instance,
        {**prediction, "model_patch": ""},
        args,
    )["status"] == "empty_patch"
    assert not report_path.exists()
    assert not cache_key_path.exists()


def test_require_complete_rejects_evaluation_infrastructure_errors(
    tmp_path,
    monkeypatch,
):
    args = build_parser().parse_args(
        [
            "--dataset",
            "dataset",
            "--dataset-revision",
            "dataset-commit",
            "--predictions-path",
            str(tmp_path / "predictions.jsonl"),
            "--model",
            "model",
            "--run-id",
            "run",
            "--report-dir",
            str(tmp_path / "reports"),
            "--summary-output",
            str(tmp_path / "summary.json"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--sif-dir",
            str(tmp_path / "sifs"),
            "--require-complete",
        ]
    )
    prediction = {
        "instance_id": "repo__repo-1",
        "model_name_or_path": "model",
        "model_patch": "patch",
    }
    instance = {"instance_id": "repo__repo-1"}
    loaded_revision = []
    monkeypatch.setattr(
        evaluate_apptainer,
        "get_predictions_from_file",
        lambda *_args, **_kwargs: [prediction],
    )
    monkeypatch.setattr(
        evaluate_apptainer,
        "load_swebench_tasks",
        lambda *_args, revision=None, **_kwargs: (
            loaded_revision.append(revision) or [instance]
        ),
    )
    monkeypatch.setattr(
        evaluate_apptainer,
        "run_instance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("container unavailable")
        ),
    )

    with pytest.raises(RuntimeError, match="infrastructure outcomes"):
        run_apptainer_evaluation(args)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert loaded_revision == ["dataset-commit"]
    assert summary["dataset_revision"] == "dataset-commit"
    assert summary["scored_instances"] == 0
    assert summary["status_ids"] == {"error": ["repo__repo-1"]}
