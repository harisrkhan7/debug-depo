from debug_depo.evaluate_apptainer import (
    DEFAULT_IMAGE_TEMPLATE,
    EVAL_BIND_DIR,
    build_apptainer_command,
    image_uri_from_template,
    pull_sif_if_missing,
    sif_path_for_instance,
)


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


def test_sif_path_uses_stable_instance_filename(tmp_path):
    assert sif_path_for_instance(tmp_path, "astropy__astropy-12907") == (
        tmp_path / "astropy__astropy-12907.sif"
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
