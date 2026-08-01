from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HYPERSTACK = ROOT / "hyperstack"


def run_script(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", *args],
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        text=True,
        capture_output=True,
        check=False,
    )


def test_all_hyperstack_shell_scripts_parse() -> None:
    for path in sorted(HYPERSTACK.glob("*.sh")):
        completed = run_script("-n", str(path.relative_to(ROOT)))
        assert completed.returncode == 0, f"{path.name}: {completed.stderr}"


def test_eight_gpu_defaults_are_exposed(tmp_path: Path) -> None:
    completed = run_script(
        "-c",
        (
            "source hyperstack/env.sh; "
            'printf "%s|%s|%s|%s|%s|%s|%s" '
            '"$NUM_SHARDS" "$ROLLOUT_WORKERS" "$CACHE_BUILD_MAX_WORKERS" '
            '"$EVAL_MAX_WORKERS" "$NUM_PROCESSES" "$GPU_IDS" '
            '"$APPTAINER_MKSQUASHFS_ARGS"'
        ),
        env={
            "HYPERSTACK_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "HYPERSTACK_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "8|8|50|80|8|0,1,2,3,4,5,6,7|-processors 2"


def test_help_reports_current_collection_worker_default() -> None:
    completed = run_script("hyperstack/run.sh", "help")

    assert completed.returncode == 0, completed.stderr
    assert "collect with 8 GPU shards, 8 workers each" in completed.stdout
    assert "12 workers each" not in completed.stdout


def test_validation_pipeline_dry_run_uses_holdout_and_all_shards(tmp_path: Path) -> None:
    completed = run_script(
        "hyperstack/run.sh",
        "validate",
        env={
            "DRY_RUN": "1",
            "HYPERSTACK_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "HYPERSTACK_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "swesmith-validation-500" in completed.stdout
    assert "swesmith_validation_500_instance_ids.txt" in completed.stdout
    assert "shards/GPUs:         8 / 0,1,2,3,4,5,6,7" in completed.stdout
    assert "workers per shard:   8" in completed.stdout


def test_swesmith_pipeline_defaults_to_initial_1000_task_run(tmp_path: Path) -> None:
    completed = run_script(
        "hyperstack/run.sh",
        "pipeline",
        "swesmith",
        env={
            "DRY_RUN": "1",
            "HYPERSTACK_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "HYPERSTACK_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "swesmith-train-1000" in completed.stdout
    assert "expected tasks:      1000" in completed.stdout
    assert "swesmith_train_1000_instance_ids.txt" in completed.stdout
    assert "workers per shard:   8" in completed.stdout
    assert "workers:         80" in completed.stdout
    assert "would analyze swesmith run" in completed.stdout


def test_cache_workers_must_stay_in_requested_range(tmp_path: Path) -> None:
    completed = run_script(
        "hyperstack/build_cache.sh",
        "smoke",
        env={
            "CACHE_BUILD_MAX_WORKERS": "101",
            "DRY_RUN": "1",
            "HYPERSTACK_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "HYPERSTACK_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
        },
    )
    assert completed.returncode == 2
    assert "at most 100" in completed.stderr


def test_verified_smoke_dry_run_exercises_all_gpus(tmp_path: Path) -> None:
    completed = run_script(
        "hyperstack/run.sh",
        "smoke",
        "verified",
        env={
            "DRY_RUN": "1",
            "HYPERSTACK_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "HYPERSTACK_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "agentforge-verified-hyperstack-smoke" in completed.stdout
    assert "expected tasks:      8" in completed.stdout
    assert "shards/GPUs:         8 / 0,1,2,3,4,5,6,7" in completed.stdout
    assert "workers per shard:   8" in completed.stdout
    assert "expected shards: 8" in completed.stdout
    assert "would analyze verified run" in completed.stdout


def test_swesmith_smoke_dry_run_exercises_all_gpus(tmp_path: Path) -> None:
    completed = run_script(
        "hyperstack/run.sh",
        "smoke",
        "swesmith",
        env={
            "DRY_RUN": "1",
            "HYPERSTACK_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "HYPERSTACK_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "swesmith-hyperstack-smoke" in completed.stdout
    assert "expected tasks:      8" in completed.stdout
    assert "shards/GPUs:         8 / 0,1,2,3,4,5,6,7" in completed.stdout
    assert "workers per shard:   8" in completed.stdout
    assert "expected shards: 8" in completed.stdout
    assert "task IDs:        merged predictions" in completed.stdout
    assert "would analyze swesmith run" in completed.stdout


def test_push_dry_run_uses_hyperstack_destination() -> None:
    completed = run_script(
        "hyperstack/run.sh",
        "push",
        env={
            "DRY_RUN": "1",
            "HYPERSTACK_REMOTE": "root@example.test",
            "HYPERSTACK_REMOTE_REPO_DIR": "/root/debug-depo",
            "RSYNC_BIN": "/usr/bin/true",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "root@example.test:/root/debug-depo/" in completed.stdout
    assert "hyperstack/local.env are not transferred" in completed.stdout


def test_pull_dry_run_creates_local_hyperstack_folder(tmp_path: Path) -> None:
    destination = tmp_path / "scratch" / "hyperstack"
    completed = run_script(
        "hyperstack/run.sh",
        "pull",
        env={
            "DRY_RUN": "1",
            "HYPERSTACK_REMOTE": "root@example.test",
            "HYPERSTACK_REMOTE_PERSISTENT_ROOT": "/root/debug-depo-persistent",
            "LOCAL_HYPERSTACK_SCRATCH_DIR": str(destination),
            "RSYNC_BIN": "/usr/bin/true",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert destination.is_dir()
    assert "root@example.test:/root/debug-depo-persistent/scratch/" in completed.stdout
    assert str(destination) in completed.stdout
    assert "does not copy" in completed.stdout
    assert "ephemeral caches, SIFs, runtime state, or temporary files" in completed.stdout


def test_preference_wrapper_maps_stage_specific_hyperparameters() -> None:
    script = (HYPERSTACK / "preference.sh").read_text(encoding="utf-8")
    expected_mappings = {
        "DMPO_LEARNING_RATE": "LEARNING_RATE=$DMPO_LEARNING_RATE",
        "DMPO_BETA": "BETA=$DMPO_BETA",
        "DMPO_GAMMA": "GAMMA=$DMPO_GAMMA",
        "DEPO_LEARNING_RATE": "LEARNING_RATE=$DEPO_LEARNING_RATE",
        "DEPO_BETA": "BETA=$DEPO_BETA",
    }
    for source, assignment in expected_mappings.items():
        assert f'${{{source}:-}}' in script
        assert assignment in script


def test_storage_defaults_keep_runs_persistent_and_caches_ephemeral(
    tmp_path: Path,
) -> None:
    persistent = tmp_path / "persistent"
    ephemeral = tmp_path / "ephemeral"
    completed = run_script(
        "-c",
        (
            "source hyperstack/env.sh; "
            'printf "%s\\n" '
            '"$DEBUG_DEPO_SCRATCH" "$DEBUG_DEPO_CACHE_ROOT" '
            '"$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" '
            '"$SWEBENCH_APPTAINER_SIF_DIR" "$SWESMITH_APPTAINER_SIF_DIR" '
            '"$VLLM_IMAGE"'
        ),
        env={
            "HYPERSTACK_PERSISTENT_ROOT": str(persistent),
            "HYPERSTACK_EPHEMERAL_ROOT": str(ephemeral),
        },
    )
    assert completed.returncode == 0, completed.stderr
    paths = [Path(value) for value in completed.stdout.splitlines()]
    assert paths[0] == persistent / "scratch"
    assert all(path.is_relative_to(ephemeral) for path in paths[1:])


def test_hyperstack_runtime_never_invokes_docker() -> None:
    forbidden = re.compile(
        r"\brequire_command\s+docker\b|\bdocker\s+(?:info|login|pull|rm|run)\b"
        r"|VLLM_DOCKER_IMAGE"
    )
    offenders: list[str] = []
    for path in sorted(HYPERSTACK.glob("*.sh")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if forbidden.search(line):
                offenders.append(f"{path.name}:{line_number}: {line}")
    assert not offenders, "\n".join(offenders)


def test_vllm_server_uses_apptainer_with_nvidia_support() -> None:
    script = (HYPERSTACK / "serve_vllm.sh").read_text(encoding="utf-8")
    assert "require_command apptainer" in script
    assert 'apptainer inspect "$VLLM_IMAGE"' in script
    assert "apptainer exec --nv" in script
    assert 'APPTAINERENV_CUDA_VISIBLE_DEVICES="$GPU_ID"' in script


def test_setup_pulls_vllm_sif_atomically() -> None:
    script = (HYPERSTACK / "setup.sh").read_text(encoding="utf-8")
    assert 'apptainer inspect "$VLLM_IMAGE"' in script
    assert 'apptainer pull "$temporary_vllm_image"' in script
    assert 'mv -f -- "$temporary_vllm_image" "$VLLM_IMAGE"' in script


def test_vllm_sif_may_use_persistent_or_ephemeral_storage() -> None:
    common = (HYPERSTACK / "common.sh").read_text(encoding="utf-8")
    assert '"$vllm_image_device" != "$persistent_device"' in common
    assert '"$vllm_image_device" != "$ephemeral_device"' in common


def test_parallel_cache_build_disables_shared_oci_layer_cache() -> None:
    script = (HYPERSTACK / "build_cache.sh").read_text(encoding="utf-8")
    assert 'APPTAINER_DISABLE_CACHE="${APPTAINER_DISABLE_CACHE:-1}"' in script


def test_collection_rejects_non_apptainer_task_runtime(tmp_path: Path) -> None:
    completed = run_script(
        "hyperstack/collect.sh",
        "verified",
        env={
            "DRY_RUN": "1",
            "HYPERSTACK_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "HYPERSTACK_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
            "MINI_SWE_RUNNER": "docker",
        },
    )
    assert completed.returncode == 2
    assert "requires MINI_SWE_RUNNER=singularity" in completed.stderr


def test_evaluation_rejects_non_apptainer_runtime(tmp_path: Path) -> None:
    completed = run_script(
        "hyperstack/evaluate.sh",
        "swesmith",
        env={
            "DRY_RUN": "1",
            "HYPERSTACK_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "HYPERSTACK_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
            "SWESMITH_EVAL_RUNTIME": "docker",
        },
    )
    assert completed.returncode == 2
    assert "requires SWESMITH_EVAL_RUNTIME=apptainer" in completed.stderr
