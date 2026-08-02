from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLOUD = ROOT / "cloud"


def run_script(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    test_env = {
        "GPU_IDS": "0,1,2,3,4,5,6,7",
        "CLOUD_LOCAL_ENV": os.devnull,
        "NUM_SHARDS": "8",
        "NUM_PROCESSES": "8",
    }
    test_env.update(env or {})
    return subprocess.run(
        ["bash", *args],
        cwd=ROOT,
        env={**os.environ, **test_env},
        text=True,
        capture_output=True,
        check=False,
    )


def fake_nvidia_smi(tmp_path: Path, gpu_count: int) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "nvidia-smi"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        f"for ((index = 0; index < {gpu_count}; index++)); do echo \"$index\"; done\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return bin_dir


def test_all_cloud_shell_scripts_parse() -> None:
    for path in sorted(CLOUD.glob("*.sh")):
        completed = run_script("-n", str(path.relative_to(ROOT)))
        assert completed.returncode == 0, f"{path.name}: {completed.stderr}"


def test_gpu_defaults_are_discovered_from_nvidia_smi(tmp_path: Path) -> None:
    bin_dir = fake_nvidia_smi(tmp_path, gpu_count=4)
    completed = run_script(
        "-c",
        (
            "source cloud/env.sh; "
            'printf "%s|%s|%s|%s|%s|%s|%s|%s" '
            '"$NUM_SHARDS" "$ROLLOUT_WORKERS" "$CACHE_BUILD_MAX_WORKERS" '
            '"$EVAL_MAX_WORKERS" "$NUM_PROCESSES" "$GPU_IDS" '
            '"$CLOUD_GPU_SOURCE" "$APPTAINER_MKSQUASHFS_ARGS"'
        ),
        env={
            "CLOUD_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "CLOUD_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "GPU_IDS": "",
            "NUM_SHARDS": "",
            "NUM_PROCESSES": "",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "4|8|50|80|4|0,1,2,3|nvidia-smi|-processors 2"


def test_help_reports_current_collection_worker_default() -> None:
    completed = run_script("cloud/run.sh", "help")

    assert completed.returncode == 0, completed.stderr
    assert "collect with one shard per detected GPU" in completed.stdout
    assert "12 workers each" not in completed.stdout
    assert "storage" in completed.stdout
    assert "prefetch-model" in completed.stdout


def test_model_prefetch_dry_run_is_serial_and_disables_hf_transfer(
    tmp_path: Path,
) -> None:
    completed = run_script(
        "cloud/run.sh",
        "prefetch-model",
        env={
            "DRY_RUN": "1",
            "CLOUD_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "CLOUD_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "Kwai-Klear/Klear-AgentForge-8B-SFT" in completed.stdout
    assert "hf_transfer: disabled" in completed.stdout


def test_setup_prefetches_model_before_reporting_success() -> None:
    setup = (CLOUD / "setup.sh").read_text(encoding="utf-8")
    prefetch = (CLOUD / "prefetch_model.sh").read_text(encoding="utf-8")
    server = (CLOUD / "serve_vllm.sh").read_text(encoding="utf-8")

    assert 'bash "$CLOUD_DIR/prefetch_model.sh"' in setup
    assert "snapshot_download(repo_id=model)" in prefetch
    assert 'python3 - "$VLLM_MODEL"' in prefetch
    assert "APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=0" in prefetch
    assert 'APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"' in server


def test_collection_dry_run_creates_one_shard_per_detected_gpu(tmp_path: Path) -> None:
    bin_dir = fake_nvidia_smi(tmp_path, gpu_count=4)
    completed = run_script(
        "cloud/collect.sh",
        "verified",
        env={
            "DRY_RUN": "1",
            "CLOUD_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "CLOUD_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "GPU_IDS": "",
            "NUM_SHARDS": "",
            "NUM_PROCESSES": "",
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert "shards/GPUs:         4 / 0,1,2,3" in completed.stdout
    assert "GPU selection:       nvidia-smi" in completed.stdout
    assert "total worker slots:  32" in completed.stdout


def test_validation_runs_each_supplied_task_once_and_analyzes(tmp_path: Path) -> None:
    task_ids = tmp_path / "validation_ids.txt"
    task_ids.write_text(
        "".join(f"repo.task-{index}\n" for index in range(8)),
        encoding="utf-8",
    )
    completed = run_script(
        "cloud/run.sh",
        "validate",
        env={
            "DRY_RUN": "1",
            "RUN_NAME": "validation-8-dmpo-test",
            "TASK_IDS_FILE": str(task_ids),
            "MODEL_PATH": "/models/dmpo-test",
            "CLOUD_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "CLOUD_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "Cloud deterministic SWE-smith validation" in completed.stdout
    assert "model:             /models/dmpo-test" in completed.stdout
    assert f"task IDs:          {task_ids}" in completed.stdout
    assert "expected tasks:    8" in completed.stdout
    assert "trajectories/task: 1" in completed.stdout
    assert "temperature:       0.0" in completed.stdout
    assert "context length:    32768" in completed.stdout
    assert "shards/GPUs:         8 / 0,1,2,3,4,5,6,7" in completed.stdout
    assert "workers per shard:   8" in completed.stdout
    assert "expected tasks:  8" in completed.stdout
    assert "would analyze swesmith run" in completed.stdout


def test_swesmith_pipeline_defaults_to_initial_1000_task_run(tmp_path: Path) -> None:
    completed = run_script(
        "cloud/run.sh",
        "pipeline",
        "swesmith",
        env={
            "DRY_RUN": "1",
            "CLOUD_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "CLOUD_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "swesmith-train-1000" in completed.stdout
    assert "expected tasks:      1000" in completed.stdout
    assert "swesmith_train_1000_instance_ids.txt" in completed.stdout
    assert "workers per shard:   8" in completed.stdout
    assert "workers:         80" in completed.stdout
    assert "would analyze swesmith run" in completed.stdout


def test_trajectory_suite_dry_run_orders_collection_before_validation(tmp_path: Path) -> None:
    completed = run_script(
        "cloud/run.sh",
        "trajectory-suite",
        env={
            "DRY_RUN": "1",
            "TRAIN_RUN_NAME": "swesmith-suite-test",
            "CLOUD_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "CLOUD_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
        },
    )
    assert completed.returncode == 0, completed.stderr
    markers = [
        "swesmith-suite-test\n",
        "swesmith-suite-test-sft-validation-100",
        "swesmith-suite-test-sft-validation-200",
        "swesmith-suite-test-sft-validation-500",
    ]
    positions = [completed.stdout.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert completed.stdout.count("trajectories/task: 1") == 3
    assert completed.stdout.count("workers:         100") == 4


def test_cache_workers_must_stay_in_requested_range(tmp_path: Path) -> None:
    completed = run_script(
        "cloud/build_cache.sh",
        "smoke",
        env={
            "CACHE_BUILD_MAX_WORKERS": "101",
            "DRY_RUN": "1",
            "CLOUD_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "CLOUD_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
        },
    )
    assert completed.returncode == 2
    assert "at most 100" in completed.stderr


def test_verified_smoke_dry_run_exercises_all_gpus(tmp_path: Path) -> None:
    completed = run_script(
        "cloud/run.sh",
        "smoke",
        "verified",
        env={
            "DRY_RUN": "1",
            "CLOUD_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "CLOUD_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "agentforge-verified-cloud-smoke" in completed.stdout
    assert "expected tasks:      8" in completed.stdout
    assert "shards/GPUs:         8 / 0,1,2,3,4,5,6,7" in completed.stdout
    assert "workers per shard:   8" in completed.stdout
    assert "expected shards: 8" in completed.stdout
    assert "would analyze verified run" in completed.stdout


def test_swesmith_smoke_dry_run_exercises_all_gpus(tmp_path: Path) -> None:
    completed = run_script(
        "cloud/run.sh",
        "smoke",
        "swesmith",
        env={
            "DRY_RUN": "1",
            "CLOUD_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "CLOUD_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "swesmith-cloud-smoke" in completed.stdout
    assert "expected tasks:      8" in completed.stdout
    assert "shards/GPUs:         8 / 0,1,2,3,4,5,6,7" in completed.stdout
    assert "workers per shard:   8" in completed.stdout
    assert "expected shards: 8" in completed.stdout
    assert "task IDs:        merged predictions" in completed.stdout
    assert "would analyze swesmith run" in completed.stdout


def test_push_dry_run_uses_lambda_destination() -> None:
    completed = run_script(
        "cloud/run.sh",
        "push",
        env={
            "DRY_RUN": "1",
            "CLOUD_REMOTE": "ubuntu@example.test",
            "CLOUD_REMOTE_REPO_DIR": "/home/ubuntu/debug-depo",
            "RSYNC_BIN": "/usr/bin/true",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "ubuntu@example.test:/home/ubuntu/debug-depo/" in completed.stdout
    assert "cloud/local.env are not transferred" in completed.stdout


def test_pull_dry_run_creates_local_cloud_folder(tmp_path: Path) -> None:
    destination = tmp_path / "scratch" / "cloud"
    completed = run_script(
        "cloud/run.sh",
        "pull",
        env={
            "DRY_RUN": "1",
            "CLOUD_REMOTE": "ubuntu@example.test",
            "CLOUD_REMOTE_PERSISTENT_ROOT": "/lambda/nfs/debug-depo/debug-depo-persistent",
            "LOCAL_CLOUD_SCRATCH_DIR": str(destination),
            "RSYNC_BIN": "/usr/bin/true",
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert destination.is_dir()
    assert (
        "ubuntu@example.test:/lambda/nfs/debug-depo/debug-depo-persistent/scratch/"
        in completed.stdout
    )
    assert str(destination) in completed.stdout
    assert "does not copy" in completed.stdout
    assert "ephemeral caches, SIFs, runtime state, or temporary files" in completed.stdout


def test_preference_wrapper_maps_stage_specific_hyperparameters() -> None:
    script = (CLOUD / "preference.sh").read_text(encoding="utf-8")
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
            "source cloud/env.sh; "
            'printf "%s\\n" '
            '"$DEBUG_DEPO_SCRATCH" "$DEBUG_DEPO_CACHE_ROOT" '
            '"$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" '
            '"$SWEBENCH_APPTAINER_SIF_DIR" "$SWESMITH_APPTAINER_SIF_DIR" '
            '"$VLLM_IMAGE"'
        ),
        env={
            "CLOUD_PERSISTENT_ROOT": str(persistent),
            "CLOUD_EPHEMERAL_ROOT": str(ephemeral),
        },
    )
    assert completed.returncode == 0, completed.stderr
    paths = [Path(value) for value in completed.stdout.splitlines()]
    assert paths[0] == persistent / "scratch"
    assert all(path.is_relative_to(ephemeral) for path in paths[1:])


def test_legacy_hyperstack_storage_aliases_are_supported(tmp_path: Path) -> None:
    persistent = tmp_path / "legacy-persistent"
    ephemeral = tmp_path / "legacy-ephemeral"
    completed = run_script(
        "-c",
        (
            "source cloud/env.sh; "
            'printf "%s|%s|%s|%s" '
            '"$CLOUD_PERSISTENT_ROOT" "$CLOUD_EPHEMERAL_ROOT" '
            '"$HYPERSTACK_PERSISTENT_ROOT" "$HYPERSTACK_EPHEMERAL_ROOT"'
        ),
        env={
            "CLOUD_PERSISTENT_ROOT": "",
            "CLOUD_EPHEMERAL_ROOT": "",
            "HYPERSTACK_PERSISTENT_ROOT": str(persistent),
            "HYPERSTACK_EPHEMERAL_ROOT": str(ephemeral),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"{persistent}|{ephemeral}|{persistent}|{ephemeral}"


def test_lambda_local_env_is_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "cloud/local.env" in gitignore.splitlines()


def test_storage_check_is_read_only() -> None:
    script = (CLOUD / "check_storage.sh").read_text(encoding="utf-8")
    for destructive_command in ("mkfs", "mount ", "/etc/fstab", "blkid"):
        assert destructive_command not in script


def test_lambda_virtiofs_mounts_are_supported(tmp_path: Path) -> None:
    completed = run_script(
        "-c",
        "source cloud/common.sh; is_lambda_persistent_fstype virtiofs",
        env={
            "CLOUD_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "CLOUD_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
        },
    )
    assert completed.returncode == 0, completed.stderr


def test_cloud_runtime_never_invokes_docker() -> None:
    forbidden = re.compile(
        r"\brequire_command\s+docker\b|\bdocker\s+(?:info|login|pull|rm|run)\b"
        r"|VLLM_DOCKER_IMAGE"
    )
    offenders: list[str] = []
    for path in sorted(CLOUD.glob("*.sh")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if forbidden.search(line):
                offenders.append(f"{path.name}:{line_number}: {line}")
    assert not offenders, "\n".join(offenders)


def test_vllm_server_uses_apptainer_with_nvidia_support() -> None:
    script = (CLOUD / "serve_vllm.sh").read_text(encoding="utf-8")
    assert "require_command apptainer" in script
    assert 'apptainer inspect "$VLLM_IMAGE"' in script
    assert "apptainer exec --nv" in script
    assert 'APPTAINERENV_CUDA_VISIBLE_DEVICES="$GPU_ID"' in script


def test_setup_pulls_vllm_sif_atomically() -> None:
    script = (CLOUD / "setup.sh").read_text(encoding="utf-8")
    assert 'apptainer inspect "$VLLM_IMAGE"' in script
    assert 'apptainer pull "$temporary_vllm_image"' in script
    assert 'mv -f -- "$temporary_vllm_image" "$VLLM_IMAGE"' in script


def test_vllm_sif_may_use_persistent_or_ephemeral_storage() -> None:
    common = (CLOUD / "common.sh").read_text(encoding="utf-8")
    assert '"$vllm_image_device" != "$persistent_device"' in common
    assert '"$vllm_image_device" != "$ephemeral_device"' in common


def test_parallel_cache_build_disables_shared_oci_layer_cache() -> None:
    script = (CLOUD / "build_cache.sh").read_text(encoding="utf-8")
    assert 'APPTAINER_DISABLE_CACHE="${APPTAINER_DISABLE_CACHE:-1}"' in script


def test_collection_rejects_non_apptainer_task_runtime(tmp_path: Path) -> None:
    completed = run_script(
        "cloud/collect.sh",
        "verified",
        env={
            "DRY_RUN": "1",
            "CLOUD_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "CLOUD_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
            "MINI_SWE_RUNNER": "docker",
        },
    )
    assert completed.returncode == 2
    assert "requires MINI_SWE_RUNNER=singularity" in completed.stderr


def test_evaluation_rejects_non_apptainer_runtime(tmp_path: Path) -> None:
    completed = run_script(
        "cloud/evaluate.sh",
        "swesmith",
        env={
            "DRY_RUN": "1",
            "CLOUD_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "CLOUD_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
            "SWESMITH_EVAL_RUNTIME": "docker",
        },
    )
    assert completed.returncode == 2
    assert "requires SWESMITH_EVAL_RUNTIME=apptainer" in completed.stderr
