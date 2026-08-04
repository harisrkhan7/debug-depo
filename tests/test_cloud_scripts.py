from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLOUD = ROOT / "cloud"


def run_script(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    test_env = {
        "CLOUD_LOCAL_ENV": os.devnull,
        "GPU_IDS": "0,1,2,3,4,5,6,7",
        "NUM_PROCESSES": "8",
        "NUM_SHARDS": "8",
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


def recording_rsync(tmp_path: Path) -> tuple[Path, Path]:
    arguments_file = tmp_path / "rsync-arguments"
    executable = tmp_path / "rsync"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        '[[ "${1:-}" == "--help" ]] && exit 0\n'
        'printf "%s\\n" "$@" >"$RSYNC_ARGUMENTS_FILE"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, arguments_file


def copying_rclone(tmp_path: Path) -> tuple[Path, Path]:
    arguments_file = tmp_path / "rclone-arguments"
    executable = tmp_path / "rclone"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$@" >"$RCLONE_ARGUMENTS_FILE"\n'
        '[[ "$1" == "copy" ]]\n'
        'mkdir -p "$3"\n'
        'cp -R "$2/." "$3/"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, arguments_file


def test_cloud_shell_scripts_parse() -> None:
    for path in sorted(CLOUD.glob("*.sh")):
        completed = run_script("-n", str(path.relative_to(ROOT)))
        assert completed.returncode == 0, f"{path.name}: {completed.stderr}"


def test_swesmith_evaluation_timeout_defaults_to_600_and_allows_override(
    tmp_path: Path,
) -> None:
    arguments_file = tmp_path / "uv-arguments"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$@" >"$UV_ARGUMENTS_FILE"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = {
        "UV": str(fake_uv),
        "UV_ARGUMENTS_FILE": str(arguments_file),
        "PREDICTIONS_PATH": str(tmp_path / "predictions.jsonl"),
        "SUMMARY_OUTPUT": str(tmp_path / "summary.json"),
        "LOG_DIR": str(tmp_path / "logs"),
        "DRY_RUN": "1",
    }

    completed = run_script("scripts/evaluate_swesmith.sh", env=env)
    assert completed.returncode == 0, completed.stderr
    arguments = arguments_file.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("--timeout") + 1] == "600"

    completed = run_script(
        "scripts/evaluate_swesmith.sh",
        env={**env, "EVAL_TIMEOUT": "900"},
    )
    assert completed.returncode == 0, completed.stderr
    arguments = arguments_file.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("--timeout") + 1] == "900"


def test_shard_supervisor_detects_vllm_exit(tmp_path: Path) -> None:
    completed = run_script(
        "-c",
        (
            "source cloud/common.sh; "
            "sleep 30 & collector_pid=$!; "
            "supervise_collector 999999 \"$collector_pid\" 0 1 "
            "\"$TEST_ACTIVITY_PATH\"; status=$?; "
            "kill \"$collector_pid\"; wait \"$collector_pid\" 2>/dev/null; "
            "printf '%s\\n' \"$status\""
        ),
        env={
            "CLOUD_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "CLOUD_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
            "TEST_ACTIVITY_PATH": str(tmp_path / "activity.log"),
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "70"
    assert "vLLM exited" in completed.stderr


def test_shard_supervisor_detects_no_progress(tmp_path: Path) -> None:
    completed = run_script(
        "-c",
        (
            "source cloud/common.sh; "
            "sleep 30 & vllm_pid=$!; sleep 30 & collector_pid=$!; "
            "supervise_collector \"$vllm_pid\" \"$collector_pid\" 1 1 "
            "\"$TEST_ACTIVITY_PATH\"; status=$?; "
            "kill \"$vllm_pid\" \"$collector_pid\"; "
            "wait \"$vllm_pid\" \"$collector_pid\" 2>/dev/null; "
            "printf '%s\\n' \"$status\""
        ),
        env={
            "CLOUD_PERSISTENT_ROOT": str(tmp_path / "persistent"),
            "CLOUD_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
            "TEST_ACTIVITY_PATH": str(tmp_path / "activity.log"),
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "71"
    assert "no collector progress" in completed.stderr


def test_environment_discovers_gpus_and_separates_storage(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\nfor index in 0 1 2 3; do echo \"$index\"; done\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    persistent = tmp_path / "persistent"
    ephemeral = tmp_path / "ephemeral"

    completed = run_script(
        "-c",
        (
            "source cloud/env.sh; "
            'printf "%s\\n" "$GPU_IDS" "$NUM_SHARDS" "$NUM_PROCESSES" '
            '"$DEBUG_DEPO_SCRATCH" "$DEBUG_DEPO_SIF_ROOT" "$VLLM_IMAGE"'
        ),
        env={
            "CLOUD_PERSISTENT_ROOT": str(persistent),
            "CLOUD_EPHEMERAL_ROOT": str(ephemeral),
            "GPU_IDS": "",
            "NUM_PROCESSES": "",
            "NUM_SHARDS": "",
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
        },
    )

    assert completed.returncode == 0, completed.stderr
    gpu_ids, shards, processes, scratch, sifs, vllm = completed.stdout.splitlines()
    assert (gpu_ids, shards, processes) == ("0,1,2,3", "4", "4")
    assert Path(scratch).is_relative_to(persistent)
    assert Path(sifs).is_relative_to(ephemeral)
    assert Path(vllm).is_relative_to(ephemeral)


def test_supported_workflows_complete_dry_runs(tmp_path: Path) -> None:
    common_env = {
        "CLOUD_PERSISTENT_ROOT": str(tmp_path / "persistent"),
        "CLOUD_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
        "DRY_RUN": "1",
    }
    workflows = (
        ("cloud/run.sh", "pipeline", "verified"),
        ("cloud/run.sh", "pipeline", "swesmith"),
        ("cloud/run.sh", "smoke", "verified"),
        ("cloud/run.sh", "smoke", "swesmith"),
        ("cloud/run.sh", "recover-shard", "swesmith", "2"),
        ("cloud/run.sh", "trajectory-suite"),
    )

    for workflow in workflows:
        completed = run_script(*workflow, env=common_env)
        assert completed.returncode == 0, f"{' '.join(workflow)}: {completed.stderr}"


def test_invalid_cloud_configuration_fails_before_work_starts(tmp_path: Path) -> None:
    common_env = {
        "CLOUD_PERSISTENT_ROOT": str(tmp_path / "persistent"),
        "CLOUD_EPHEMERAL_ROOT": str(tmp_path / "ephemeral"),
        "DRY_RUN": "1",
    }
    invalid_runs = (
        (("cloud/build_cache.sh", "smoke"), {"CACHE_BUILD_MAX_WORKERS": "101"}),
        (("cloud/collect.sh", "verified"), {"MINI_SWE_RUNNER": "docker"}),
        (("cloud/evaluate.sh", "swesmith"), {"SWESMITH_EVAL_RUNTIME": "docker"}),
        (
            ("cloud/run.sh", "recover-shard", "swesmith", "2"),
            {
                "MINI_SWE_MODEL_TIMEOUT_SECONDS": "1200",
                "CLOUD_SHARD_STALL_TIMEOUT_SECONDS": "600",
            },
        ),
    )

    for command, invalid_env in invalid_runs:
        completed = run_script(*command, env={**common_env, **invalid_env})
        assert completed.returncode == 2, " ".join(command)


def test_cloud_transfers_pass_only_the_intended_trees_to_rsync(tmp_path: Path) -> None:
    fake_rsync, arguments_file = recording_rsync(tmp_path)
    transfer_env = {
        "DRY_RUN": "1",
        "RSYNC_ARGUMENTS_FILE": str(arguments_file),
        "RSYNC_BIN": str(fake_rsync),
    }

    pushed = run_script(
        "cloud/run.sh",
        "push",
        env={
            **transfer_env,
            "CLOUD_REMOTE": "ubuntu@example.test",
            "CLOUD_REMOTE_REPO_DIR": "/home/ubuntu/debug-depo",
        },
    )
    assert pushed.returncode == 0, pushed.stderr
    push_arguments = arguments_file.read_text(encoding="utf-8").splitlines()
    assert push_arguments[-2:] == [f"{ROOT}/", "ubuntu@example.test:/home/ubuntu/debug-depo/"]

    destination = tmp_path / "cloud-results"
    pulled = run_script(
        "cloud/run.sh",
        "pull",
        env={
            **transfer_env,
            "CLOUD_REMOTE": "ubuntu@example.test",
            "CLOUD_REMOTE_PERSISTENT_ROOT": "/lambda/nfs/Debug-Depo/debug-depo-persistent",
            "LOCAL_CLOUD_SCRATCH_DIR": str(destination),
        },
    )
    assert pulled.returncode == 0, pulled.stderr
    pull_arguments = arguments_file.read_text(encoding="utf-8").splitlines()
    assert pull_arguments[-2:] == [
        "ubuntu@example.test:/lambda/nfs/Debug-Depo/debug-depo-persistent/scratch/",
        f"{destination}/",
    ]
    assert not any("/sifs/" in argument for argument in pull_arguments)
    for pattern in (
        "**/experiments/**/*.safetensors",
        "**/experiments/**/*.bin",
        "**/experiments/**/*.pt",
        "**/experiments/**/*.pth",
        "**/experiments/**/*.ckpt",
        "**/experiments/**/*.gguf",
    ):
        assert pattern in pull_arguments

    pulled_with_models = run_script(
        "cloud/run.sh",
        "pull",
        env={
            **transfer_env,
            "CLOUD_REMOTE": "ubuntu@example.test",
            "CLOUD_REMOTE_PERSISTENT_ROOT": "/lambda/nfs/Debug-Depo/debug-depo-persistent",
            "LOCAL_CLOUD_SCRATCH_DIR": str(destination),
            "PULL_EXPERIMENT_MODELS": "1",
        },
    )
    assert pulled_with_models.returncode == 0, pulled_with_models.stderr
    pull_arguments = arguments_file.read_text(encoding="utf-8").splitlines()
    assert not any("experiments" in argument for argument in pull_arguments)


def test_sif_sync_round_trip_copies_the_complete_tree(tmp_path: Path) -> None:
    persistent = tmp_path / "Debug-Depo" / "debug-depo-persistent"
    first_vm = tmp_path / "vm-1"
    second_vm = tmp_path / "vm-2"
    expected = {
        "swebench/verified.sif": b"verified",
        "swesmith/smith.sif": b"smith",
        "vllm/vllm.sif": b"vllm",
    }
    for relative_path, content in expected.items():
        path = first_vm / "sifs" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    fake_rclone, arguments_file = copying_rclone(tmp_path)
    sync_env = {
        "RCLONE_ARGUMENTS_FILE": str(arguments_file),
        "RCLONE_BIN": str(fake_rclone),
    }

    persisted = run_script(
        "cloud/run.sh",
        "sifs",
        "persist",
        env={
            **sync_env,
            "CLOUD_PERSISTENT_ROOT": str(persistent),
            "CLOUD_EPHEMERAL_ROOT": str(first_vm),
        },
    )
    assert persisted.returncode == 0, persisted.stderr

    restored = run_script(
        "cloud/run.sh",
        "sifs",
        "restore",
        env={
            **sync_env,
            "CLOUD_PERSISTENT_ROOT": str(persistent),
            "CLOUD_EPHEMERAL_ROOT": str(second_vm),
        },
    )
    assert restored.returncode == 0, restored.stderr
    assert {
        relative_path: (second_vm / "sifs" / relative_path).read_bytes()
        for relative_path in expected
    } == expected
    arguments = arguments_file.read_text(encoding="utf-8").splitlines()
    assert arguments[:3] == ["copy", str(persistent / "sifs"), str(second_vm / "sifs")]
