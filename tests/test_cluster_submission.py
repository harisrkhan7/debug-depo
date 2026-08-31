from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def base_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {
        name: os.environ[name] for name in ("HOME", "PATH", "TMPDIR", "USER") if name in os.environ
    }
    env.update(
        {
            "DEBUG_DEPO_SCRATCH": str(tmp_path / "scratch"),
            **overrides,
        }
    )
    return env


def recording_command(tmp_path: Path, name: str) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / f"{name}.log"
    executable = bin_dir / name
    help_guard = '[[ "${1:-}" == "--help" ]] && exit 0\n' if name == "rsync" else ""
    qsub_response = (
        'job_number="$(wc -l <"$COMMAND_LOG" | tr -d "[:space:]")"\n'
        'printf "%s.server\\n" "$job_number"\n'
        if name == "qsub"
        else ""
    )
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"{help_guard}"
        "first=1\n"
        'for argument in "$@"; do\n'
        '  ((first)) || printf "\\t" >>"$COMMAND_LOG"\n'
        '  printf "%s" "$argument" >>"$COMMAND_LOG"\n'
        "  first=0\n"
        "done\n"
        'printf "\\n" >>"$COMMAND_LOG"\n'
        f"{qsub_response}",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, log


def run_submission(
    tmp_path: Path,
    script: str,
    **overrides: str,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    qsub, log = recording_command(tmp_path, "qsub")
    env = base_env(
        tmp_path,
        COMMAND_LOG=str(log),
        PATH=f"{qsub.parent}:{os.environ['PATH']}",
        **overrides,
    )
    completed = subprocess.run(
        ["bash", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    calls = (
        [line.split("\t") for line in log.read_text(encoding="utf-8").splitlines()]
        if log.exists()
        else []
    )
    return completed, calls


def joined(call: list[str]) -> str:
    return " ".join(call)


def test_verified_submission_chains_collection_evaluation_and_analysis(tmp_path: Path) -> None:
    completed, calls = run_submission(
        tmp_path,
        "cluster/submit_verified_full.sh",
        AFTEROK_JOB_ID="upstream.server",
    )

    assert completed.returncode == 0, completed.stderr
    assert len(calls) == 3
    assert calls[0][-1] == "cluster/pbs/collect_verified_full.pbs"
    assert "RUN_NAME=agentforge-verified-full" in joined(calls[0])
    assert "EXPECTED_COUNT=500" in joined(calls[1])
    assert "depend=afterok:1.server" in joined(calls[1])
    assert calls[2][-1] == "cluster/pbs/analyze_verified.pbs"
    assert "depend=afterok:2.server" in joined(calls[2])


def test_swesmith_full_submits_the_tracked_training_split(tmp_path: Path) -> None:
    completed, calls = run_submission(
        tmp_path,
        "cluster/submit_swesmith_full.sh",
        AFTEROK_JOB_ID="upstream.server",
    )

    assert completed.returncode == 0, completed.stderr
    assert len(calls) == 3
    collection = joined(calls[0])
    evaluation = joined(calls[1])
    assert calls[0][-1] == "cluster/pbs/collect_swesmith_array.pbs"
    assert "TASK_IDS_FILE=data/splits/swesmith_train_1000_instance_ids.txt" in collection
    assert "TASK_IDS_FILE=data/splits/swesmith_train_1000_instance_ids.txt" in evaluation
    assert "EXPECTED_TASKS=1000" in collection
    assert "RUNS_PER_TEMPERATURE=2" in collection
    assert "TOTAL_SAMPLES=4" in collection
    assert "-J 0-9" in collection
    assert "depend=afterok:1.server" in evaluation
    assert "depend=afterok:2.server" in joined(calls[2])


def test_swesmith_bounded_full_evaluation_uses_merged_membership(tmp_path: Path) -> None:
    completed, calls = run_submission(
        tmp_path,
        "cluster/submit_swesmith_full.sh",
        TASK_LIMIT="10",
    )

    assert completed.returncode == 0, completed.stderr
    assert len(calls) == 3
    assert "TASK_IDS_FILE=data/splits/swesmith_train_1000_instance_ids.txt" in joined(calls[0])
    assert "EXPECTED_TASKS=10" in joined(calls[1])
    assert "TASK_IDS_FILE=data/splits/swesmith_train_1000_instance_ids.txt" not in joined(calls[1])
    assert "TASK_IDS_FILE=" not in joined(calls[1])


def test_swesmith_pilot_cache_pipeline_is_one_dependency_chain(tmp_path: Path) -> None:
    completed, calls = run_submission(
        tmp_path,
        "cluster/submit_swesmith_pilot_with_cache.sh",
    )

    assert completed.returncode == 0, completed.stderr
    assert len(calls) == 4
    assert calls[0][-1] == "cluster/pbs/build_apptainer_cache_full.pbs"
    assert "CACHE_BUILD_DATASETS=swesmith" in joined(calls[0])
    assert "SWESMITH_CACHE_LIMIT=30" in joined(calls[0])
    assert "depend=afterok:1.server" in joined(calls[1])
    assert "depend=afterok:2.server" in joined(calls[2])
    assert "depend=afterok:3.server" in joined(calls[3])


def test_preference_training_chains_data_training_and_packaging(tmp_path: Path) -> None:
    completed, calls = run_submission(
        tmp_path,
        "cluster/submit_preference_training.sh",
        EXPERIMENT_ARM="dmpo-depo",
        AFTEROK_JOB_ID="upstream.server",
        PREFERENCE_DATA_MODE="build",
        SUBMIT_MODEL_EVALUATIONS="0",
    )

    assert completed.returncode == 0, completed.stderr
    assert "Preference training PBS logs:" in completed.stdout
    assert [call[-1] for call in calls] == [
        "cluster/pbs/build_dmpo_pairs.pbs",
        "cluster/pbs/build_depo_data.pbs",
        "cluster/pbs/train_preference.pbs",
        "cluster/pbs/package_preference.pbs",
        "cluster/pbs/train_preference.pbs",
        "cluster/pbs/package_preference.pbs",
    ]
    assert "depend=afterok:1.server:2.server" in joined(calls[2])
    assert "depend=afterok:3.server" in joined(calls[3])
    assert "depend=afterok:4.server" in joined(calls[4])
    assert "depend=afterok:5.server" in joined(calls[5])


def test_invalid_submission_configuration_never_calls_qsub(tmp_path: Path) -> None:
    cases = (
        (
            "cluster/submit_verified_full.sh",
            {"EXPECTED_COUNT": "5", "NUM_SHARDS": "10"},
        ),
        ("cluster/submit_swesmith_pilot.sh", {"MINI_SWE_RUNNER": "pool_way"}),
        ("cluster/submit_preference_training.sh", {"EXPERIMENT_ARM": "unknown"}),
        ("cluster/submit_apptainer_cache.sh", {"CACHE_BUILD_MODE": "unknown"}),
    )
    for index, (script, overrides) in enumerate(cases):
        case_path = tmp_path / str(index)
        case_path.mkdir()
        completed, calls = run_submission(case_path, script, **overrides)
        assert completed.returncode == 2, script
        assert calls == []


def test_cluster_sync_excludes_scratch_from_rsync(tmp_path: Path) -> None:
    rsync, log = recording_command(tmp_path, "rsync")
    env = base_env(
        tmp_path,
        COMMAND_LOG=str(log),
        DRY_RUN="1",
        REMOTE="cluster.example",
        REMOTE_DIR="/work/debug-depo",
        RSYNC_BIN=str(rsync),
    )
    completed = subprocess.run(
        ["bash", "cluster/sync_to_cx3.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = log.read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert arguments[-2:] == [f"{ROOT}/", "cluster.example:/work/debug-depo/"]
    assert "scratch/" in arguments


def run_cluster_pull(
    tmp_path: Path, **overrides: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    rsync, log = recording_command(tmp_path, "rsync")
    env = base_env(
        tmp_path,
        COMMAND_LOG=str(log),
        REMOTE="cluster.example",
        REMOTE_RDS="/rds/user",
        REMOTE_HOME="/home/user",
        REMOTE_RUNS_DIR="/rds/user/ephemeral/debug-depo/runs",
        LOCAL_DIR=str(tmp_path / "artifacts" / "runs"),
        PULL_CACHE_BUILDS="0",
        RSYNC_BIN=str(rsync),
        **overrides,
    )
    completed = subprocess.run(
        ["bash", "cluster/pull_cluster_artifacts.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    arguments = log.read_text(encoding="utf-8").splitlines()[0].split("\t")
    return completed, arguments


def test_cluster_pull_skips_experiment_model_payloads_by_default(tmp_path: Path) -> None:
    completed, arguments = run_cluster_pull(tmp_path)

    assert completed.returncode == 0, completed.stderr
    for pattern in (
        "**/experiments/**/*.safetensors",
        "**/experiments/**/*.bin",
        "**/experiments/**/*.pt",
        "**/experiments/**/*.pth",
        "**/experiments/**/*.ckpt",
        "**/experiments/**/*.gguf",
    ):
        assert pattern in arguments
    manifest = (tmp_path / "artifacts" / "runs" / "manifest.txt").read_text(encoding="utf-8")
    assert "pull_experiment_models=0" in manifest


def test_cluster_pull_can_include_experiment_model_payloads(tmp_path: Path) -> None:
    completed, arguments = run_cluster_pull(tmp_path, PULL_EXPERIMENT_MODELS="1")

    assert completed.returncode == 0, completed.stderr
    assert not any("experiments" in argument for argument in arguments)
    manifest = (tmp_path / "artifacts" / "runs" / "manifest.txt").read_text(encoding="utf-8")
    assert "pull_experiment_models=1" in manifest


def test_cluster_resource_profiles_keep_small_and_full_jobs_distinct() -> None:
    expected_resources = {
        "collect_swesmith_smoke.pbs": "select=1:ncpus=4:ngpus=1:mem=32gb",
        "collect_swesmith_pilot.pbs": "select=1:ncpus=8:ngpus=1:mem=48gb",
        "collect_swesmith_array.pbs": "select=1:ncpus=12:ngpus=1:mem=64gb",
        "train_preference.pbs": "select=1:ncpus=8:ngpus=1:mem=64gb",
    }
    for filename, expected in expected_resources.items():
        contents = (ROOT / "cluster" / "pbs" / filename).read_text(encoding="utf-8")
        assert expected in contents


def test_cluster_runtime_defaults_match_the_maintained_workflow(tmp_path: Path) -> None:
    completed, calls = run_submission(
        tmp_path,
        "cluster/submit_apptainer_cache_full.sh",
    )

    assert completed.returncode == 0, completed.stderr
    assert len(calls) == 1
    assert (
        "SWESMITH_TASK_IDS_FILE=data/splits/swesmith_cache_5700_instance_ids.txt"
        in joined(calls[0])
    )

    defaults = (ROOT / "cluster" / "env" / "defaults.sh").read_text(encoding="utf-8")
    assert "docker://vllm/vllm-openai:v0.11.0" in defaults
    assert 'STREAM_OUTPUT="${STREAM_OUTPUT:-1}"' in defaults

    evaluation = (ROOT / "cluster" / "run_swesmith_evaluation_job.sh").read_text(
        encoding="utf-8"
    )
    assert "DEFAULT_EXPECTED_SHARDS=10" in evaluation
    assert "DEFAULT_EXPECTED_TASKS=1000" in evaluation
    assert "DEFAULT_TOTAL_SAMPLES=4" in evaluation


def test_external_dependencies_are_pinned() -> None:
    installers = {
        "scripts/install_mini_swe_agent_plus.sh": "3dfa5e26831306978ff3cfa2da15b49113ded0e6",
        "scripts/install_swesmith.sh": "9b74ac08118a85c39c356802f7961893af73e07f",
    }
    for script, revision in installers.items():
        contents = (ROOT / script).read_text(encoding="utf-8")
        assert revision in contents
