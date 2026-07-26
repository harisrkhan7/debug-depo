import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_SETTING_NAMES = {
    "AGENTFORGE_MODEL",
    "AFTEROK_JOB_ID",
    "ALPHA_STEPS",
    "ALPHA_TOKENS",
    "BASE_MODEL",
    "BASE_SEED",
    "CACHE_BUILD_DATASETS",
    "CACHE_BUILD_MAX_WORKERS",
    "CACHE_BUILD_MODE",
    "CACHE_SUBMIT_PRINT_JOB_ID_ONLY",
    "CLUSTER_LOG_DIR",
    "CONTEXT_LENGTH",
    "DATASET",
    "DEPO_OUTPUT_DIR",
    "DEPO_MODEL_DIR",
    "DEPO_BETA",
    "DEPO_LEARNING_RATE",
    "DEPO_TOKEN_METRIC",
    "DEPO_TRAIN_OUTPUT_DIR",
    "DEPO_TRIAL_NAME",
    "DEPO_TRIAL_ROOT",
    "DEBUG_DEPO_SCRATCH",
    "DMPO_OUTPUT_DIR",
    "DMPO_MODEL_DIR",
    "DMPO_BETA",
    "DMPO_GAMMA",
    "DMPO_LEARNING_RATE",
    "DMPO_TRAIN_OUTPUT_DIR",
    "DMPO_TRIAL_NAME",
    "DMPO_TRIAL_ROOT",
    "DMPO_MODE",
    "DRY_RUN",
    "EVAL_MAX_WORKERS",
    "EVAL_TIMEOUT",
    "EXPERIMENT_ARM",
    "EXPECTED_TASKS",
    "EXPECTED_COUNT",
    "EXPECTED_SWEBENCH_TASKS",
    "HARNESS",
    "LIMIT",
    "MAX_STEPS",
    "MAX_LENGTH",
    "MAX_TRAIN_ROWS",
    "MAX_PAIRS_PER_TASK",
    "MIN_COST_RATIO",
    "MINI_SWE_ENVIRONMENT_CLASS",
    "MINI_SWE_CONFIG",
    "MINI_SWE_MODEL",
    "MINI_SWE_RUNNER",
    "NUM_SHARDS",
    "NUM_PROCESSES",
    "PACKAGE_MODEL",
    "PER_DEVICE_BATCH_SIZE",
    "PREFERENCE_OBJECTIVE",
    "PREFERENCE_MAX_ROLLOUTS",
    "PREFERENCE_DATA_MODE",
    "PREFERENCE_SAMPLE_INDICES",
    "REBUILD_PREFERENCE_DATA",
    "TRAIN_MEMORY",
    "TRAIN_NCPUS",
    "TRAIN_RUN_NAME",
    "TRAIN_RUN_ROOT",
    "MODEL_PATH",
    "EVAL_RUN_NAME",
    "EVAL_NUM_SHARDS",
    "EVAL_ROLLOUT_WORKERS",
    "EVAL_CONTEXT_LENGTH",
    "EVAL_EXPECTED_TASKS",
    "EVAL_TEMPERATURE",
    "EVAL_TASK_IDS_FILE",
    "COLLECTION_STAGE_LABEL",
    "EVALUATION_STAGE_LABEL",
    "ANALYSIS_STAGE_LABEL",
    "SUBMIT_MODEL_EVALUATIONS",
    "SUBMIT_DMPO_EVALUATION",
    "DMPO_EVAL_RUN_NAME",
    "DEPO_EVAL_RUN_NAME",
    "GRADIENT_ACCUMULATION_STEPS",
    "EPOCHS",
    "SAVE_STEPS",
    "OVERWRITE",
    "ROLLOUT_WORKERS",
    "RUN_ID",
    "RUN_NAME",
    "RUN_ROOT",
    "RUNS_PER_TEMPERATURE",
    "SMOKE_LIMIT",
    "SPLIT",
    "SUBMIT_ANALYSIS",
    "SUBMIT_EVAL",
    "SWESMITH_EVAL_RUNTIME",
    "SWEBENCH_CACHE_LIMIT",
    "SWEBENCH_DATASET",
    "SWEBENCH_DATASET_REVISION",
    "SWEBENCH_SPLIT",
    "SWEBENCH_TASK_IDS_FILE",
    "SWESMITH_DATASET_REVISION",
    "SWESMITH_CACHE_LIMIT",
    "SWESMITH_DATASET",
    "SWESMITH_SPLIT",
    "SWESMITH_TASK_IDS_FILE",
    "SWESMITH_MODE",
    "TASK_IDS_FILE",
    "TASK_LIMIT",
    "TEMPERATURE",
    "TEMPERATURES",
    "TOKEN_METRIC",
    "MODEL_NAME_OR_PATH",
    "TOTAL_SAMPLES",
    "TIMEOUT_SECONDS",
    "TOP_P",
    "INCLUDE_FAILURE_EFFICIENCY_PAIRS",
}


def dry_run(script: str, **overrides: str) -> str:
    env = os.environ.copy()
    for name in RUN_SETTING_NAMES:
        env.pop(name, None)
    env.update(
        {
            "DEBUG_DEPO_SCRATCH": str(ROOT / "scratch"),
            "DRY_RUN": "1",
            "SUBMIT_ANALYSIS": "1",
            "SUBMIT_EVAL": "1",
            **overrides,
        }
    )
    completed = subprocess.run(
        ["bash", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def job_section(output: str, title: str) -> str:
    section = output.split(f"{title}\n", maxsplit=1)[1]
    return section.split("\n\n", maxsplit=1)[0]


def test_verified_full_submission_preserves_evaluation_defaults():
    output = dry_run("cluster/submit_verified_full.sh")

    collection = job_section(output, "Collection job")
    evaluation = job_section(output, "Evaluation job")
    analysis = job_section(output, "Analysis job")
    assert "RUN_NAME=agentforge-verified-full" in collection
    assert "DATASET=princeton-nlp/SWE-bench_Verified" in collection
    assert (
        "SWEBENCH_DATASET_REVISION="
        "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
        in collection
    )
    assert "SPLIT=test" in collection
    assert "NUM_SHARDS=10" in collection
    assert "ROLLOUT_WORKERS=6" in collection
    assert "CONTEXT_LENGTH=65536" in collection
    assert "TIMEOUT_SECONDS=21600" in collection
    assert "TEMPERATURE=0.0" in collection
    assert "RUN_ID=agentforge_verified_full" in evaluation
    assert "DATASET=princeton-nlp/SWE-bench_Verified" in evaluation
    assert (
        "SWEBENCH_DATASET_REVISION="
        "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
        in evaluation
    )
    assert "SPLIT=test" in evaluation
    assert "EXPECTED_COUNT=500" in evaluation
    assert "EXPECTED_COUNT=500" in analysis
    assert (
        f"Cluster logs: {ROOT}/scratch/runs/"
        "agentforge-verified-full/cluster-logs"
        in output
    )


def test_verified_full_submission_ignores_inherited_limit():
    output = dry_run("cluster/submit_verified_full.sh", LIMIT="7")
    collection = job_section(output, "Collection job")

    assert "\n    LIMIT=" not in collection


def test_verified_full_submission_rejects_more_shards_than_expected_tasks():
    env = os.environ.copy()
    for name in RUN_SETTING_NAMES:
        env.pop(name, None)
    env.update(
        {
            "DRY_RUN": "1",
            "EXPECTED_COUNT": "5",
            "NUM_SHARDS": "10",
        }
    )

    completed = subprocess.run(
        ["bash", "cluster/submit_verified_full.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "NUM_SHARDS (10) cannot exceed EXPECTED_COUNT (5)." in completed.stderr
    assert "qsub" not in completed.stdout


def test_verified_full_submission_propagates_future_dataset_configuration():
    output = dry_run(
        "cluster/submit_verified_full.sh",
        RUN_NAME="future-train",
        DATASET="SWE-bench/SWE-smith-py",
        SPLIT="train",
        TASK_IDS_FILE="data/splits/train_ids.txt",
        EXPECTED_COUNT="123",
    )

    collection = job_section(output, "Collection job")
    evaluation = job_section(output, "Evaluation job")
    analysis = job_section(output, "Analysis job")
    for line in (collection, evaluation):
        assert "DATASET=SWE-bench/SWE-smith-py" in line
        assert "SPLIT=train" in line
        assert "TASK_IDS_FILE=data/splits/train_ids.txt" in line
        assert "SWEBENCH_DATASET_REVISION=" not in line
    assert "RUN_ID=future_train" in evaluation
    assert "EXPECTED_COUNT=123" in evaluation
    assert "EXPECTED_COUNT=123" in analysis


def test_verified_smoke_submission_propagates_validation_configuration():
    output = dry_run(
        "cluster/submit_verified_smoke.sh",
        RUN_NAME="future-validation",
        DATASET="org/dataset",
        SWEBENCH_DATASET_REVISION="validation-commit",
        SPLIT="validation",
        TASK_IDS_FILE="data/splits/validation_ids.txt",
        EXPECTED_COUNT="5",
    )

    collection = job_section(output, "Collection job")
    evaluation = job_section(output, "Evaluation job")
    analysis = job_section(output, "Analysis job")
    for line in (collection, evaluation):
        assert "DATASET=org/dataset" in line
        assert "SWEBENCH_DATASET_REVISION=validation-commit" in line
        assert "SPLIT=validation" in line
        assert "TASK_IDS_FILE=data/splits/validation_ids.txt" in line
    assert "RUN_ID=future_validation" in evaluation
    assert "EXPECTED_COUNT=5" in evaluation
    assert "EXPECTED_COUNT=5" in analysis


def test_swesmith_smoke_submission_has_four_runs_at_each_temperature():
    output = dry_run("cluster/submit_swesmith_smoke.sh")

    collection = job_section(output, "SWE-smith collection job")
    evaluation = job_section(output, "SWE-smith evaluation job")
    analysis = job_section(output, "SWE-smith analysis job")
    assert "DATASET=SWE-bench/SWE-smith-py" in collection
    assert "SPLIT=train" in collection
    assert (
        "SWESMITH_DATASET_REVISION="
        "77cab9055d42ab4a5c25c89a8f937096db13558e"
        in collection
    )
    assert "RUNS_PER_TEMPERATURE=4" in collection
    assert "TEMPERATURES=0.6:0.7" in collection
    assert "TOTAL_SAMPLES=8" in collection
    assert (
        "MINI_SWE_MODEL=hosted_vllm/Kwai-Klear/Klear-AgentForge-8B-SFT"
        in collection
    )
    assert "MINI_SWE_RUNNER=singularity" in collection
    assert "MINI_SWE_ENVIRONMENT_CLASS=singularity" in collection
    assert "ROLLOUT_WORKERS=2" in collection
    assert "EXPECTED_TASKS=2" in evaluation
    assert "EVAL_MAX_WORKERS=2" in evaluation
    assert "RUNS_PER_TEMPERATURE=4" in evaluation
    assert "TOTAL_SAMPLES=8" in evaluation
    assert "EXPECTED_TASKS=2" in analysis
    assert (
        f"Cluster logs: {ROOT}/scratch/runs/swesmith-smoke/cluster-logs"
        in output
    )


def test_swesmith_pilot_is_small_by_default():
    output = dry_run("cluster/submit_swesmith_pilot.sh")

    collection = job_section(output, "SWE-smith collection job")
    evaluation = job_section(output, "SWE-smith evaluation job")
    assert "PBS script: cluster/pbs/collect_swesmith_pilot.pbs" in collection
    assert "LIMIT=30" in collection
    assert "EXPECTED_TASKS=30" in collection
    assert "NUM_SHARDS=3" in collection
    assert "ROLLOUT_WORKERS=5" in collection
    assert "PBS script: cluster/pbs/evaluate_swesmith_pilot.pbs" in evaluation
    assert "EVAL_MAX_WORKERS=12" in evaluation
    pilot_pbs = (ROOT / "cluster/pbs/collect_swesmith_pilot.pbs").read_text(
        encoding="utf-8"
    )
    assert "#PBS -J 0-2" in pilot_pbs


def test_preference_data_submission_runs_two_one_time_builders_in_parallel():
    output = dry_run(
        "cluster/submit_preference_data.sh",
        RUN_NAME="swesmith-pilot-20260719",
    )

    assert "Run root:" in output
    assert "TOKEN_METRIC=total_tokens" in output
    assert "MIN_COST_RATIO=1.1" in output
    assert "cluster/pbs/build_dmpo_pairs.pbs" in output
    assert "cluster/pbs/build_depo_data.pbs" in output
    assert "run independently and may overlap" in output
    assert "REBUILD_PREFERENCE_DATA=0" in output
    assert "depend=afterok:<dmpo-job-id>" not in output


def test_swesmith_pilot_with_cache_is_one_dependency_chain():
    output = dry_run("cluster/submit_swesmith_pilot_with_cache.sh")

    cache = job_section(output, "Apptainer cache full job")
    collection = job_section(output, "SWE-smith collection job")
    evaluation = job_section(output, "SWE-smith evaluation job")
    analysis = job_section(output, "SWE-smith analysis job")
    assert "Pilot dependency chain: cache -> collect -> evaluate -> analyze" in output
    assert "CACHE_BUILD_MODE=full" in cache
    assert "CACHE_BUILD_DATASETS=swesmith" in cache
    assert "CACHE_BUILD_MAX_WORKERS=8" in cache
    assert "SWESMITH_CACHE_LIMIT=30" in cache
    assert (
        "SWESMITH_TASK_IDS_FILE="
        "data/splits/swesmith_train_5000_instance_ids.txt"
        in cache
    )
    assert "Dependency: successful job <cache-job-id>" in collection
    assert "LIMIT=30" in collection
    assert "EXPECTED_TASKS=30" in collection
    assert "NUM_SHARDS=3" in collection
    assert (
        "TASK_IDS_FILE=data/splits/swesmith_train_5000_instance_ids.txt"
        in collection
    )
    assert "Dependency: successful collection" in evaluation
    assert "TASK_IDS_FILE=" not in evaluation
    assert "Dependency: successful evaluation" in analysis
    assert "TASK_IDS_FILE=" not in analysis


def test_swesmith_pilot_with_cache_submits_pbs_job_ids_in_order(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    qsub_log = tmp_path / "qsub.log"
    fake_qsub = fake_bin / "qsub"
    fake_qsub.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >>"$QSUB_LOG"
case "$*" in
  *apptainer-cache-full*) printf '101.server\\n' ;;
  *swesmith-pilot-collect*) printf '102[].server\\n' ;;
  *swesmith-pilot-eval*) printf '103.server\\n' ;;
  *swesmith-pilot-analysis*) printf '104.server\\n' ;;
  *) printf 'unexpected qsub invocation: %s\\n' "$*" >&2; exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    fake_qsub.chmod(0o755)
    env = os.environ.copy()
    for name in RUN_SETTING_NAMES:
        env.pop(name, None)
    env.update(
        {
            "DEBUG_DEPO_SCRATCH": str(tmp_path / "scratch"),
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "QSUB_LOG": str(qsub_log),
        }
    )

    completed = subprocess.run(
        ["bash", "cluster/submit_swesmith_pilot_with_cache.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    submissions = qsub_log.read_text(encoding="utf-8").splitlines()
    log_dir = tmp_path / "scratch/runs/swesmith-pilot/cluster-logs"
    assert len(submissions) == 4
    assert log_dir.is_dir()
    for submission in submissions:
        assert f"-o {log_dir}/ -e {log_dir}/" in submission
    assert "-N apptainer-cache-full" in submissions[0]
    assert "CACHE_BUILD_DATASETS=swesmith" in submissions[0]
    assert "SWESMITH_CACHE_LIMIT=30" in submissions[0]
    assert "-N swesmith-pilot-collect" in submissions[1]
    assert f"RUN_ROOT={log_dir.parent}" in submissions[1]
    assert "-J 0-2" in submissions[1]
    assert "-W depend=afterok:101.server" in submissions[1]
    assert "-N swesmith-pilot-eval" in submissions[2]
    assert "-W depend=afterok:102[].server" in submissions[2]
    assert "-N swesmith-pilot-analysis" in submissions[3]
    assert "-W depend=afterok:103.server" in submissions[3]
    assert "Submitted pilot-scoped SWE-smith cache build: 101.server" in (
        completed.stdout
    )


def test_swesmith_notebook_defaults_to_pilot_with_installed_dependencies():
    notebook = json.loads(
        (ROOT / "notebooks/inspect_swesmith_collection.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    setup = (ROOT / "cluster/setup_jupyter_env.sh").read_text(encoding="utf-8")

    assert 'run_name = os.environ.get("RUN_NAME", "swesmith-pilot")' in source
    assert "30 tasks across three collection shards" in source
    assert "cluster/submit_swesmith_pilot.sh" in source
    assert 'python -m pip install -e "$ROOT_DIR[notebooks]"' in setup


def test_swesmith_reduced_pbs_resources_do_not_change_full_resources():
    expected_headers = {
        "collect_swesmith_smoke.pbs": "select=1:ncpus=4:ngpus=1:mem=32gb",
        "evaluate_swesmith_smoke.pbs": "select=1:ncpus=4:mem=32gb",
        "collect_swesmith_pilot.pbs": "select=1:ncpus=8:ngpus=1:mem=48gb",
        "evaluate_swesmith_pilot.pbs": "select=1:ncpus=16:mem=128gb",
        "collect_swesmith_array.pbs": "select=1:ncpus=12:ngpus=1:mem=64gb",
        "evaluate_swesmith.pbs": "select=1:ncpus=32:mem=256gb",
        "analyze_swesmith.pbs": "select=1:ncpus=2:mem=8gb",
        "analyze_swesmith_full.pbs": "select=1:ncpus=4:mem=32gb",
    }

    for filename, resource in expected_headers.items():
        contents = (ROOT / "cluster/pbs" / filename).read_text(encoding="utf-8")
        assert resource in contents


def test_pbs_logs_are_assigned_by_submission_wrappers():
    for pbs_path in (ROOT / "cluster/pbs").glob("*.pbs"):
        contents = pbs_path.read_text(encoding="utf-8")
        assert "#PBS -o cluster/logs/" not in contents
        assert "#PBS -e cluster/logs/" not in contents

    wrappers = [
        "cluster/submit_apptainer_cache.sh",
        "cluster/submit_swesmith.sh",
        "cluster/submit_verified_analysis.sh",
        "cluster/submit_verified_full.sh",
        "cluster/submit_verified_smoke.sh",
    ]
    for wrapper in wrappers:
        contents = (ROOT / wrapper).read_text(encoding="utf-8")
        assert 'CLUSTER_LOG_DIR/"' in contents
        assert '-o "$CLUSTER_LOG_DIR/"' in contents
        assert '-e "$CLUSTER_LOG_DIR/"' in contents


def test_swesmith_full_defaults_to_the_tracked_training_sample():
    output = dry_run("cluster/submit_swesmith_full.sh")

    collection = job_section(output, "SWE-smith collection job")
    evaluation = job_section(output, "SWE-smith evaluation job")
    analysis = job_section(output, "SWE-smith analysis job")
    assert "SWESMITH_MODE=full" in collection
    assert "RUN_NAME=swesmith-full" in collection
    assert "EXPECTED_TASKS=5000" in collection
    assert (
        "TASK_IDS_FILE=data/splits/swesmith_train_5000_instance_ids.txt"
        in collection
    )
    assert "NUM_SHARDS=50" in collection
    assert "ROLLOUT_WORKERS=6" in collection
    assert "\n    LIMIT=" not in collection
    assert "EXPECTED_TASKS=5000" in evaluation
    assert "EVAL_MAX_WORKERS=25" in evaluation
    assert "EXPECTED_TASKS=5000" in analysis
    assert "PBS script: cluster/pbs/analyze_swesmith_full.pbs" in analysis
    full_analysis_pbs = (
        ROOT / "cluster/pbs/analyze_swesmith_full.pbs"
    ).read_text(encoding="utf-8")
    assert "walltime=03:00:00" in full_analysis_pbs
    full_collection_pbs = (
        ROOT / "cluster/pbs/collect_swesmith_array.pbs"
    ).read_text(encoding="utf-8")
    assert "#PBS -J 0-49" in full_collection_pbs
    assert "walltime=24:00:00" in full_collection_pbs


def test_swesmith_full_allows_the_complete_dataset_to_be_selected_explicitly():
    output = dry_run(
        "cluster/submit_swesmith_full.sh",
        TASK_IDS_FILE="",
        EXPECTED_TASKS="50908",
    )

    collection = job_section(output, "SWE-smith collection job")
    evaluation = job_section(output, "SWE-smith evaluation job")
    assert "EXPECTED_TASKS=50908" in collection
    assert "\n    TASK_IDS_FILE=" not in collection
    assert "EXPECTED_TASKS=50908" in evaluation


def test_swesmith_full_subset_defaults_expected_tasks_to_limit():
    output = dry_run(
        "cluster/submit_swesmith_full.sh",
        TASK_LIMIT="250",
        NUM_SHARDS="10",
    )

    collection = job_section(output, "SWE-smith collection job")
    evaluation = job_section(output, "SWE-smith evaluation job")
    assert "LIMIT=250" in collection
    assert "EXPECTED_TASKS=250" in collection
    assert "EXPECTED_TASKS=250" in evaluation


def test_swesmith_submission_propagates_supported_miniswe_overrides():
    output = dry_run(
        "cluster/submit_swesmith_pilot.sh",
        MINI_SWE_MODEL="hosted_vllm/org/model",
        MINI_SWE_CONFIG="/project/custom.yaml",
        MINI_SWE_RUNNER="swebench",
        MINI_SWE_ENVIRONMENT_CLASS="docker",
    )

    collection = job_section(output, "SWE-smith collection job")
    assert "MINI_SWE_MODEL=hosted_vllm/org/model" in collection
    assert "MINI_SWE_CONFIG=/project/custom.yaml" in collection
    assert "MINI_SWE_RUNNER=swebench" in collection
    assert "MINI_SWE_ENVIRONMENT_CLASS=docker" in collection


def test_swesmith_submission_rejects_pool_way_before_qsub():
    env = os.environ.copy()
    for name in RUN_SETTING_NAMES:
        env.pop(name, None)
    env.update(
        {
            "DRY_RUN": "1",
            "MINI_SWE_RUNNER": "pool_way",
            "MINI_SWE_ENVIRONMENT_CLASS": "docker",
        }
    )

    completed = subprocess.run(
        ["bash", "cluster/submit_swesmith_pilot.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "pool_way is unsupported" in completed.stderr
    assert "SWE-smith collection job" not in completed.stdout


def test_swesmith_scripts_use_normal_uv_run_and_strict_completion():
    collection = (ROOT / "scripts/collect_swesmith.sh").read_text(encoding="utf-8")
    evaluation = (ROOT / "scripts/evaluate_swesmith.sh").read_text(encoding="utf-8")
    analysis = (ROOT / "scripts/analyze_swesmith.sh").read_text(encoding="utf-8")

    for script in (collection, evaluation, analysis):
        assert "--no-sync" not in script
        assert '"$UV_BIN" run python' in script
    assert "--require-complete" in collection
    assert "--require-complete" in evaluation

    verified_collection = (ROOT / "scripts/collect_rollouts.sh").read_text(
        encoding="utf-8"
    )
    verified_evaluation = (ROOT / "scripts/evaluate_apptainer.sh").read_text(
        encoding="utf-8"
    )
    assert "--require-complete" in verified_collection
    assert "--require-complete" in verified_evaluation


def test_apptainer_cache_pbs_has_mode_specific_resources():
    smoke_pbs = (
        ROOT / "cluster/pbs/build_apptainer_cache_smoke.pbs"
    ).read_text(
        encoding="utf-8"
    )
    full_pbs = (
        ROOT / "cluster/pbs/build_apptainer_cache_full.pbs"
    ).read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "cluster/run_apptainer_cache_job.sh").read_text(
        encoding="utf-8"
    )
    script = (ROOT / "scripts/build_apptainer_cache.sh").read_text(
        encoding="utf-8"
    )

    assert "select=1:ncpus=8:mem=64gb" in smoke_pbs
    assert "walltime=24:00:00" in smoke_pbs
    assert "CACHE_BUILD_MODE=smoke" in smoke_pbs
    assert 'CACHE_BUILD_MAX_WORKERS="${CACHE_BUILD_MAX_WORKERS:-2}"' in smoke_pbs
    assert "select=1:ncpus=32:mem=128gb" in full_pbs
    assert "walltime=48:00:00" in full_pbs
    assert "CACHE_BUILD_MODE=full" in full_pbs
    assert 'CACHE_BUILD_MAX_WORKERS="${CACHE_BUILD_MAX_WORKERS:-20}"' in full_pbs
    for pbs in (smoke_pbs, full_pbs):
        assert "cluster/run_apptainer_cache_job.sh" in pbs
    assert 'MODE="${CACHE_BUILD_MODE:-smoke}"' in runner
    assert "smoke)" in runner
    assert "full)" in runner
    assert "DEFAULT_WORKERS=20" in runner
    assert "data/splits/train_instance_ids.txt" in runner
    assert "--expected-swebench-tasks" in script
    assert "SWESMITH_TASK_IDS_FILE" in script
    assert "--mini-image-template" in (
        ROOT / "scripts/collect_rollouts.sh"
    ).read_text(encoding="utf-8")


def test_apptainer_cache_submission_wrappers_select_mode_defaults():
    smoke = dry_run("cluster/submit_apptainer_cache_smoke.sh")
    full = dry_run("cluster/submit_apptainer_cache_full.sh")

    smoke_job = job_section(smoke, "Apptainer cache smoke job")
    full_job = job_section(full, "Apptainer cache full job")
    assert "PBS script: cluster/pbs/build_apptainer_cache_smoke.pbs" in smoke_job
    assert "CACHE_BUILD_MODE=smoke" in smoke_job
    assert "CACHE_BUILD_MAX_WORKERS=2" in smoke_job
    assert "PBS script: cluster/pbs/build_apptainer_cache_full.pbs" in full_job
    assert "CACHE_BUILD_MODE=full" in full_job
    assert "CACHE_BUILD_MAX_WORKERS=20" in full_job
    for job in (smoke_job, full_job):
        assert "CACHE_BUILD_DATASETS=both" in job
        assert "SWESMITH_TASK_IDS_FILE=data/splits/train_instance_ids.txt" in job
        assert "EXPECTED_SWEBENCH_TASKS=500" in job


def test_apptainer_cache_submission_propagates_supported_overrides():
    output = dry_run(
        "cluster/submit_apptainer_cache_full.sh",
        CACHE_BUILD_DATASETS="swesmith",
        CACHE_BUILD_MAX_WORKERS="12",
        SWESMITH_TASK_IDS_FILE="data/splits/validation_instance_ids.txt",
    )
    job = job_section(output, "Apptainer cache full job")

    assert "CACHE_BUILD_DATASETS=swesmith" in job
    assert "CACHE_BUILD_MAX_WORKERS=12" in job
    assert (
        "SWESMITH_TASK_IDS_FILE=data/splits/validation_instance_ids.txt"
        in job
    )


def test_cluster_artifact_pull_includes_cache_build_summaries():
    script = (ROOT / "cluster/pull_cluster_artifacts.sh").read_text(
        encoding="utf-8"
    )

    assert 'REMOTE_CACHE_BUILDS_DIR="${REMOTE_CACHE_BUILDS_DIR:-' in script
    assert 'LOCAL_CACHE_BUILDS_DIR="${LOCAL_CACHE_BUILDS_DIR:-' in script
    assert (
        'copy_remote_path "$REMOTE_CACHE_BUILDS_DIR/" '
        '"$LOCAL_CACHE_BUILDS_DIR/"'
        in script
    )
    assert 'cache_builds_status="pulled"' in script
    assert "remote_cache_builds_dir=%s" in script
    assert "local_cache_builds_dir=%s" in script
    assert "cache_builds_status=%s" in script


def test_swesmith_installers_pin_external_repositories():
    mini = (ROOT / "scripts/install_mini_swe_agent_plus.sh").read_text(
        encoding="utf-8"
    )
    smith = (ROOT / "scripts/install_swesmith.sh").read_text(encoding="utf-8")
    collection = (ROOT / "scripts/collect_swesmith.sh").read_text(
        encoding="utf-8"
    )
    verified_collection = (ROOT / "scripts/collect_rollouts.sh").read_text(
        encoding="utf-8"
    )

    assert "3dfa5e26831306978ff3cfa2da15b49113ded0e6" in mini
    assert "9b74ac08118a85c39c356802f7961893af73e07f" in smith
    assert "pull --ff-only" not in mini
    assert "pull --ff-only" not in smith
    assert "pull_sif_if_missing" in mini
    assert "MSWEA_SINGULARITY_SIF_DIR" in mini
    for collector in (verified_collection, collection):
        assert "MSWEA_SINGULARITY_SIF_DIR" in collector
    assert "SWEBENCH_APPTAINER_SIF_DIR" in verified_collection
    assert "SWESMITH_APPTAINER_SIF_DIR" in collection


def test_cluster_sync_excludes_the_complete_scratch_directory():
    sync_script = (ROOT / "cluster/sync_to_cx3.sh").read_text(encoding="utf-8")

    assert '--exclude "scratch/"' in sync_script
    assert '--exclude "scratch/*"' not in sync_script


def test_preference_training_dry_run_chains_data_training_packages_and_evaluation():
    output = dry_run(
        "cluster/submit_preference_training.sh",
        RUN_NAME="preference-pilot",
        PREFERENCE_DATA_MODE="build",
    )

    assert "1. DMPO data:" in output
    assert "2. DEPO data:" in output
    assert "3. DMPO train:" in output
    assert "4. DMPO package:" in output
    assert "5. DEPO train:" in output
    assert "6. DEPO package:" in output
    assert "Experiment arm: dmpo-depo" in output
    assert "2. DEPO data: qsub -v" in output
    assert "depend=afterok:<data-jobs>" in output
    assert "depend=afterok:<dmpo-train>" in output
    assert "depend=afterok:<parent-and-data>" in output
    assert "depend=afterok:<depo-train>" in output
    assert "MODEL_NAME_OR_PATH=Kwai-Klear/Klear-AgentForge-8B-SFT" in output
    assert "LEARNING_RATE=1e-6" in output
    assert "LEARNING_RATE=2e-5" in output
    assert "ALPHA_TOKENS=2.0" in output
    assert "PACKAGE_MODEL=0" in output
    assert "DMPO trial: default" in output
    assert "DEPO trial: default" in output
    assert "preference-pilot-dmpo-default-evaluation-500" in output
    assert "preference-pilot-dmpo-default-depo-default-evaluation-500" in output
    assert "existing SWE-bench Verified 500-task evaluation split" in output
    assert "experiments/dmpo/default/model" in output
    assert "experiments/dmpo-depo/default/depo/default/model" in output


def test_preference_trial_names_isolate_outputs_and_can_reuse_data():
    output = dry_run(
        "cluster/submit_preference_training.sh",
        RUN_NAME="preference-pilot",
        DMPO_TRIAL_NAME="gamma09",
        DEPO_TRIAL_NAME="alpha1",
        PREFERENCE_DATA_MODE="reuse",
        DMPO_MODE="reuse",
    )

    assert "Preference data mode: reuse" in output
    assert "reuse existing DMPO and DEPO files" in output
    assert "experiments/dmpo/gamma09/training" in output
    assert "experiments/dmpo/gamma09/model" in output
    assert "experiments/dmpo-depo/gamma09/depo/alpha1/training" in output
    assert "experiments/dmpo-depo/gamma09/depo/alpha1/model" in output
    assert "preference-pilot-dmpo-gamma09-evaluation-500" in output
    assert "preference-pilot-dmpo-gamma09-depo-alpha1-evaluation-500" in output
    assert "DMPO train/package: reuse" in output


def test_dmpo_arm_only_builds_trains_and_evaluates_dmpo():
    output = dry_run(
        "cluster/submit_preference_training.sh",
        RUN_NAME="preference-pilot",
        EXPERIMENT_ARM="dmpo",
        DMPO_TRIAL_NAME="gamma07",
        PREFERENCE_DATA_MODE="build",
    )

    assert "Model lineage: Kwai-Klear/Klear-AgentForge-8B-SFT -> DMPO" in output
    assert "DMPO data:" in output
    assert "DMPO train:" in output
    assert "DMPO package:" in output
    assert "preference-pilot-dmpo-gamma07-evaluation-500" in output
    assert "DEPO data:" not in output
    assert "DEPO train:" not in output
    assert "DEPO package:" not in output
    assert "DEPO: not part of this arm" in output


def test_preference_training_defaults_to_reusing_one_time_data():
    output = dry_run(
        "cluster/submit_preference_training.sh",
        RUN_NAME="preference-pilot",
        EXPERIMENT_ARM="dmpo",
        DMPO_TRIAL_NAME="pilot-dmpo",
        MAX_TRAIN_ROWS="64",
        MAX_LENGTH="8192",
        EPOCHS="1",
    )

    assert "Preference data mode: reuse" in output
    assert "reuse existing DMPO files" in output
    assert "DMPO data:" not in output
    assert "MAX_TRAIN_ROWS=64" in output
    assert "MAX_LENGTH=8192" in output
    assert "Training resources: 1 node, 8 CPUs, 1 GPUs, 64gb memory" in output


def test_depo_arm_uses_baseline_parent_and_separate_output_root():
    output = dry_run(
        "cluster/submit_preference_training.sh",
        RUN_NAME="preference-pilot",
        EXPERIMENT_ARM="depo",
        DEPO_TRIAL_NAME="alpha2",
    )

    assert "Model lineage: Kwai-Klear/Klear-AgentForge-8B-SFT -> DEPO" in output
    assert "MODEL_NAME_OR_PATH=Kwai-Klear/Klear-AgentForge-8B-SFT" in output
    assert "BASE_MODEL=Kwai-Klear/Klear-AgentForge-8B-SFT" in output
    assert "experiments/depo/alpha2/training" in output
    assert "experiments/depo/alpha2/model" in output
    assert "preference-pilot-depo-alpha2-evaluation-500" in output
    assert "DMPO data:" not in output
    assert "DMPO train:" not in output
    assert "DMPO package:" not in output


def test_sequential_arm_can_reuse_dmpo_for_another_depo_trial():
    output = dry_run(
        "cluster/submit_preference_training.sh",
        RUN_NAME="preference-pilot",
        EXPERIMENT_ARM="dmpo-depo",
        DMPO_TRIAL_NAME="gamma07",
        DEPO_TRIAL_NAME="alpha4",
        PREFERENCE_DATA_MODE="reuse",
        DMPO_MODE="reuse",
    )

    assert "DMPO train/package: reuse" in output
    assert "MODEL_NAME_OR_PATH=" in output
    assert "experiments/dmpo/gamma07/model" in output
    assert "experiments/dmpo-depo/gamma07/depo/alpha4/model" in output
    assert (
        "DMPO: preference-pilot-dmpo-gamma07-evaluation-500 (enabled: 0)"
        in output
    )


def test_preference_submission_rejects_unknown_experiment_arm():
    env = os.environ.copy()
    for name in RUN_SETTING_NAMES:
        env.pop(name, None)
    env.update({"DRY_RUN": "1", "EXPERIMENT_ARM": "unknown"})

    completed = subprocess.run(
        ["bash", "cluster/submit_preference_training.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "EXPERIMENT_ARM must be dmpo, depo, or dmpo-depo." in completed.stderr


def test_preference_training_scripts_package_standalone_models():
    shared = (ROOT / "scripts/train_preference.sh").read_text(encoding="utf-8")
    dmpo = (ROOT / "scripts/train_dmpo.sh").read_text(encoding="utf-8")
    depo = (ROOT / "scripts/train_depo.sh").read_text(encoding="utf-8")
    train_pbs = (ROOT / "cluster/pbs/train_preference.pbs").read_text(encoding="utf-8")

    assert "accelerate_args=(launch" in shared
    assert "accelerate_args+=(--multi_gpu)" in shared
    assert 'PACKAGE_MODEL:-1' in shared
    assert "debug-depo-package-model" in shared
    assert "package_manifest.json" in shared
    assert "PREFERENCE_OBJECTIVE=dmpo" in dmpo
    assert "PREFERENCE_OBJECTIVE=depo" in depo
    assert "scripts/train_preference.sh" in dmpo
    assert "scripts/train_preference.sh" in depo
    assert "select=1:ncpus=8:ngpus=1:mem=64gb" in train_pbs
    assert "walltime=48:00:00" in train_pbs
    assert 'PACKAGE_MODEL="${PACKAGE_MODEL:-0}"' in train_pbs

    package_script = (ROOT / "scripts/package_preference_model.sh").read_text(
        encoding="utf-8"
    )
    assert "debug-depo-package-model" in package_script
    package_pbs = (ROOT / "cluster/pbs/package_preference.pbs").read_text(
        encoding="utf-8"
    )
    assert "select=1:ncpus=4:mem=64gb" in package_pbs
    assert "walltime=04:00:00" in package_pbs


def test_training_setup_preserves_rollout_dependencies():
    setup = (ROOT / "cluster/setup_training_env.sh").read_text(encoding="utf-8")

    assert '"$UV_BIN" sync --extra dev --extra training' not in setup
    assert '"$UV_BIN" sync --inexact' in setup
    assert "--extra swebench" in setup
    assert "minisweagent" in setup


def test_preference_notebook_is_guarded_and_runs_dmpo_before_sequential_depo():
    notebook = json.loads(
        (ROOT / "notebooks/cluster_preference_training.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "swesmith-pilot-20260719" in source
    assert "MAX_TRAIN_ROWS = os.getenv('MAX_TRAIN_ROWS', '64')" in source
    assert "MAX_LENGTH = os.getenv('TRAIN_MAX_LENGTH', '8192')" in source
    assert "SUBMIT_DATA_JOBS = False" in source
    assert "SUBMIT_DMPO = False" in source
    assert "SUBMIT_DEPO = False" in source
    assert "cluster/submit_preference_data.sh" in source
    assert "'EXPERIMENT_ARM': 'dmpo'" in source
    assert "'EXPERIMENT_ARM': 'dmpo-depo'" in source
    assert "'DMPO_MODE': 'reuse'" in source
    assert "swebench_verified_pilot_5_instance_ids.txt" in source


def test_preference_evaluation_reuses_working_500_task_verified_pipeline():
    output = dry_run(
        "cluster/submit_preference_evaluation.sh",
        PREFERENCE_OBJECTIVE="dmpo",
        TRAIN_RUN_NAME="preference-pilot",
        TRAIN_RUN_ROOT=str(ROOT / "scratch/runs/preference-pilot"),
        AFTEROK_JOB_ID="123.cluster",
    )

    rollouts = job_section(output, "DMPO rollout generation job")
    evaluation = job_section(output, "DMPO evaluation job")
    analysis = job_section(output, "DMPO analysis job")
    assert "PBS script: cluster/pbs/collect_verified_full.pbs" in rollouts
    assert "dependency: successful job 123.cluster" in rollouts
    assert "RUN_NAME=preference-pilot-dmpo-default-evaluation-500" in rollouts
    assert "DATASET=princeton-nlp/SWE-bench_Verified" in rollouts
    assert "NUM_SHARDS=10" in rollouts
    assert "TEMPERATURE=0.0" in rollouts
    assert "ROLLOUT_WORKERS=6" in rollouts
    assert "CONTEXT_LENGTH=32768" in rollouts
    assert "experiments/dmpo/default/model" in rollouts
    assert "PBS script: cluster/pbs/evaluate_verified_full.pbs" in evaluation
    assert "EXPECTED_COUNT=500" in evaluation
    assert "PBS script: cluster/pbs/analyze_verified.pbs" in analysis


def test_direct_depo_evaluation_has_distinct_model_and_run_paths():
    output = dry_run(
        "cluster/submit_preference_evaluation.sh",
        PREFERENCE_OBJECTIVE="depo",
        EXPERIMENT_ARM="depo",
        DEPO_TRIAL_NAME="alpha2",
        TRAIN_RUN_NAME="preference-pilot",
        TRAIN_RUN_ROOT=str(ROOT / "scratch/runs/preference-pilot"),
        AFTEROK_JOB_ID="456.cluster",
    )

    rollouts = job_section(output, "DEPO rollout generation job")
    assert "RUN_NAME=preference-pilot-depo-alpha2-evaluation-500" in rollouts
    assert "experiments/depo/alpha2/model" in rollouts
    assert "experiments/dmpo-depo" not in rollouts
