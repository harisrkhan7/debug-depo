import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_SETTING_NAMES = {
    "AGENTFORGE_MODEL",
    "CONTEXT_LENGTH",
    "DATASET",
    "DRY_RUN",
    "EVAL_MAX_WORKERS",
    "EVAL_TIMEOUT",
    "EXPECTED_COUNT",
    "HARNESS",
    "LIMIT",
    "MAX_STEPS",
    "MINI_SWE_ENVIRONMENT_CLASS",
    "MINI_SWE_RUNNER",
    "NUM_SHARDS",
    "OVERWRITE",
    "ROLLOUT_WORKERS",
    "RUN_ID",
    "RUN_NAME",
    "SMOKE_LIMIT",
    "SPLIT",
    "SUBMIT_ANALYSIS",
    "SUBMIT_EVAL",
    "TASK_IDS_FILE",
    "TEMPERATURE",
    "TIMEOUT_SECONDS",
    "TOP_P",
}


def dry_run(script: str, **overrides: str) -> str:
    env = os.environ.copy()
    for name in RUN_SETTING_NAMES:
        env.pop(name, None)
    env.update({"DRY_RUN": "1", "SUBMIT_ANALYSIS": "1", "SUBMIT_EVAL": "1", **overrides})
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


def test_full_submission_preserves_verified_evaluation_defaults():
    output = dry_run("cluster/submit_full.sh")

    collection = job_section(output, "Collection job")
    evaluation = job_section(output, "Evaluation job")
    analysis = job_section(output, "Analysis job")
    assert "RUN_NAME=agentforge-verified-full" in collection
    assert "DATASET=princeton-nlp/SWE-bench_Verified" in collection
    assert "SPLIT=test" in collection
    assert "NUM_SHARDS=10" in collection
    assert "TEMPERATURE=0.0" in collection
    assert "RUN_ID=agentforge_verified_full" in evaluation
    assert "DATASET=princeton-nlp/SWE-bench_Verified" in evaluation
    assert "SPLIT=test" in evaluation
    assert "EXPECTED_COUNT=500" in evaluation
    assert "EXPECTED_COUNT=500" in analysis


def test_full_submission_ignores_inherited_limit():
    output = dry_run("cluster/submit_full.sh", LIMIT="7")
    collection = job_section(output, "Collection job")

    assert "\n    LIMIT=" not in collection


def test_full_submission_rejects_more_shards_than_expected_tasks():
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
        ["bash", "cluster/submit_full.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "NUM_SHARDS (10) cannot exceed EXPECTED_COUNT (5)." in completed.stderr
    assert "qsub" not in completed.stdout


def test_full_submission_propagates_future_dataset_configuration():
    output = dry_run(
        "cluster/submit_full.sh",
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
    assert "RUN_ID=future_train" in evaluation
    assert "EXPECTED_COUNT=123" in evaluation
    assert "EXPECTED_COUNT=123" in analysis


def test_smoke_submission_propagates_validation_configuration():
    output = dry_run(
        "cluster/submit_smoke.sh",
        RUN_NAME="future-validation",
        DATASET="org/dataset",
        SPLIT="validation",
        TASK_IDS_FILE="data/splits/validation_ids.txt",
        EXPECTED_COUNT="5",
    )

    collection = job_section(output, "Collection job")
    evaluation = job_section(output, "Evaluation job")
    analysis = job_section(output, "Analysis job")
    for line in (collection, evaluation):
        assert "DATASET=org/dataset" in line
        assert "SPLIT=validation" in line
        assert "TASK_IDS_FILE=data/splits/validation_ids.txt" in line
    assert "RUN_ID=future_validation" in evaluation
    assert "EXPECTED_COUNT=5" in evaluation
    assert "EXPECTED_COUNT=5" in analysis
