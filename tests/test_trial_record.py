import csv
import json

import pytest

from debug_depo.trial_record import build_trial_record


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_trial(
    run_root,
    *,
    objective,
    experiment_arm,
    dmpo_name=None,
    depo_name=None,
    complete=True,
):
    if objective == "dmpo":
        trial_root = run_root / "experiments" / "dmpo" / dmpo_name
        learning_rate = 1e-6
        beta = 0.1
    elif experiment_arm == "depo":
        trial_root = run_root / "experiments" / "depo" / depo_name
        learning_rate = 2e-5
        beta = 0.2
    else:
        trial_root = run_root / "experiments" / "dmpo-depo" / dmpo_name / "depo" / depo_name
        learning_rate = 2e-5
        beta = 0.2

    training_dir = trial_root / "training"
    model_dir = trial_root / "model"
    data_path = run_root / "preference-data" / objective / "data.jsonl"
    config = {
        "objective": objective,
        "model_name_or_path": (
            "Kwai-Klear/Klear-AgentForge-8B-SFT"
            if objective == "dmpo"
            else str(run_root / "experiments" / "dmpo" / dmpo_name / "model")
        ),
        "data_path": str(data_path),
        "output_dir": str(training_dir),
        "max_rows": 0,
        "max_length": 32768,
        "epochs": 3,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": learning_rate,
        "weight_decay": 0.0,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "constant_with_warmup",
        "beta": beta,
        "gamma": 0.7,
        "desirable_weight": 1.0,
        "undesirable_weight": 1.0,
        "alpha_tokens": 2.0,
        "alpha_steps": 2.0,
        "token_metric": "completion_tokens",
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.0,
        "lora_target_modules": "q_proj,v_proj",
        "gradient_checkpointing": True,
        "mixed_precision": "bf16",
        "seed": 42,
        "dataloader_workers": 2,
        "logging_steps": 5,
        "save_steps": 100,
        "expected_num_processes": 8,
        "trust_remote_code": False,
        "attn_implementation": "sdpa",
    }
    trial_config = {
        "schema_version": 1,
        "experiment_arm": experiment_arm,
        "dmpo_trial_name": dmpo_name,
        "depo_trial_name": depo_name,
        "data_sha256": "a" * 64,
        "config": config,
    }
    write_json(training_dir / "trial_config.json", trial_config)
    if complete:
        write_json(
            training_dir / "training_manifest.json",
            {
                "schema_version": 1,
                "objective": objective,
                "data_sha256": "a" * 64,
                "global_steps": 72,
                "git_commit": "b" * 40,
                "config": config,
            },
        )
        write_json(
            model_dir / "package_manifest.json",
            {
                "schema_version": 1,
                "format": "standalone_huggingface_model",
                "base_model": config["model_name_or_path"],
                "adapter_path": str(training_dir / "adapter"),
                "output_dir": str(model_dir),
            },
        )
        write_json(model_dir / "config.json", {"model_type": "test"})
    return trial_root


def comparison_arm(name, path, *, resolved=0.4, tokens_per_resolved=1000.0):
    return {
        "name": name,
        "path": str(path),
        "is_baseline": name == "sft",
        "efficiency": {
            "resolution_rate": resolved,
            "total_tokens_per_resolved_task": tokens_per_resolved,
        },
        "resolution_rate_delta_vs_baseline": resolved - 0.4,
        "success_noninferior": resolved >= 0.39,
        "rankable": True,
        "selection_eligible": resolved >= 0.39,
        "resolution_transitions_vs_baseline": {
            "both_resolved": 38,
            "both_unresolved": 58,
            "gained": 2,
            "lost": 2,
        },
        "paired_deltas_vs_baseline": {
            "all": {
                "total_tokens": {"mean": -125.5},
                "action_steps": {"mean": -1.25},
            },
            "both_resolved": {},
        },
    }


def make_evaluation(run_root, *, model, budget):
    rollout_csv = run_root / "analysis" / "rollouts.csv"
    rollout_csv.parent.mkdir(parents=True, exist_ok=True)
    rollout_csv.write_text("instance_id\n", encoding="utf-8")
    write_json(
        run_root / "collection" / "shard-0" / "collection_manifest.json",
        {"schema_version": 3, "model": str(model)},
    )
    return rollout_csv


def make_comparison(path, *, budget, baseline_path, candidate_name, candidate_path):
    baseline = comparison_arm("sft", baseline_path, tokens_per_resolved=1500.0)
    candidate = comparison_arm(
        candidate_name,
        candidate_path,
        resolved=0.41,
        tokens_per_resolved=900.0,
    )
    payload = {
        "schema_version": 1,
        "baseline": "sft",
        "task_count": budget,
        "task_matrix_sha256": "c" * 64,
        "selected_arm": candidate_name,
        "eligible_ranking": [
            {
                "name": candidate_name,
                "total_tokens_per_resolved_task": 900.0,
                "resolution_rate": 0.41,
            },
            {
                "name": "sft",
                "total_tokens_per_resolved_task": 1500.0,
                "resolution_rate": 0.4,
            },
        ],
        "arms": [baseline, candidate],
    }
    write_json(path, payload)
    return path


def test_build_trial_record_joins_evaluation_by_packaged_model_path(tmp_path):
    runs_root = tmp_path / "runs"
    training_run = runs_root / "train"
    trial_root = make_trial(
        training_run,
        objective="dmpo",
        experiment_arm="dmpo",
        dmpo_name="g07",
    )
    baseline_csv = make_evaluation(
        runs_root / "validation-100-sft",
        model="Kwai-Klear/Klear-AgentForge-8B-SFT",
        budget=100,
    )
    candidate_csv = make_evaluation(
        runs_root / "validation-100-candidate",
        model=trial_root / "model",
        budget=100,
    )
    comparison = make_comparison(
        tmp_path / "comparison-100.json",
        budget=100,
        baseline_path=baseline_csv,
        candidate_name="arbitrary-arm-name",
        candidate_path=candidate_csv,
    )

    rows = build_trial_record(
        run_root=training_run,
        comparisons=[(100, comparison)],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["trial_id"] == "dmpo/g07"
    assert row["effective_global_batch_size"] == 128
    assert row["training_complete"] is True
    assert row["package_complete"] is True
    assert row["val_100_resolution_rate"] == 0.41
    assert row["val_100_total_tokens_per_resolved_task"] == 900.0
    assert row["val_100_mean_total_tokens_delta_vs_baseline"] == -125.5
    assert row["val_100_rank"] == 1
    assert row["val_100_selected"] is True
    assert row["latest_budget"] == 100
    assert row["latest_result"] == "selected"

    output = training_run / "experiments" / "trial-record.csv"
    with output.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 1
    assert written[0]["trial_id"] == "dmpo/g07"
    assert written[0]["training_complete"] == "true"
    assert written[0]["val_100_selected"] == "true"
    assert written[0]["val_200_resolution_rate"] == ""


def test_build_trial_record_uses_canonical_alias_and_records_depo_parent(tmp_path):
    training_run = tmp_path / "runs" / "train"
    make_trial(
        training_run,
        objective="dmpo",
        experiment_arm="dmpo",
        dmpo_name="g09",
    )
    make_trial(
        training_run,
        objective="depo",
        experiment_arm="dmpo-depo",
        dmpo_name="g09",
        depo_name="paper-a2-a2",
        complete=False,
    )
    baseline = tmp_path / "missing-sft.csv"
    candidate = tmp_path / "missing-depo.csv"
    comparison = make_comparison(
        tmp_path / "comparison-200.json",
        budget=200,
        baseline_path=baseline,
        candidate_name="dmpo-depo/g09/paper-a2-a2",
        candidate_path=candidate,
    )

    rows = build_trial_record(
        run_root=training_run,
        comparisons=[(200, comparison)],
    )

    by_id = {row["trial_id"]: row for row in rows}
    depo = by_id["dmpo-depo/g09/paper-a2-a2"]
    assert depo["parent_trial_id"] == "dmpo/g09"
    assert depo["gamma"] == ""
    assert depo["token_metric"] == "completion_tokens"
    assert depo["training_complete"] is False
    assert depo["package_complete"] is False
    assert depo["val_200_selected"] is True
    assert by_id["dmpo/g09"]["val_200_resolution_rate"] == ""


def test_build_trial_record_rejects_different_task_matrices_at_one_budget(tmp_path):
    training_run = tmp_path / "runs" / "train"
    make_trial(
        training_run,
        objective="dmpo",
        experiment_arm="dmpo",
        dmpo_name="g07",
    )
    first = make_comparison(
        tmp_path / "first.json",
        budget=100,
        baseline_path=tmp_path / "sft.csv",
        candidate_name="g07",
        candidate_path=tmp_path / "candidate.csv",
    )
    second = make_comparison(
        tmp_path / "second.json",
        budget=100,
        baseline_path=tmp_path / "sft.csv",
        candidate_name="g07",
        candidate_path=tmp_path / "candidate.csv",
    )
    payload = json.loads(second.read_text(encoding="utf-8"))
    payload["task_matrix_sha256"] = "d" * 64
    write_json(second, payload)

    with pytest.raises(ValueError, match="use different task matrices"):
        build_trial_record(
            run_root=training_run,
            comparisons=[(100, first), (100, second)],
        )


def test_build_trial_record_ignores_candidate_when_reused_as_later_baseline(tmp_path):
    training_run = tmp_path / "runs" / "train"
    make_trial(
        training_run,
        objective="dmpo",
        experiment_arm="dmpo",
        dmpo_name="g07",
    )
    make_trial(
        training_run,
        objective="depo",
        experiment_arm="dmpo-depo",
        dmpo_name="g07",
        depo_name="paper-a2-a2",
    )
    dmpo_comparison = make_comparison(
        tmp_path / "dmpo.json",
        budget=100,
        baseline_path=tmp_path / "sft.csv",
        candidate_name="g07",
        candidate_path=tmp_path / "dmpo.csv",
    )
    depo_comparison = make_comparison(
        tmp_path / "depo.json",
        budget=100,
        baseline_path=tmp_path / "dmpo.csv",
        candidate_name="paper-a2-a2",
        candidate_path=tmp_path / "depo.csv",
    )
    payload = json.loads(depo_comparison.read_text(encoding="utf-8"))
    payload["baseline"] = "g07"
    payload["arms"][0]["name"] = "g07"
    write_json(depo_comparison, payload)

    rows = build_trial_record(
        run_root=training_run,
        comparisons=[(100, dmpo_comparison), (100, depo_comparison)],
    )

    by_id = {row["trial_id"]: row for row in rows}
    assert by_id["dmpo/g07"]["val_100_baseline"] == "sft"
    assert by_id["dmpo-depo/g07/paper-a2-a2"]["val_100_baseline"] == "g07"
