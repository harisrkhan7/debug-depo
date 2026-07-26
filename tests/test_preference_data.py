import json
from pathlib import Path

import pytest

from debug_depo.build_depo_data import build_depo_data
from debug_depo.build_dmpo_pairs import build_dmpo_pairs
from debug_depo.preference_data import load_evaluated_trajectories
from debug_depo.preference_data import (
    parse_sample_indices,
    select_sample_indices,
    validate_preference_artifacts,
)
from debug_depo.utils import read_jsonl


def _assistant(content: str, prompt: int, completion: int) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "extra": {
            "response": {
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": prompt + completion,
                }
            }
        },
    }


def _write_rollout(
    root: Path,
    *,
    instance_id: str,
    sample_index: int,
    resolved: bool,
    prompt_tokens: tuple[int, int],
    completion_tokens: tuple[int, int],
) -> None:
    trajectory_dir = (
        root
        / "collection"
        / "shard-0"
        / "samples"
        / f"sample-{sample_index}"
        / "trajectories"
        / instance_id
    )
    raw_dir = trajectory_dir / instance_id
    raw_dir.mkdir(parents=True)
    (trajectory_dir / "trajectory.json").write_text(
        json.dumps({"instance_id": instance_id, "status": "completed"}),
        encoding="utf-8",
    )
    messages = [
        {"role": "system", "content": "debug agent"},
        {"role": "user", "content": f"fix {instance_id}"},
        _assistant("inspect", prompt_tokens[0], completion_tokens[0]),
        {"role": "user", "content": "observation"},
        _assistant("patch", prompt_tokens[1], completion_tokens[1]),
        # Trajectories may contain a final environment acknowledgement.
        {"role": "user", "content": "submitted"},
    ]
    (raw_dir / f"{instance_id}.traj.json").write_text(
        json.dumps({"messages": messages}),
        encoding="utf-8",
    )
    evaluation_dir = root / "evaluation" / f"sample-{sample_index}"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    status = "resolved" if resolved else "unresolved"
    (evaluation_dir / "summary.json").write_text(
        json.dumps(
            {
                "status_ids": {status: [instance_id]},
                "resolved_ids": [instance_id] if resolved else [],
            }
        ),
        encoding="utf-8",
    )


def _fixture_run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    instance_id = "repo__project.task"
    _write_rollout(
        root,
        instance_id=instance_id,
        sample_index=0,
        resolved=True,
        prompt_tokens=(30, 30),
        completion_tokens=(20, 20),
    )
    _write_rollout(
        root,
        instance_id=instance_id,
        sample_index=1,
        resolved=True,
        prompt_tokens=(70, 70),
        completion_tokens=(30, 30),
    )
    _write_rollout(
        root,
        instance_id=instance_id,
        sample_index=2,
        resolved=False,
        prompt_tokens=(15, 15),
        completion_tokens=(10, 10),
    )
    shard = root / "collection" / "shard-0"
    (shard / "collection_manifest.json").write_text(
        json.dumps(
            {
                "expected_tasks": 1,
                "num_shards": 1,
                "shard_index": 0,
                "n_tasks": 1,
                "task_instance_ids": [instance_id],
                "temperatures": [0.6],
                "runs_per_temperature": 3,
                "total_samples_per_task": 3,
            }
        ),
        encoding="utf-8",
    )
    (shard / "summary.json").write_text("{}", encoding="utf-8")
    return root


def _write_collection_layout(
    root: Path,
    *,
    temperatures: list[float],
    runs_per_temperature: int = 4,
) -> None:
    collection = root / "collection" / "shard-0"
    collection.mkdir(parents=True, exist_ok=True)
    (collection / "collection_manifest.json").write_text(
        json.dumps(
            {
                "temperatures": temperatures,
                "runs_per_temperature": runs_per_temperature,
            }
        ),
        encoding="utf-8",
    )
    for index in range(len(temperatures) * runs_per_temperature):
        (collection / "samples" / f"sample-{index}").mkdir(parents=True)


def test_balanced_rollout_selection_adapts_to_two_or_four_temperatures(tmp_path):
    two_temperature_run = tmp_path / "two"
    _write_collection_layout(two_temperature_run, temperatures=[0.6, 0.7])
    assert select_sample_indices(two_temperature_run) == [0, 1, 4, 5]

    four_temperature_run = tmp_path / "four"
    _write_collection_layout(
        four_temperature_run,
        temperatures=[0.4, 0.5, 0.6, 0.7],
    )
    assert select_sample_indices(four_temperature_run) == [0, 4, 8, 12]


def test_explicit_rollout_selection_overrides_balancing(tmp_path):
    root = tmp_path / "run"
    _write_collection_layout(root, temperatures=[0.6, 0.7])

    assert parse_sample_indices("0, 2:4 6") == [0, 2, 4, 6]
    assert select_sample_indices(root, sample_indices=[2, 3, 6, 7]) == [2, 3, 6, 7]


def test_load_evaluated_trajectories_strips_provider_payload_and_trailing_user(tmp_path):
    records = load_evaluated_trajectories(_fixture_run(tmp_path))

    assert len(records) == 3
    assert records[0].total_tokens == 100
    assert records[0].completion_tokens == 40
    assert records[0].steps == 2
    assert records[0].prompt == [
        {"role": "system", "content": "debug agent"},
        {"role": "user", "content": "fix repo__project.task"},
    ]
    assert records[0].completion[-1]["role"] == "assistant"
    assert all(set(message) == {"role", "content"} for message in records[0].completion)


def test_dmpo_prefers_success_then_cost_and_never_cheap_failure(tmp_path):
    root = _fixture_run(tmp_path)
    output = tmp_path / "dmpo.jsonl"
    summary = build_dmpo_pairs(
        root,
        output_path=output,
        summary_path=tmp_path / "dmpo-summary.json",
        min_cost_ratio=1.1,
    )
    rows = read_jsonl(output)

    assert summary["pairs"] == 3
    assert summary["selected_sample_indices"] == [0, 1, 2]
    assert summary["preference_reason_counts"] == {
        "resolved_token_efficiency": 1,
        "task_success": 2,
    }
    efficiency = next(row for row in rows if row["preference_reason"].startswith("resolved"))
    assert efficiency["chosen_metadata"]["sample_index"] == 0
    assert efficiency["rejected_metadata"]["sample_index"] == 1
    success_rows = [row for row in rows if row["preference_reason"] == "task_success"]
    assert all(row["chosen_metadata"]["resolved"] for row in success_rows)
    assert all(not row["rejected_metadata"]["resolved"] for row in success_rows)


def test_depo_writes_unpaired_labels_and_dual_efficiency_features(tmp_path):
    root = _fixture_run(tmp_path)
    combined = tmp_path / "depo.jsonl"
    desirable = tmp_path / "desirable.jsonl"
    undesirable = tmp_path / "undesirable.jsonl"
    summary = build_depo_data(
        root,
        output_path=combined,
        desirable_output=desirable,
        undesirable_output=undesirable,
        summary_path=tmp_path / "depo-summary.json",
    )

    assert summary["format"] == "unpaired KTO-style binary trajectory labels"
    assert summary["selected_sample_indices"] == [0, 1, 2]
    assert summary["desirable"] == 2
    assert summary["undesirable"] == 1
    assert len(read_jsonl(desirable)) == 2
    assert len(read_jsonl(undesirable)) == 1
    first = read_jsonl(combined)[0]
    assert first["efficiency"]["total_tokens_per_step"] == 50
    assert first["efficiency"]["inverse_total_tokens_per_step"] == 0.02
    assert first["efficiency"]["inverse_steps"] == 0.5


@pytest.mark.parametrize("builder", ["dmpo", "depo"])
def test_preference_builders_reject_incomplete_selected_evaluations(tmp_path, builder):
    root = _fixture_run(tmp_path)
    instance_id = "repo__project.task"
    (root / "evaluation" / "sample-1" / "summary.json").write_text(
        json.dumps({"status_ids": {"error": [instance_id]}, "resolved_ids": []}),
        encoding="utf-8",
    )
    output_dir = tmp_path / builder

    with pytest.raises(ValueError, match="complete evaluated trajectories"):
        if builder == "dmpo":
            build_dmpo_pairs(
                root,
                output_path=output_dir / "pairs.jsonl",
                summary_path=output_dir / "summary.json",
            )
        else:
            build_depo_data(
                root,
                output_path=output_dir / "trajectories.jsonl",
                desirable_output=output_dir / "desirable.jsonl",
                undesirable_output=output_dir / "undesirable.jsonl",
                summary_path=output_dir / "summary.json",
            )

    assert not (output_dir / "summary.json").exists()


def test_completed_preference_artifacts_detect_tampering(tmp_path):
    root = _fixture_run(tmp_path)
    output = tmp_path / "preference-data" / "dmpo" / "pairs.jsonl"
    summary_path = output.parent / "summary.json"
    build_dmpo_pairs(root, output_path=output, summary_path=summary_path)

    summary = validate_preference_artifacts("dmpo", output.parent)
    assert summary["complete"] is True
    assert summary["artifacts"]["pairs"]["rows"] == 3

    output.write_text(output.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_preference_artifacts("dmpo", output.parent)
