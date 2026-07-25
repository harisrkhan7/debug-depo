import json
from pathlib import Path

from debug_depo.build_depo_data import build_depo_data
from debug_depo.build_dmpo_pairs import build_dmpo_pairs
from debug_depo.preference_data import load_evaluated_trajectories
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
    return root


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
    assert summary["desirable"] == 2
    assert summary["undesirable"] == 1
    assert len(read_jsonl(desirable)) == 2
    assert len(read_jsonl(undesirable)) == 1
    first = read_jsonl(combined)[0]
    assert first["efficiency"]["total_tokens_per_step"] == 50
    assert first["efficiency"]["inverse_total_tokens_per_step"] == 0.02
    assert first["efficiency"]["inverse_steps"] == 0.5
