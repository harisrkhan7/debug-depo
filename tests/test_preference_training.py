import pytest

from debug_depo.preference_training import (
    EpochShuffleSampler,
    PreferenceDataset,
    _ensure_compatible_trial_config,
    _latest_checkpoint,
    depo_efficiency_bonus,
    dmpo_turn_weights,
    tokenize_trajectory,
    validate_training_rows,
)


class PrefixStableTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return list(rendered.encode())


def test_dmpo_turn_weights_match_paper_discount():
    assert dmpo_turn_weights(3, 1) == pytest.approx([1, 2 / 3, 1 / 3])
    assert dmpo_turn_weights(3, 0.5) == pytest.approx([1, 3 / 7, 1 / 7])
    assert dmpo_turn_weights(3, 0) == pytest.approx([1, 0, 0])


def test_dmpo_turn_weights_reject_invalid_values():
    with pytest.raises(ValueError, match="positive"):
        dmpo_turn_weights(0, 0.7)
    with pytest.raises(ValueError, match="gamma"):
        dmpo_turn_weights(1, -0.1)


def test_trial_directory_rejects_changed_hyperparameters(tmp_path):
    output = tmp_path / "trial"
    output.mkdir()
    payload = {
        "schema_version": 1,
        "dmpo_trial_name": "gamma07",
        "depo_trial_name": "default",
        "data_sha256": "abc",
        "config": {"beta": 0.1, "gamma": 0.7},
    }

    _ensure_compatible_trial_config(output, payload)
    _ensure_compatible_trial_config(output, payload)

    changed = {**payload, "config": {**payload["config"], "gamma": 0.9}}
    with pytest.raises(ValueError, match="gamma"):
        _ensure_compatible_trial_config(output, changed)


def test_latest_checkpoint_ignores_interrupted_higher_number(tmp_path):
    complete = tmp_path / "checkpoint-10"
    complete.mkdir()
    (complete / "trainer_state.json").write_text(
        '{"epoch": 0, "batch_in_epoch": 10, "global_step": 10}',
        encoding="utf-8",
    )
    interrupted = tmp_path / "checkpoint-20"
    interrupted.mkdir()

    assert _latest_checkpoint(tmp_path) == complete


def test_epoch_shuffle_is_exact_when_resuming_later_epochs():
    sampler = EpochShuffleSampler(20, seed=42)
    sampler.set_epoch(2)
    uninterrupted = list(sampler)

    resumed = EpochShuffleSampler(20, seed=42)
    resumed.set_epoch(2)
    assert list(resumed)[7:] == uninterrupted[7:]
    assert uninterrupted != list(EpochShuffleSampler(20, seed=42))


def test_depo_bonus_is_only_applied_to_desirable_rows():
    row = {
        "label": "desirable",
        "efficiency": {
            "inverse_total_tokens_per_step": 0.02,
            "inverse_completion_tokens_per_step": 0.05,
            "inverse_steps": 0.25,
        },
    }
    assert depo_efficiency_bonus(
        row,
        alpha_tokens=2,
        alpha_steps=3,
        token_metric="total_tokens",
    ) == pytest.approx(0.79)
    row["label"] = "undesirable"
    assert depo_efficiency_bonus(
        row,
        alpha_tokens=2,
        alpha_steps=3,
        token_metric="total_tokens",
    ) == 0


def test_tokenize_trajectory_masks_prompt_and_observations_and_weights_turns():
    tokenizer = PrefixStableTokenizer()
    prompt = [{"role": "user", "content": "task"}]
    completion = [
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "observation"},
        {"role": "assistant", "content": "second"},
    ]
    tokenized = tokenize_trajectory(
        tokenizer,
        prompt,
        completion,
        max_length=1000,
        turn_weights=[1.0, 0.5],
    )
    rendered = bytes(tokenized["input_ids"]).decode()
    weights = tokenized["token_weights"]

    first_start = rendered.index("first")
    observation_start = rendered.index("observation")
    second_start = rendered.index("second")
    assert weights[first_start] == 1.0
    assert weights[observation_start] == 0.0
    assert weights[second_start] == 0.5
    assert all(weight == 0 for weight in weights[: rendered.index("<assistant>")])


def test_validate_depo_requires_both_labels():
    base = {
        "id": "x",
        "label": "desirable",
        "prompt": [{"role": "user", "content": "task"}],
        "completion": [{"role": "assistant", "content": "answer"}],
        "efficiency": {},
    }
    with pytest.raises(ValueError, match="both desirable and undesirable"):
        validate_training_rows("depo", [base])
    summary = validate_training_rows(
        "depo",
        [base, {**base, "id": "y", "label": "undesirable"}],
    )
    assert summary["labels"] == {"desirable": 1, "undesirable": 1}


def test_depo_dataset_builds_mismatched_kto_kl_trajectory():
    rows = [
        {
            "id": "x",
            "label": "desirable",
            "prompt": [{"role": "user", "content": "first task"}],
            "completion": [{"role": "assistant", "content": "first answer"}],
            "efficiency": {
                "inverse_total_tokens_per_step": 0.1,
                "inverse_steps": 1,
            },
        },
        {
            "id": "y",
            "label": "undesirable",
            "prompt": [{"role": "user", "content": "second task"}],
            "completion": [{"role": "assistant", "content": "second answer"}],
            "efficiency": {
                "inverse_total_tokens_per_step": 0.1,
                "inverse_steps": 1,
            },
        },
    ]
    item = PreferenceDataset(
        rows,
        PrefixStableTokenizer(),
        objective="depo",
        max_length=1000,
        gamma=0.7,
        alpha_tokens=2,
        alpha_steps=2,
        token_metric="total_tokens",
    )[0]

    kl_text = bytes(item["kl"]["input_ids"]).decode()
    assert "first task" in kl_text
    assert "second answer" in kl_text
