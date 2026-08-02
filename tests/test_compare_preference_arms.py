import csv
import json

import pytest

from debug_depo.compare_preference_arms import build_parser, compare_preference_arms


COLUMNS = (
    "instance_id",
    "evaluation_status",
    "resolved",
    "agent_action_steps",
    "prompt_tokens_total",
    "completion_tokens_total",
    "total_tokens",
)


def write_arm(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def arm_rows(outcomes, totals):
    rows = []
    for index, (resolved, total) in enumerate(zip(outcomes, totals, strict=True)):
        rows.append(
            {
                "instance_id": f"task-{index}",
                "evaluation_status": "resolved" if resolved else "unresolved",
                "resolved": resolved,
                "agent_action_steps": index + 1,
                "prompt_tokens_total": total - 10,
                "completion_tokens_total": 10,
                "total_tokens": total,
            }
        )
    return rows


def test_success_tolerance_defaults_to_three_percentage_points():
    args = build_parser().parse_args(
        [
            "--baseline",
            "sft=baseline.csv",
            "--arm",
            "dmpo=dmpo.csv",
            "--output",
            "comparison.json",
        ]
    )

    assert args.success_tolerance == 0.03


def test_compare_selects_lowest_cost_success_noninferior_arm(tmp_path):
    baseline = tmp_path / "baseline.csv"
    efficient = tmp_path / "efficient.csv"
    cheap_regression = tmp_path / "cheap-regression.csv"
    output = tmp_path / "comparison.json"
    write_arm(baseline, arm_rows([True, True, False, False], [100, 200, 300, 400]))
    write_arm(efficient, arm_rows([True, True, False, False], [60, 120, 150, 180]))
    write_arm(cheap_regression, arm_rows([True, False, False, False], [20, 20, 20, 20]))

    summary = compare_preference_arms(
        baseline=("sft", baseline),
        arms=[("dmpo", efficient), ("regressed", cheap_regression)],
        output=output,
        success_tolerance=0.1,
        expected_tasks=4,
    )

    assert summary["expected_tasks"] == 4
    assert len(summary["task_matrix_sha256"]) == 64
    assert summary["success_threshold"] == 0.4
    assert summary["selected_arm"] == "dmpo"
    assert [row["name"] for row in summary["eligible_ranking"]] == ["dmpo", "sft"]
    by_name = {row["name"]: row for row in summary["arms"]}
    assert by_name["sft"]["efficiency"]["total_tokens_per_resolved_task"] == 500
    assert by_name["dmpo"]["efficiency"]["total_tokens_per_resolved_task"] == 255
    assert (
        by_name["dmpo"]["paired_deltas_vs_baseline"]["all"]["total_tokens"]["mean"]
        == -122.5
    )
    assert by_name["dmpo"]["resolution_transitions_vs_baseline"] == {
        "both_resolved": 2,
        "both_unresolved": 2,
        "gained": 0,
        "lost": 0,
    }
    assert by_name["regressed"]["success_noninferior"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == summary


def test_compare_rejects_a_mismatched_task_matrix(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    write_arm(baseline, arm_rows([True, False], [100, 200]))
    rows = arm_rows([True, False], [90, 180])
    rows[-1]["instance_id"] = "unexpected-task"
    write_arm(candidate, rows)

    with pytest.raises(ValueError, match="does not match baseline task IDs"):
        compare_preference_arms(
            baseline=("sft", baseline),
            arms=[("dmpo", candidate)],
            output=tmp_path / "comparison.json",
        )


def test_compare_rejects_unscored_evaluations(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    rows = arm_rows([True, False], [100, 200])
    write_arm(baseline, rows)
    rows[-1]["evaluation_status"] = "missing_report"
    write_arm(candidate, rows)

    with pytest.raises(ValueError, match="unscored evaluation outcomes"):
        compare_preference_arms(
            baseline=("sft", baseline),
            arms=[("dmpo", candidate)],
            output=tmp_path / "comparison.json",
        )


def test_compare_rejects_an_unexpected_task_count(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    write_arm(baseline, arm_rows([True, False], [100, 200]))
    write_arm(candidate, arm_rows([True, False], [90, 180]))

    with pytest.raises(ValueError, match="Expected 500 tasks, found 2"):
        compare_preference_arms(
            baseline=("sft", baseline),
            arms=[("dmpo", candidate)],
            output=tmp_path / "comparison.json",
            expected_tasks=500,
        )


def test_compare_does_not_rank_incomplete_total_token_telemetry(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    write_arm(baseline, arm_rows([True, False], [100, 200]))
    rows = arm_rows([True, False], [90, 180])
    rows[-1]["total_tokens"] = ""
    write_arm(candidate, rows)

    summary = compare_preference_arms(
        baseline=("sft", baseline),
        arms=[("dmpo", candidate)],
        output=tmp_path / "comparison.json",
        allow_incomplete_telemetry=True,
    )

    by_name = {row["name"]: row for row in summary["arms"]}
    assert summary["selected_arm"] == "sft"
    assert by_name["dmpo"]["success_noninferior"] is True
    assert by_name["dmpo"]["rankable"] is False
    assert by_name["dmpo"]["selection_eligible"] is False


def test_compare_rejects_incomplete_efficiency_telemetry_by_default(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    write_arm(baseline, arm_rows([True, False], [100, 200]))
    rows = arm_rows([True, False], [90, 180])
    rows[-1]["completion_tokens_total"] = ""
    write_arm(candidate, rows)

    with pytest.raises(ValueError, match="incomplete efficiency telemetry"):
        compare_preference_arms(
            baseline=("sft", baseline),
            arms=[("dmpo", candidate)],
            output=tmp_path / "comparison.json",
        )
