from debug_depo.efficiency import distribution, summarize_efficiency


FIELDS = {
    "action_steps": "steps",
    "prompt_tokens": "prompt",
    "completion_tokens": "completion",
    "total_tokens": "total",
}


def test_distribution_reports_coverage_and_interpolated_p90():
    summary = distribution([1.0, 2.0, 10.0], expected=4)

    assert summary == {
        "available": 3,
        "missing": 1,
        "coverage": 0.75,
        "sum": 13.0,
        "min": 1.0,
        "mean": 13.0 / 3,
        "median": 2.0,
        "p90": 8.4,
        "max": 10.0,
    }


def test_efficiency_summary_separates_all_and_resolved_and_requires_cost_coverage():
    rows = [
        {
            "resolved": True,
            "steps": 1,
            "prompt": 10,
            "completion": 2,
            "total": 12,
        },
        {
            "resolved": False,
            "steps": 3,
            "prompt": 20,
            "completion": 4,
            "total": 24,
        },
        {
            "resolved": True,
            "steps": 2,
            "prompt": "",
            "completion": 3,
            "total": "",
        },
    ]

    summary = summarize_efficiency(rows, fields=FIELDS)

    assert summary["trajectories"] == 3
    assert summary["resolved_trajectories"] == 2
    assert summary["resolution_rate"] == 2 / 3
    assert summary["all"]["action_steps"]["median"] == 2
    assert summary["all"]["total_tokens"]["available"] == 2
    assert summary["all"]["total_tokens"]["missing"] == 1
    assert summary["resolved"]["completion_tokens"]["mean"] == 2.5
    assert summary["total_tokens_per_resolved_task"] is None


def test_efficiency_summary_uses_all_attempt_cost_per_resolved_task():
    rows = [
        {
            "resolved": True,
            "steps": 1,
            "prompt": 8,
            "completion": 2,
            "total": 10,
        },
        {
            "resolved": False,
            "steps": 2,
            "prompt": 16,
            "completion": 4,
            "total": 20,
        },
    ]

    summary = summarize_efficiency(rows, fields=FIELDS)

    assert summary["total_tokens_per_resolved_task"] == 30
