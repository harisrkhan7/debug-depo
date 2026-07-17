from debug_depo.metrics import merge_predictions, summarize_predictions
from debug_depo.utils import write_jsonl


def test_merge_predictions_deduplicates_by_instance_id(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl(
        first,
        [
            {"instance_id": "a", "model_name_or_path": "m", "model_patch": ""},
            {"instance_id": "b", "model_name_or_path": "m", "model_patch": "diff"},
        ],
    )
    write_jsonl(
        second,
        [{"instance_id": "a", "model_name_or_path": "m", "model_patch": "new"}],
    )

    rows = merge_predictions([first, second], keep="last")
    summary = summarize_predictions(rows)

    assert rows[0]["model_patch"] == "new"
    assert rows[1]["instance_id"] == "b"
    assert summary["n_predictions"] == 2
    assert summary["n_with_patch"] == 2
