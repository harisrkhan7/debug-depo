import json

from debug_depo.data import (
    instance_ids,
    load_swebench_tasks,
    read_instance_ids_file,
    select_tasks,
    write_task_selection,
)
from debug_depo.utils import load_hf_token_from_file


def task(instance_id):
    return {
        "instance_id": instance_id,
        "repo": "example/repo",
        "problem_statement": "Fix it",
        "patch": "diff --git a/a.py b/a.py\n",
    }


def test_load_local_jsonl_tasks(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(task("a")) + "\n" + json.dumps(task("b")) + "\n")

    tasks = load_swebench_tasks(str(path), "test")

    assert instance_ids(tasks) == ["a", "b"]


def test_select_tasks_preserves_requested_order_and_shards():
    tasks = [task("a"), task("b"), task("c"), task("d")]

    selected = select_tasks(
        tasks,
        instance_ids=["d", "b", "a"],
        num_shards=2,
        shard_index=1,
    )

    assert instance_ids(selected) == ["b"]


def test_verified_sized_selection_splits_evenly_across_ten_shards():
    tasks = [task(f"repo__repo-{index}") for index in range(500)]

    shards = [
        select_tasks(tasks, num_shards=10, shard_index=shard_index)
        for shard_index in range(10)
    ]

    assert [len(shard) for shard in shards] == [50] * 10
    selected_ids = [instance_id for shard in shards for instance_id in instance_ids(shard)]
    assert len(selected_ids) == len(set(selected_ids)) == 500
    assert set(selected_ids) == set(instance_ids(tasks))


def test_read_instance_ids_file_supports_text_and_json(tmp_path):
    text_path = tmp_path / "ids.txt"
    text_path.write_text("# comment\na\n\nb\n")
    json_path = tmp_path / "ids.json"
    json_path.write_text(json.dumps({"instance_ids": ["c", "d"]}))

    assert read_instance_ids_file(text_path) == ["a", "b"]
    assert read_instance_ids_file(json_path) == ["c", "d"]


def test_write_task_selection_writes_jsonl_and_ids(tmp_path):
    summary = write_task_selection([task("a"), task("b")], tmp_path, name="subset")

    assert summary["n_tasks"] == 2
    assert (tmp_path / "subset.jsonl").is_file()
    assert (tmp_path / "subset_instance_ids.txt").read_text().splitlines() == ["a", "b"]


def test_load_hf_token_from_file_sets_hub_env_names(tmp_path):
    token_path = tmp_path / "hf_token"
    token_path.write_text("hf_test_token\n")
    env = {"HF_TOKEN_FILE": str(token_path)}

    token = load_hf_token_from_file(env=env)

    assert token == "hf_test_token"
    assert env["HF_TOKEN"] == "hf_test_token"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "hf_test_token"
