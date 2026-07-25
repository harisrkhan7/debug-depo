import json
from types import SimpleNamespace

import pytest

import debug_depo.miniswe_task as miniswe_task


def task(instance_id="repo__project.task-1"):
    return {
        "instance_id": instance_id,
        "problem_statement": "Fix the task",
        "image_name": "example/image:latest",
    }


def test_load_task_json_validates_the_expected_instance(tmp_path):
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task()), encoding="utf-8")

    assert miniswe_task.load_task_json(
        task_path,
        expected_instance_id="repo__project.task-1",
    ) == task()

    with pytest.raises(ValueError, match="does not match"):
        miniswe_task.load_task_json(
            task_path,
            expected_instance_id="repo__project.task-2",
        )


def test_adapter_injects_one_task_into_the_official_runner(tmp_path, monkeypatch):
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task()), encoding="utf-8")
    calls = {}

    def original_load_dataset(*_args, **_kwargs):
        raise AssertionError("The upstream dataset loader should be replaced")

    module = SimpleNamespace(
        __name__="minisweagent.run.extra.swebench",
        load_dataset=original_load_dataset,
    )

    def app(*, args, prog_name, standalone_mode):
        calls["args"] = args
        calls["prog_name"] = prog_name
        calls["standalone_mode"] = standalone_mode
        calls["instances"] = module.load_dataset(
            "SWE-bench/SWE-smith-py",
            split="train",
        )

    module.app = app
    monkeypatch.setattr(
        miniswe_task.importlib,
        "import_module",
        lambda _name: module,
    )

    result = miniswe_task.main(
        [
            "--task-json",
            str(task_path),
            "--instance-id",
            "repo__project.task-1",
            "--runner",
            "swebench",
            "--",
            "--model",
            "model",
        ]
    )

    assert result == 0
    assert calls == {
        "args": ["--model", "model"],
        "prog_name": "minisweagent.run.extra.swebench",
        "standalone_mode": False,
        "instances": [task()],
    }
    assert module.load_dataset is original_load_dataset
