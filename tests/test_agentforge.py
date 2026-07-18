import json
import shlex
import sys

import yaml
import debug_depo.agentforge as agentforge_module

from debug_depo.agentforge import (
    AgentForgeConfig,
    AgentForgeRunError,
    default_miniswe_model,
    extract_patch,
    miniswe_failure,
    render_miniswe_command,
    render_command,
    run_agentforge_instance,
)
from debug_depo.utils import read_json


def instance():
    return {
        "instance_id": "repo__repo-1",
        "repo": "repo/repo",
        "problem_statement": "Fix it",
        "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
    }


def test_render_command_exposes_expected_fields(tmp_path):
    config = AgentForgeConfig(
        model="model",
        llm_base_url="http://localhost:8000/v1",
        llm_api_key="key",
    )

    command = render_command(
        "run --id {instance_id} --task {task_json} --out {output_dir} --model {model}",
        instance=instance(),
        task_json=tmp_path / "task.json",
        output_dir=tmp_path,
        config=config,
    )

    assert command == (
        f"run --id repo__repo-1 --task {tmp_path / 'task.json'} "
        f"--out {tmp_path} --model model"
    )


def test_extract_patch_from_json_file(tmp_path):
    expected = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
    (tmp_path / "result.json").write_text(json.dumps({"model_patch": expected}))

    patch, source = extract_patch(tmp_path)

    assert patch == expected
    assert source == str(tmp_path / "result.json")


def test_extract_patch_from_official_miniswe_preds_json(tmp_path):
    expected = "diff --git a/a.py b/a.py\n"
    (tmp_path / "preds.json").write_text(
        json.dumps({"repo__repo-1": {"instance_id": "repo__repo-1", "model_patch": expected}})
    )

    patch, source = extract_patch(tmp_path)

    assert patch == expected
    assert source == str(tmp_path / "preds.json")


def test_miniswe_failure_reads_uncaught_exit_status(tmp_path):
    status_path = tmp_path / "exit_statuses_1.yaml"
    status_path.write_text(
        "instances_by_exit_status:\n"
        "    Uncaught NameError:\n"
        "    - repo__repo-1\n"
    )

    assert miniswe_failure(tmp_path) == ("Uncaught NameError", str(status_path))


def test_miniswe_submitted_status_is_not_failure(tmp_path):
    (tmp_path / "exit_statuses_1.yaml").write_text(
        "instances_by_exit_status:\n"
        "    Submitted:\n"
        "    - repo__repo-1\n"
    )

    assert miniswe_failure(tmp_path) is None


def test_render_miniswe_command_uses_official_module_and_filter(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {"step_limit": 200, "cost_limit": 3.0},
                "model": {"model_kwargs": {"temperature": 1.0, "drop_params": True}},
            },
            sort_keys=False,
        )
    )
    config = AgentForgeConfig(
        model="Kwai-Klear/Klear-AgentForge-8B-SFT",
        mini_config=str(config_path),
        max_steps=7,
        temperature=0.25,
        top_p=0.8,
    )

    command = render_miniswe_command(
        instance=instance(),
        output_dir=tmp_path,
        config=config,
    )

    assert "minisweagent.run.extra.swebench_pool_way" in command
    assert "--subset verified" in command
    assert "--split test" in command
    assert "--filter '^repo__repo\\-1$'" in command
    parts = shlex.split(command)
    generated_config_path = parts[parts.index("--config") + 1]
    generated_config = yaml.safe_load(open(generated_config_path, encoding="utf-8"))
    assert generated_config["agent"]["step_limit"] == 7
    assert generated_config["model"]["model_kwargs"]["temperature"] == 0.25
    assert generated_config["model"]["model_kwargs"]["top_p"] == 0.8
    assert default_miniswe_model("org/model") == "hosted_vllm/org/model"
    assert default_miniswe_model("hosted_vllm/org/model") == "hosted_vllm/org/model"


def test_render_miniswe_command_uses_configured_dataset_and_split(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"agent": {}, "model": {"model_kwargs": {}}}),
        encoding="utf-8",
    )
    config = AgentForgeConfig(
        model="model",
        dataset="SWE-bench/SWE-smith-py",
        split="train",
        mini_config=str(config_path),
    )

    command = render_miniswe_command(
        instance=instance(),
        output_dir=tmp_path,
        config=config,
    )

    parts = shlex.split(command)
    assert parts[parts.index("--subset") + 1] == "SWE-bench/SWE-smith-py"
    assert parts[parts.index("--split") + 1] == "train"


def test_render_miniswe_command_can_use_singularity_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(
        agentforge_module,
        "miniswe_edit_tool_startup_command",
        lambda: "cat > /testbed/edit_via_str_replace <<'PY'\nprint('ok')\nPY",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {"step_limit": 200},
                "model": {"model_kwargs": {"temperature": 1.0}},
            },
            sort_keys=False,
        )
    )
    config = AgentForgeConfig(
        model="Kwai-Klear/Klear-AgentForge-8B-SFT",
        mini_config=str(config_path),
        mini_runner="singularity",
    )

    command = render_miniswe_command(
        instance=instance(),
        output_dir=tmp_path,
        config=config,
    )

    assert "minisweagent.run.extra.swebench" in command
    assert "minisweagent.run.extra.swebench_pool_way" not in command
    assert "--environment-class singularity" in command
    assert "--docker-start-concurrency" not in command
    parts = shlex.split(command)
    generated_config_path = parts[parts.index("--config") + 1]
    generated_config = yaml.safe_load(open(generated_config_path, encoding="utf-8"))
    assert "edit_via_str_replace" in generated_config["run"]["env_startup_command"]
    env_config = generated_config["environment"]["env"]
    assert env_config["PATH"].startswith("/opt/miniconda3/envs/testbed/bin:")
    assert env_config["CONDA_DEFAULT_ENV"] == "testbed"
    assert env_config["CONDA_PREFIX"] == "/opt/miniconda3/envs/testbed"
    assert env_config["PYTHONNOUSERSITE"] == "1"


def test_singularity_runner_rejects_docker_environment_class(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"agent": {}, "model": {"model_kwargs": {}}}))
    config = AgentForgeConfig(
        model="Kwai-Klear/Klear-AgentForge-8B-SFT",
        mini_config=str(config_path),
        mini_runner="singularity",
        mini_environment_class="docker",
    )

    try:
        render_miniswe_command(instance=instance(), output_dir=tmp_path, config=config)
    except AgentForgeRunError as exc:
        assert "cannot be combined" in str(exc)
    else:
        raise AssertionError("Expected AgentForgeRunError")


def test_mock_agentforge_can_emit_gold_patch(tmp_path):
    result = run_agentforge_instance(
        instance(),
        tmp_path,
        AgentForgeConfig(model="model", mock=True, mock_patch="gold"),
    )

    trajectory = read_json(tmp_path / "trajectories" / "repo__repo-1" / "trajectory.json")
    assert result["patch"] == instance()["patch"]
    assert trajectory["mode"] == "mock"


def test_external_agentforge_command_reads_common_result_file(tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text(
        "import json, pathlib, sys\n"
        "out = pathlib.Path(sys.argv[1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'result.json').write_text(json.dumps({'patch': 'diff --git a/a.py b/a.py\\n'}))\n"
    )
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(helper))} {{output_dir}}"

    result = run_agentforge_instance(
        instance(),
        tmp_path,
        AgentForgeConfig(model="model", command=command, timeout_seconds=10),
    )

    assert result["status"] == "completed"
    assert result["patch"] == "diff --git a/a.py b/a.py\n"
