import json
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import debug_depo.agentforge as agentforge_module
import pytest
import yaml

from debug_depo.agentforge import (
    AgentForgeConfig,
    AgentForgeRunError,
    default_miniswe_model,
    extract_patch,
    miniswe_failure,
    miniswe_result_status,
    render_command,
    render_miniswe_command,
    run_agentforge_instance,
    run_subprocess_with_optional_streaming,
    terminate_active_subprocesses,
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
        f"run --id repo__repo-1 --task {tmp_path / 'task.json'} --out {tmp_path} --model model"
    )


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
        "instances_by_exit_status:\n    Uncaught NameError:\n    - repo__repo-1\n"
    )

    assert miniswe_failure(tmp_path) == ("Uncaught NameError", str(status_path))


def test_miniswe_status_classification_separates_model_and_infrastructure_outcomes():
    assert miniswe_result_status("Submitted") == "completed"
    assert miniswe_result_status("LimitsExceeded") == "model_terminated"
    assert miniswe_result_status("ContextWindowExceededError") == "model_terminated"
    assert miniswe_result_status("RetryError") == "error"


def test_render_miniswe_command_uses_official_module_and_filter(tmp_path, monkeypatch):
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
    monkeypatch.setenv("MINI_SWE_MODEL_TIMEOUT_SECONDS", "1200")

    command = render_miniswe_command(
        instance=instance(),
        output_dir=tmp_path,
        config=config,
    )

    assert "debug_depo.miniswe_task" in command
    assert "--runner pool_way" in command
    assert f"--task-json {tmp_path / 'task.json'}" in command
    assert "--instance-id repo__repo-1" in command
    assert "--subset verified" in command
    assert "--split test" in command
    assert "--filter '^repo__repo\\-1$'" in command
    parts = shlex.split(command)
    generated_config_path = parts[parts.index("--config") + 1]
    generated_config = yaml.safe_load(open(generated_config_path, encoding="utf-8"))
    assert generated_config["agent"]["step_limit"] == 7
    assert generated_config["model"]["model_kwargs"]["temperature"] == 0.25
    assert generated_config["model"]["model_kwargs"]["top_p"] == 0.8
    assert generated_config["model"]["model_kwargs"]["timeout"] == 1200
    assert default_miniswe_model("org/model") == "hosted_vllm/org/model"
    assert default_miniswe_model("hosted_vllm/org/model") == "hosted_vllm/org/model"


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

    assert "debug_depo.miniswe_task" in command
    assert "--runner swebench" in command
    assert "--runner pool_way" not in command
    assert "--environment-class singularity" in command
    assert "--docker-start-concurrency" not in command
    parts = shlex.split(command)
    generated_config_path = parts[parts.index("--config") + 1]
    generated_config = yaml.safe_load(open(generated_config_path, encoding="utf-8"))
    assert "edit_via_str_replace" in generated_config["run"]["env_startup_command"]
    assert "git checkout --force" not in generated_config["run"]["env_startup_command"]
    env_config = generated_config["environment"]["env"]
    assert env_config["PATH"].startswith("/opt/miniconda3/envs/testbed/bin:")
    assert env_config["CONDA_DEFAULT_ENV"] == "testbed"
    assert env_config["CONDA_PREFIX"] == "/opt/miniconda3/envs/testbed"
    assert env_config["PYTHONNOUSERSITE"] == "1"


def test_swesmith_miniswe_config_checks_out_task_before_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        agentforge_module,
        "miniswe_edit_tool_startup_command",
        lambda: "install-edit-tool",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {},
                "model": {"model_kwargs": {}},
                "run": {"env_startup_command": "existing-startup"},
            },
            sort_keys=False,
        )
    )
    config = AgentForgeConfig(
        model="model",
        dataset="SWE-bench/SWE-smith",
        mini_config=str(config_path),
        mini_runner="singularity",
        initialize_swesmith_task=True,
    )

    command = render_miniswe_command(
        instance=instance(),
        output_dir=tmp_path,
        config=config,
    )

    parts = shlex.split(command)
    generated = yaml.safe_load(Path(parts[parts.index("--config") + 1]).read_text(encoding="utf-8"))
    startup = generated["run"]["env_startup_command"]
    assert "cd /testbed || exit 20" in startup
    assert "git checkout --force repo__repo-1 || exit 21" in startup
    assert startup.index("git checkout") < startup.index("existing-startup")
    assert startup.index("existing-startup") < startup.index("install-edit-tool")


def test_swesmith_task_initialization_rejects_pool_way(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"agent": {}, "model": {"model_kwargs": {}}}))
    config = AgentForgeConfig(
        model="model",
        mini_config=str(config_path),
        mini_runner="pool_way",
        mini_environment_class="docker",
        initialize_swesmith_task=True,
    )

    try:
        render_miniswe_command(instance=instance(), output_dir=tmp_path, config=config)
    except AgentForgeRunError as exc:
        assert "does not execute task startup commands" in str(exc)
    else:
        raise AssertionError("Expected AgentForgeRunError")


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


def test_external_agentforge_command_reads_common_result_file_without_persisting_secret(
    tmp_path,
):
    helper = tmp_path / "helper.py"
    helper.write_text(
        "import json, pathlib, sys\n"
        "out = pathlib.Path(sys.argv[1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'result.json').write_text(json.dumps({'patch': 'diff --git a/a.py b/a.py\\n'}))\n"
    )
    command = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(helper))} {{output_dir}} {{llm_api_key}}"
    )

    result = run_agentforge_instance(
        instance(),
        tmp_path,
        AgentForgeConfig(
            model="model",
            command=command,
            llm_api_key="real-secret-key",
            timeout_seconds=10,
        ),
    )
    trajectory_path = tmp_path / "trajectories" / "repo__repo-1" / "trajectory.json"
    trajectory = read_json(trajectory_path)

    assert result["status"] == "completed"
    assert result["patch"] == "diff --git a/a.py b/a.py\n"
    assert trajectory["config"]["llm_api_key"] == "<redacted>"
    assert "real-secret-key" not in trajectory["command"]
    assert "real-secret-key" not in trajectory_path.read_text(encoding="utf-8")


def test_subprocess_timeout_terminates_the_complete_process_group(
    tmp_path,
    monkeypatch,
):
    class TimedOutProcess:
        pid = 4321
        returncode = None

        def __init__(self):
            self.communicate_calls = 0

        def communicate(self, timeout=None):
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise subprocess.TimeoutExpired("command", timeout)
            return "partial stdout", "partial stderr"

        def wait(self, timeout=None):
            self.returncode = -signal.SIGTERM
            return self.returncode

        def poll(self):
            return self.returncode

    process = TimedOutProcess()
    popen_kwargs = {}
    signals = []

    def fake_popen(*_args, **kwargs):
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr(agentforge_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        agentforge_module.os,
        "killpg",
        lambda process_group, sent_signal: signals.append((process_group, sent_signal)),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_subprocess_with_optional_streaming(
            "command",
            cwd=None,
            env={},
            timeout_seconds=1,
            stdout_path=tmp_path / "stdout.txt",
            stderr_path=tmp_path / "stderr.txt",
            stream_output=False,
        )

    assert popen_kwargs["start_new_session"] is True
    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    assert (tmp_path / "stdout.txt").read_text() == "partial stdout"
    assert (tmp_path / "stderr.txt").read_text() == "partial stderr"


def test_active_rollout_cleanup_terminates_registered_process_groups(monkeypatch):
    class ActiveProcess:
        pid = 4321
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("command", timeout)
            return self.returncode

    process = ActiveProcess()
    signals = []

    def fake_killpg(process_group, sent_signal):
        signals.append((process_group, sent_signal))
        if sent_signal == signal.SIGKILL:
            process.returncode = -sent_signal

    monkeypatch.setattr(agentforge_module.os, "killpg", fake_killpg)
    agentforge_module._register_active_subprocess(process)
    try:
        terminate_active_subprocesses(grace_seconds=0)
    finally:
        agentforge_module._unregister_active_subprocess(process)

    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
