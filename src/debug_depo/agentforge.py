"""Adapter for running an external AgentForge SWE-bench harness."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from debug_depo.constants import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_STEPS,
    DEFAULT_SWEBENCH_DATASET,
    DEFAULT_SWEBENCH_SPLIT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from debug_depo.utils import (
    ensure_dir,
    json_safe,
    load_hf_token_from_file,
    read_json,
    slugify,
    utc_now,
    write_json,
)


PATCH_KEYS = ("model_patch", "patch", "diff", "output_patch", "git_diff")
RESULT_JSON_NAMES = (
    "prediction.json",
    "preds.json",
    "result.json",
    "trajectory.json",
    "rollout.json",
    "output.json",
)
MINISWE_SUCCESS_STATUSES = {"Submitted"}
SWEBENCH_TESTBED_PATH = (
    "/opt/miniconda3/envs/testbed/bin:"
    "/opt/miniconda3/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
MINISWE_SUBSET_ALIASES = {
    "princeton-nlp/swe-bench": "full",
    "princeton-nlp/swe-bench_lite": "lite",
    "princeton-nlp/swe-bench_verified": "verified",
    "swe-bench/swe-smith": "smith",
}


@dataclass(frozen=True)
class AgentForgeConfig:
    model: str
    dataset: str = DEFAULT_SWEBENCH_DATASET
    split: str = DEFAULT_SWEBENCH_SPLIT
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    harness: str = "command"
    command: str | None = None
    cwd: str | None = None
    mini_model: str | None = None
    mini_config: str | None = None
    mini_runner: str = "pool_way"
    mini_environment_class: str | None = None
    mini_workers: int = 1
    mini_docker_start_concurrency: int = 1
    max_steps: int = DEFAULT_MAX_STEPS
    context_length: int = DEFAULT_CONTEXT_LENGTH
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    timeout_seconds: int = 7200
    mock: bool = False
    mock_patch: str = "empty"
    stream_output: bool = False


class AgentForgeRunError(RuntimeError):
    """Raised when the external AgentForge command cannot be run."""


class _FormatMap(dict[str, str]):
    def __missing__(self, key: str) -> str:
        raise KeyError(f"Unknown AgentForge command template field: {key}")


def miniswe_subset(dataset: str) -> str:
    """Return mini-swe's stable alias for known datasets, or the dataset path itself."""

    return MINISWE_SUBSET_ALIASES.get(dataset.lower(), dataset)


def prediction_record(instance: dict[str, Any], model: str, patch: str) -> dict[str, str]:
    return {
        "instance_id": str(instance["instance_id"]),
        "model_name_or_path": model,
        "model_patch": patch,
    }


def render_command(
    template: str,
    *,
    instance: dict[str, Any],
    task_json: Path,
    output_dir: Path,
    config: AgentForgeConfig,
) -> str:
    values = _FormatMap(
        instance_id=str(instance["instance_id"]),
        repo=str(instance.get("repo", "")),
        task_json=str(task_json),
        output_dir=str(output_dir),
        agentforge_repo=config.cwd or "",
        model=config.model,
        llm_base_url=config.llm_base_url or "",
        llm_api_key=config.llm_api_key or "",
        max_steps=str(config.max_steps),
        context_length=str(config.context_length),
        temperature=str(config.temperature),
        top_p=str(config.top_p),
    )
    return template.format_map(values)


def default_miniswe_model(model: str) -> str:
    """Return the default model string expected by mini-swe-agent-plus."""

    if model.startswith("hosted_vllm/"):
        return model
    return "hosted_vllm/" + model


def resolve_miniswe_config(config: AgentForgeConfig) -> str:
    if config.mini_config:
        return config.mini_config

    try:
        from minisweagent.config import builtin_config_dir
    except ImportError as exc:
        raise AgentForgeRunError(
            "mini-swe-agent-plus is not installed in this Python environment. "
            "On the cluster, run `bash cluster/setup_rollout_env.sh`. Otherwise install "
            "the official harness with `scripts/install_mini_swe_agent_plus.sh` or "
            "`pip install git+https://github.com/Kwai-Klear/mini-swe-agent-plus`."
        ) from exc

    config_path = Path(builtin_config_dir) / "extra" / "swebench_add_edit_tool.yaml"
    if not config_path.exists():
        raise AgentForgeRunError(f"mini-swe-agent-plus config not found: {config_path}")
    return str(config_path)


def miniswe_uses_singularity(config: AgentForgeConfig) -> bool:
    return config.mini_runner == "singularity" or config.mini_environment_class == "singularity"


def miniswe_edit_tool_startup_command() -> str:
    try:
        from minisweagent.environments import docker as miniswe_docker
    except ImportError as exc:
        raise AgentForgeRunError(
            "Could not import mini-swe-agent-plus Docker environment to locate "
            "the edit_via_str_replace helper."
        ) from exc

    tool_path = Path(miniswe_docker.__file__).with_name("str_replace.py")
    if not tool_path.exists():
        raise AgentForgeRunError(f"mini-swe-agent-plus edit helper not found: {tool_path}")
    tool_text = tool_path.read_text(encoding="utf-8")
    return (
        "cat > /testbed/edit_via_str_replace <<'PY'\n"
        f"{tool_text.rstrip()}\n"
        "PY\n"
        "chmod +x /testbed/edit_via_str_replace"
    )


def prepare_miniswe_config(config: AgentForgeConfig, output_dir: Path) -> str:
    """Write the effective mini-swe-agent config for this run."""

    try:
        import yaml
    except ImportError as exc:
        raise AgentForgeRunError(
            "PyYAML is required to adapt the mini-swe-agent-plus config. "
            "Install mini-swe-agent-plus with `scripts/install_mini_swe_agent_plus.sh`."
        ) from exc

    source_path = Path(resolve_miniswe_config(config))
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise AgentForgeRunError(f"mini-swe-agent-plus config must be a mapping: {source_path}")

    agent_config = payload.setdefault("agent", {})
    if not isinstance(agent_config, dict):
        raise AgentForgeRunError(f"mini-swe-agent-plus agent config must be a mapping: {source_path}")
    agent_config["step_limit"] = int(config.max_steps)

    model_config = payload.setdefault("model", {})
    if not isinstance(model_config, dict):
        raise AgentForgeRunError(f"mini-swe-agent-plus model config must be a mapping: {source_path}")
    model_kwargs = model_config.setdefault("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        raise AgentForgeRunError(
            f"mini-swe-agent-plus model_kwargs config must be a mapping: {source_path}"
        )
    model_kwargs["temperature"] = float(config.temperature)
    model_kwargs["top_p"] = float(config.top_p)

    if miniswe_uses_singularity(config):
        environment_config = payload.setdefault("environment", {})
        if not isinstance(environment_config, dict):
            raise AgentForgeRunError(
                f"mini-swe-agent-plus environment config must be a mapping: {source_path}"
            )
        env_config = environment_config.setdefault("env", {})
        if not isinstance(env_config, dict):
            raise AgentForgeRunError(
                f"mini-swe-agent-plus environment env config must be a mapping: {source_path}"
            )
        env_config.setdefault("PATH", SWEBENCH_TESTBED_PATH)
        env_config.setdefault("CONDA_DEFAULT_ENV", "testbed")
        env_config.setdefault("CONDA_PREFIX", "/opt/miniconda3/envs/testbed")
        env_config.setdefault("PYTHONNOUSERSITE", "1")

        run_config = payload.setdefault("run", {})
        if not isinstance(run_config, dict):
            raise AgentForgeRunError(f"mini-swe-agent-plus run config must be a mapping: {source_path}")
        startup_command = miniswe_edit_tool_startup_command()
        if existing := run_config.get("env_startup_command"):
            run_config["env_startup_command"] = f"{existing.rstrip()}\n{startup_command}"
        else:
            run_config["env_startup_command"] = startup_command

    generated_path = output_dir / "miniswe_config.generated.yaml"
    generated_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return str(generated_path)


def render_miniswe_command(
    *,
    instance: dict[str, Any],
    output_dir: Path,
    config: AgentForgeConfig,
) -> str:
    """Build the official mini-swe-agent-plus SWE-bench command for one instance."""

    instance_id = str(instance["instance_id"])
    model = config.mini_model or default_miniswe_model(config.model)

    mini_runner = config.mini_runner
    mini_environment_class = config.mini_environment_class
    if mini_runner == "singularity":
        if mini_environment_class not in {None, "", "singularity"}:
            raise AgentForgeRunError(
                "mini-swe-agent-plus runner 'singularity' cannot be combined with "
                f"environment class {mini_environment_class!r}."
            )
        mini_runner = "swebench"
        mini_environment_class = "singularity"
    if mini_runner not in {"pool_way", "swebench"}:
        raise AgentForgeRunError(
            "mini-swe-agent-plus runner must be 'pool_way', 'swebench', or 'singularity'."
        )
    if mini_runner == "pool_way" and mini_environment_class not in {None, "", "docker"}:
        raise AgentForgeRunError(
            "mini-swe-agent-plus pool_way runner only supports Docker task containers. "
            "Use --mini-runner swebench --mini-environment-class singularity on Apptainer clusters."
        )

    module = (
        "minisweagent.run.extra.swebench_pool_way"
        if mini_runner == "pool_way"
        else "minisweagent.run.extra.swebench"
    )
    command = [
        sys.executable,
        "-m",
        module,
        "--model",
        model,
        "--subset",
        miniswe_subset(config.dataset),
        "--split",
        config.split,
        "--output",
        str(output_dir),
        "--config",
        prepare_miniswe_config(config, output_dir),
        "--filter",
        f"^{re.escape(instance_id)}$",
        "--workers",
        str(config.mini_workers),
        "--redo-existing",
    ]
    if mini_runner == "pool_way":
        command.extend(
            [
                "--docker-start-concurrency",
                str(config.mini_docker_start_concurrency),
            ]
        )
    if mini_runner == "swebench" and mini_environment_class:
        command.extend(["--environment-class", mini_environment_class])
    return shlex.join(command)


def _find_patch_in_payload(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in PATCH_KEYS:
            value = payload.get(key)
            if isinstance(value, str):
                return value
        for value in payload.values():
            found = _find_patch_in_payload(value)
            if found is not None:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = _find_patch_in_payload(value)
            if found is not None:
                return found
    return None


def _json_objects_from_stdout(stdout: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(stdout):
        start = stdout.find("{", index)
        if start < 0:
            break
        try:
            payload, end = decoder.raw_decode(stdout[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(payload, dict):
            objects.append(payload)
        index = start + end
    return objects


def extract_patch(instance_dir: str | Path, stdout: str = "") -> tuple[str, str | None]:
    """Extract a SWE-bench patch from common AgentForge output locations."""

    root = Path(instance_dir)
    for suffix in (".patch", ".diff"):
        path = root / f"prediction{suffix}"
        if path.exists():
            return path.read_text(encoding="utf-8"), str(path)
        path = root / f"patch{suffix}"
        if path.exists():
            return path.read_text(encoding="utf-8"), str(path)

    for name in RESULT_JSON_NAMES:
        path = root / name
        if not path.exists():
            continue
        payload = read_json(path)
        patch = _find_patch_in_payload(payload)
        if patch is not None:
            return patch, str(path)

    for payload in reversed(_json_objects_from_stdout(stdout)):
        patch = _find_patch_in_payload(payload)
        if patch is not None:
            return patch, "stdout"

    return "", None


def miniswe_exit_statuses(instance_dir: str | Path) -> list[tuple[str, str]]:
    """Read mini-swe-agent-plus batch exit statuses without adding a YAML dependency."""

    root = Path(instance_dir)
    paths = sorted(
        root.glob("exit_statuses_*.yaml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    statuses: list[tuple[str, str]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if (
                not stripped
                or stripped == "instances_by_exit_status:"
                or stripped.startswith("- ")
                or not stripped.endswith(":")
            ):
                continue
            statuses.append((stripped[:-1].strip("\"'"), str(path)))
        if statuses:
            return statuses
    return statuses


def miniswe_failure(instance_dir: str | Path) -> tuple[str, str] | None:
    for status, path in miniswe_exit_statuses(instance_dir):
        if status not in MINISWE_SUCCESS_STATUSES:
            return status, path
    return None


def run_subprocess_with_optional_streaming(
    command: str,
    *,
    cwd: str | None,
    env: dict[str, str],
    timeout_seconds: int,
    stdout_path: Path,
    stderr_path: Path,
    stream_output: bool,
) -> subprocess.CompletedProcess[str]:
    if not stream_output:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        return completed

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def pump(source: Any, destination: Any, path: Path, chunks: list[str]) -> None:
        with path.open("w", encoding="utf-8", buffering=1) as handle:
            for line in iter(source.readline, ""):
                chunks.append(line)
                handle.write(line)
                destination.write(line)
                destination.flush()

    threads = [
        threading.Thread(
            target=pump,
            args=(process.stdout, sys.stdout, stdout_path, stdout_chunks),
            daemon=True,
        ),
        threading.Thread(
            target=pump,
            args=(process.stderr, sys.stderr, stderr_path, stderr_chunks),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        returncode = process.wait()
        message = f"\nCommand timed out after {timeout_seconds} seconds.\n"
        stderr_chunks.append(message)
        with stderr_path.open("a", encoding="utf-8") as handle:
            handle.write(message)
        sys.stderr.write(message)
        sys.stderr.flush()
    for thread in threads:
        thread.join(timeout=5)

    return subprocess.CompletedProcess(
        args=command,
        returncode=returncode,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )


def run_mock_agentforge(
    instance: dict[str, Any],
    instance_dir: Path,
    config: AgentForgeConfig,
) -> dict[str, Any]:
    patch = str(instance.get("patch", "")) if config.mock_patch == "gold" else ""
    trajectory = {
        "created_at": utc_now(),
        "instance_id": instance["instance_id"],
        "mode": "mock",
        "model": config.model,
        "status": "mocked",
        "patch_source": "gold" if config.mock_patch == "gold" else None,
        "patch": patch,
        "patch_chars": len(patch),
        "steps": [
            {
                "role": "assistant",
                "content": (
                    "Mock AgentForge run. No model call was made; this is for Mac and CI smoke tests."
                ),
            }
        ],
    }
    write_json(instance_dir / "trajectory.json", trajectory)
    return {
        "instance_id": str(instance["instance_id"]),
        "status": "mocked",
        "patch": patch,
        "patch_source": trajectory["patch_source"],
        "trajectory_path": str(instance_dir / "trajectory.json"),
    }


def run_external_agentforge(
    instance: dict[str, Any],
    instance_dir: Path,
    config: AgentForgeConfig,
) -> dict[str, Any]:
    task_json = instance_dir / "task.json"
    write_json(task_json, instance)

    if config.harness == "mini-swe-agent-plus":
        command = render_miniswe_command(
            instance=instance,
            output_dir=instance_dir,
            config=config,
        )
    elif config.command:
        command = render_command(
            config.command,
            instance=instance,
            task_json=task_json,
            output_dir=instance_dir,
            config=config,
        )
    else:
        raise AgentForgeRunError(
            "No AgentForge command was supplied. Use --agentforge-command, "
            "--harness mini-swe-agent-plus, or --mock."
        )

    env = os.environ.copy()
    load_hf_token_from_file(env=env)
    env.update(
        {
            "AGENTFORGE_MODEL": config.model,
            "AGENTFORGE_INSTANCE_ID": str(instance["instance_id"]),
            "AGENTFORGE_TASK_JSON": str(task_json),
            "AGENTFORGE_OUTPUT_DIR": str(instance_dir),
            "AGENTFORGE_MAX_STEPS": str(config.max_steps),
            "AGENTFORGE_CONTEXT_LENGTH": str(config.context_length),
            "AGENTFORGE_TEMPERATURE": str(config.temperature),
            "AGENTFORGE_TOP_P": str(config.top_p),
        }
    )
    if config.llm_base_url:
        env["AGENTFORGE_LLM_BASE_URL"] = config.llm_base_url
        env.setdefault("OPENAI_BASE_URL", config.llm_base_url)
        env.setdefault("OPENAI_API_BASE", config.llm_base_url)
        vllm_servers = instance_dir / "vllm_servers.txt"
        vllm_servers.write_text(config.llm_base_url.rstrip("/") + "\n", encoding="utf-8")
        env.setdefault("local_vllm_server_ips_filename", str(vllm_servers))
    if config.llm_api_key:
        env["AGENTFORGE_LLM_API_KEY"] = config.llm_api_key
        env.setdefault("OPENAI_API_KEY", config.llm_api_key)
    if config.harness == "mini-swe-agent-plus" and (
        config.mini_runner == "singularity" or config.mini_environment_class == "singularity"
    ):
        env.setdefault("MSWEA_SINGULARITY_EXECUTABLE", "apptainer")
        scratch = env.get("DEBUG_DEPO_SCRATCH")
        if scratch and "TMPDIR" not in env:
            tmpdir = ensure_dir(Path(scratch) / "tmp")
            env["TMPDIR"] = str(tmpdir)

    completed = run_subprocess_with_optional_streaming(
        command,
        cwd=config.cwd,
        env=env,
        timeout_seconds=config.timeout_seconds,
        stdout_path=instance_dir / "stdout.txt",
        stderr_path=instance_dir / "stderr.txt",
        stream_output=config.stream_output,
    )

    patch, patch_source = extract_patch(instance_dir, completed.stdout)
    mini_failure = miniswe_failure(instance_dir) if config.harness == "mini-swe-agent-plus" else None
    error_patch = patch if mini_failure is not None else ""
    if mini_failure is not None:
        patch = ""
        patch_source = None
    status = "completed" if completed.returncode == 0 and mini_failure is None else "error"
    trajectory = {
        "created_at": utc_now(),
        "command": command,
        "config": json_safe(asdict(config)),
        "instance_id": instance["instance_id"],
        "patch_chars": len(patch),
        "patch": patch,
        "patch_source": patch_source,
        "returncode": completed.returncode,
        "status": status,
        "stdout_path": str(instance_dir / "stdout.txt"),
        "stderr_path": str(instance_dir / "stderr.txt"),
    }
    if mini_failure is not None:
        trajectory["error_patch"] = error_patch
        trajectory["mini_swe_exit_status"] = mini_failure[0]
        trajectory["mini_swe_exit_status_path"] = mini_failure[1]
    write_json(instance_dir / "trajectory.json", trajectory)
    result = {
        "instance_id": str(instance["instance_id"]),
        "status": status,
        "patch": patch,
        "patch_source": patch_source,
        "returncode": completed.returncode,
        "trajectory_path": str(instance_dir / "trajectory.json"),
    }
    if mini_failure is not None:
        result["mini_swe_exit_status"] = mini_failure[0]
    return result


def run_agentforge_instance(
    instance: dict[str, Any],
    output_root: str | Path,
    config: AgentForgeConfig,
) -> dict[str, Any]:
    instance_id = str(instance["instance_id"])
    instance_dir = ensure_dir(Path(output_root) / "trajectories" / slugify(instance_id))
    if config.mock:
        return run_mock_agentforge(instance, instance_dir, config)
    return run_external_agentforge(instance, instance_dir, config)
