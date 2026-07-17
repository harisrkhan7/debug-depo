"""Small shared utilities for the AgentForge SWE-bench runner."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, MutableMapping


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: Any, default: str = "item") -> str:
    text = str(value or default).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or default


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return output_path


def load_hf_token_from_file(
    *,
    env: MutableMapping[str, str] | None = None,
    token_file: str | Path | None = None,
) -> str | None:
    """Load the repo's saved Hugging Face token into an environment mapping."""

    target_env = os.environ if env is None else env
    token = target_env.get("HF_TOKEN") or target_env.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        target_env.setdefault("HF_TOKEN", token)
        target_env.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        return token

    token_path = Path(
        token_file
        or target_env.get("HF_TOKEN_FILE")
        or Path.home() / ".config" / "debug-depo" / "hf_token"
    ).expanduser()
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not token:
        return None

    target_env["HF_TOKEN"] = token
    target_env.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return token


def iter_json_files(directory: str | Path) -> Iterator[Path]:
    yield from sorted(Path(directory).rglob("*.json"))


def json_safe(value: Any, max_string_length: int = 12000) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= max_string_length:
            return value
        return value[:max_string_length] + "...[truncated]"
    if isinstance(value, dict):
        return {str(key): json_safe(item, max_string_length) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item, max_string_length) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def first_present(mapping: dict[str, Any] | None, keys: Iterable[str], default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default
