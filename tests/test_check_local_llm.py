from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_local_llm.py"
SPEC = importlib.util.spec_from_file_location("check_local_llm", SCRIPT)
assert SPEC is not None
check_local_llm = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = check_local_llm
SPEC.loader.exec_module(check_local_llm)


def test_normalize_base_url_strips_trailing_slashes() -> None:
    assert check_local_llm.normalize_base_url(" http://127.0.0.1:8000/v1/// ") == (
        "http://127.0.0.1:8000/v1"
    )


def test_normalize_base_url_unwraps_markdown_link() -> None:
    assert check_local_llm.normalize_base_url(
        "[http://127.0.0.1:8000/v1](http://127.0.0.1:8000/v1)"
    ) == "http://127.0.0.1:8000/v1"


def test_normalize_base_url_unwraps_angle_brackets() -> None:
    assert check_local_llm.normalize_base_url("<http://127.0.0.1:8000/v1>") == (
        "http://127.0.0.1:8000/v1"
    )


def test_extract_model_ids_from_openai_style_response() -> None:
    payload = {
        "object": "list",
        "data": [
            {"id": "Kwai-Klear/Klear-AgentForge-8B-SFT", "object": "model"},
            {"id": "other-model", "object": "model"},
            {"object": "ignored"},
        ],
    }

    assert check_local_llm.extract_model_ids(payload) == [
        "Kwai-Klear/Klear-AgentForge-8B-SFT",
        "other-model",
    ]


def test_extract_completion_text_from_chat_response() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "OK",
                }
            }
        ]
    }

    assert check_local_llm.extract_completion_text(payload) == "OK"


def test_extract_completion_text_from_text_completion_shape() -> None:
    payload = {"choices": [{"text": "OK"}]}

    assert check_local_llm.extract_completion_text(payload) == "OK"
