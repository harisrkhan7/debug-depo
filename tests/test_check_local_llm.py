from __future__ import annotations

from debug_depo import check_local_llm


def test_normalize_base_url_accepts_common_pasted_formats() -> None:
    expected = "http://127.0.0.1:8000/v1"
    inputs = (
        " http://127.0.0.1:8000/v1/// ",
        "[http://127.0.0.1:8000/v1](http://127.0.0.1:8000/v1)",
        "<http://127.0.0.1:8000/v1>",
    )

    assert [check_local_llm.normalize_base_url(value) for value in inputs] == [expected] * 3


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


def test_extract_completion_text_supports_chat_and_text_responses() -> None:
    payloads = (
        {"choices": [{"message": {"role": "assistant", "content": "OK"}}]},
        {"choices": [{"text": "OK"}]},
    )

    assert [check_local_llm.extract_completion_text(payload) for payload in payloads] == [
        "OK",
        "OK",
    ]
