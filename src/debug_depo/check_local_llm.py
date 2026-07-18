"""Smoke-test an OpenAI-compatible local LLM server.

The check intentionally uses only the Python standard library so it can run
before project dependencies are installed. It verifies both the model-listing
endpoint and a tiny chat completion, which catches the common case where the
server accepts HTTP requests but generation is still broken or not ready.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_CHECK_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
DEFAULT_PROMPT = "Reply with OK."
DEFAULT_TIMEOUT_SECONDS = 600
MARKDOWN_LINK_RE = re.compile(r"^\[([^\]]+)\]\(([^)]+)\)$")


class CheckFailed(RuntimeError):
    """Raised when the server is reachable but fails the smoke check."""


@dataclass(frozen=True)
class JsonResponse:
    status: int
    elapsed_s: float
    body: dict[str, Any]


def normalize_base_url(value: str | None) -> str:
    value = (value or DEFAULT_BASE_URL).strip()
    markdown_match = MARKDOWN_LINK_RE.match(value)
    if markdown_match:
        value = markdown_match.group(2).strip()
    elif value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
    value = value.rstrip("/")
    return value or DEFAULT_BASE_URL


def endpoint(base_url: str, path: str) -> str:
    return f"{normalize_base_url(base_url)}/{path.lstrip('/')}"


def response_excerpt(raw: bytes, limit: int = 500) -> str:
    text = raw.decode("utf-8", errors="replace").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def parse_json_object(raw: bytes, url: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        excerpt = response_excerpt(raw)
        detail = f" Response started with: {excerpt}" if excerpt else ""
        raise CheckFailed(f"{url} returned non-JSON data.{detail}") from exc
    if not isinstance(parsed, dict):
        raise CheckFailed(f"{url} returned JSON {type(parsed).__name__}; expected an object.")
    return parsed


def explain_http_error(exc: HTTPError, raw: bytes, url: str) -> str:
    hints = {
        401: "The server rejected the API key/header.",
        403: "The server rejected this request; check auth or model access.",
        404: "The path was not found. Check that --base-url includes /v1.",
        429: "The server is rate-limiting or overloaded.",
        500: "The server crashed while handling the request.",
        503: "The server is not ready or is overloaded.",
    }
    hint = hints.get(exc.code, "The server returned an error response.")
    message = f"{url} returned HTTP {exc.code} {exc.reason}. {hint}"
    excerpt = response_excerpt(raw)
    if excerpt:
        message += f"\nResponse body: {excerpt}"
    return message


def explain_connection_error(exc: BaseException, url: str, timeout: float) -> str:
    reason = getattr(exc, "reason", exc)
    reason_text = str(reason)
    lower_reason = reason_text.lower()

    if "connection refused" in lower_reason:
        hint = (
            "Nothing is listening at that host/port yet. Start the server first, "
            "for example: MODEL=Kwai-Klear/Klear-AgentForge-8B-SFT PORT=8000 "
            "bash scripts/serve_local_llm.sh"
        )
    elif "timed out" in lower_reason or isinstance(exc, (socket.timeout, TimeoutError)):
        hint = f"The request timed out after {timeout:g}s; the model may still be loading."
        if "chat/completions" in url:
            hint += (
                " If scripts/serve_local_llm.sh is still printing "
                "'Server is listening; checking generation', wait for "
                "'Local LLM ready' before running this notebook. The launcher "
                "uses one generation worker, so a second check can queue behind "
                "the launcher's own health check."
            )
    elif "name or service not known" in lower_reason or "nodename nor servname" in lower_reason:
        hint = "The host name could not be resolved."
    else:
        hint = "The server is not reachable from this process."

    return f"Could not reach {url}: {reason_text}. {hint}"


def request_json(
    url: str,
    *,
    method: str,
    timeout: float,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
) -> JsonResponse:
    body_bytes = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body_bytes = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(url, data=body_bytes, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            elapsed_s = time.perf_counter() - started
            return JsonResponse(
                status=response.status,
                elapsed_s=elapsed_s,
                body=parse_json_object(raw, url),
            )
    except HTTPError as exc:
        raw = exc.read()
        raise CheckFailed(explain_http_error(exc, raw, url)) from exc
    except (URLError, TimeoutError, socket.timeout, OSError, ValueError) as exc:
        raise CheckFailed(explain_connection_error(exc, url, timeout)) from exc


def extract_model_ids(payload: dict[str, Any]) -> list[str]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []

    model_ids: list[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            model_ids.append(item["id"])
    return model_ids


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def extract_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""

    message = first_choice.get("message")
    if isinstance(message, dict):
        text = content_to_text(message.get("content"))
        if text:
            return text

    text = first_choice.get("text")
    return text if isinstance(text, str) else ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check that a local OpenAI-compatible LLM server can generate.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL. Default: %(default)s",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CHECK_MODEL") or os.environ.get("MODEL") or DEFAULT_CHECK_MODEL,
        help="Model id to send to /chat/completions. Default: %(default)s",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "local",
        help="Bearer token to send. Default: LLM_API_KEY, OPENAI_API_KEY, then 'local'.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("LLM_CHECK_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))),
        help="Per-request timeout in seconds. Default: %(default)g",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8,
        help="Maximum generated tokens for the chat smoke test. Default: %(default)s",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"User message for the chat smoke test. Default: {DEFAULT_PROMPT!r}",
    )
    parser.add_argument(
        "--print-payload",
        action="store_true",
        help="Print the chat request JSON before sending it.",
    )
    parser.add_argument(
        "--show-response",
        action="store_true",
        help="Print the full parsed chat response JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_url = normalize_base_url(args.base_url)

    try:
        models_url = endpoint(base_url, "models")
        print(f"Checking model list: GET {models_url}", flush=True)
        models_response = request_json(
            models_url,
            method="GET",
            timeout=args.timeout,
            api_key=args.api_key,
        )
        model_ids = extract_model_ids(models_response.body)
        if not model_ids:
            raise CheckFailed("/models returned JSON, but no model ids were found in data[].id.")

        print(
            f"OK /models: HTTP {models_response.status} in "
            f"{models_response.elapsed_s:.2f}s"
        )
        print("Models:")
        for model_id in model_ids:
            print(f"  - {model_id}")

        requested_model = (args.model or "").strip()
        model = requested_model or model_ids[0]
        if requested_model and requested_model not in model_ids:
            print(
                f"Warning: requested model {requested_model!r} was not listed by /models; "
                "trying it anyway.",
                file=sys.stderr,
            )

        chat_payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": args.prompt}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        if args.print_payload:
            print("Chat request payload:")
            print(json.dumps(chat_payload, indent=2))

        chat_url = endpoint(base_url, "chat/completions")
        print(f"Checking generation: POST {chat_url}", flush=True)
        chat_response = request_json(
            chat_url,
            method="POST",
            timeout=args.timeout,
            api_key=args.api_key,
            payload=chat_payload,
        )

        if args.show_response:
            print("Chat response JSON:")
            print(json.dumps(chat_response.body, indent=2))

        completion_text = extract_completion_text(chat_response.body).strip()
        if not completion_text:
            raise CheckFailed("/chat/completions returned no text in choices[0].message.content.")

        print(
            f"OK /chat/completions: HTTP {chat_response.status} in "
            f"{chat_response.elapsed_s:.2f}s"
        )
        print(f"Completion: {completion_text[:300]}")
        print("PASS: server listed models and generated a chat completion.")
        return 0
    except CheckFailed as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
