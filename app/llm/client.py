"""Gemma client: Ollama's OpenAI-compatible endpoint, wrapped by Langfuse so every
call shows up as a generation (model, tokens, latency) inside the current step span."""

import base64
import json
import logging
import mimetypes
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.observability.tracing import get_langfuse

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Gemma unavailable or returned invalid output after retry."""


@dataclass
class LLMMeta:
    model: str = ""
    latency_ms: float = 0
    attempts: int = 0
    raw: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    errors: list[str] = field(default_factory=list)


_client: Any = None
_wrapped = False


def get_client() -> Any:
    global _client, _wrapped
    if _client is not None:
        return _client
    s = get_settings()
    kwargs = dict(base_url=s.ollama_v1, api_key=s.ollama_api_key or "ollama", timeout=s.llm_timeout_s, max_retries=1)
    if get_langfuse() is not None:
        try:
            from langfuse.openai import OpenAI as LFOpenAI

            _client = LFOpenAI(**kwargs)
            _wrapped = True
            return _client
        except Exception:  # noqa: BLE001
            log.warning("langfuse.openai wrapper unavailable; using plain OpenAI client", exc_info=True)
    from openai import OpenAI

    _client = OpenAI(**kwargs)
    return _client


def ollama_status() -> dict[str, Any]:
    """Quick health check for the UI."""
    s = get_settings()
    try:
        r = httpx.get(f"{s.ollama_url.rstrip('/')}/api/tags", headers=s.ollama_headers, timeout=5)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
        present = s.is_remote_ollama or any(n.startswith(s.model_name) for n in names)
        return {"ok": True, "models": names[:20], "model_present": present}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "models": [], "model_present": False}


_warming = False


def warm_up() -> None:
    """Load Gemma into memory and pre-fill Ollama's prompt cache with the agent system
    prompt, so the first real turn only processes the new tokens. Never raises."""
    global _warming
    if _warming:
        return
    _warming = True
    try:
        from app.llm.prompts import AGENT_FAST_SYSTEM

        s = get_settings()
        if s.is_remote_ollama:
            return  # nothing to load on this machine
        httpx.post(
            f"{s.ollama_url.rstrip('/')}/api/chat",
            json={
                "model": s.model_name,
                "messages": [{"role": "system", "content": AGENT_FAST_SYSTEM}, {"role": "user", "content": "Conversation:"}],
                "stream": False,
                "think": False,
                "keep_alive": "60m",
                "options": {"num_predict": 1},
            },
            timeout=s.llm_timeout_s * 2,
        )
        log.info("Gemma warm-up done")
    except Exception as exc:  # noqa: BLE001
        log.warning("Gemma warm-up failed: %s", exc)
    finally:
        _warming = False


def _image_part(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_json(text: str) -> Any:
    text = (text or "").strip()
    text = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def chat_json(
    system: str,
    user: str,
    schema: type[T],
    *,
    name: str,
    images: Optional[list[Path]] = None,
    temperature: float = 0.2,
    model: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
) -> tuple[T, LLMMeta]:
    """Call Gemma with JSON-schema structured output, validate with Pydantic,
    retry once with the validation error, else raise LLMError."""
    s = get_settings()
    client = get_client()
    meta = LLMMeta(model=model or s.model_name)

    user_content: Any = user
    if images:
        user_content = [{"type": "text", "text": user}, *[_image_part(p) for p in images]]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    response_formats: list[dict[str, Any]] = [
        {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": False},
        },
        {"type": "json_object"},
    ]
    rf_index = 0
    start = time.perf_counter()

    for attempt in range(1, 3):
        meta.attempts = attempt
        kwargs: dict[str, Any] = dict(
            model=meta.model,
            messages=messages,
            temperature=temperature,
            response_format=response_formats[rf_index],
            # Gemma 4 "thinks" by default (100+ s per turn on this laptop). History-taking doesn't need it.
            reasoning_effort="none",
        )
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if _wrapped:
            kwargs["name"] = name
            if metadata:
                kwargs["metadata"] = {k: str(v) for k, v in metadata.items()}
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            meta.errors.append(msg)
            if "response_format" in msg and rf_index == 0:
                rf_index = 1
                continue
            meta.latency_ms = (time.perf_counter() - start) * 1000
            raise LLMError(f"Gemma call failed ({meta.model}): {msg}") from exc

        raw = resp.choices[0].message.content or ""
        meta.raw = raw
        if getattr(resp, "usage", None):
            meta.prompt_tokens = resp.usage.prompt_tokens
            meta.completion_tokens = resp.usage.completion_tokens
        try:
            parsed = schema.model_validate(_parse_json(raw))
            meta.latency_ms = (time.perf_counter() - start) * 1000
            return parsed, meta
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            meta.errors.append(f"invalid JSON: {exc}")
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"That output was invalid: {str(exc)[:500]}. Return ONLY valid JSON matching the schema.",
                },
            ]

    meta.latency_ms = (time.perf_counter() - start) * 1000
    raise LLMError(f"Gemma returned invalid JSON twice for {name}: {meta.errors[-1] if meta.errors else ''}")
