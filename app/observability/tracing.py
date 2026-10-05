"""Langfuse setup, PII masking and step helpers.

One Langfuse trace per encounter (deterministic trace id, session id = encounter id),
one span per pipeline step. Gemma calls made inside a step become nested generations
via the langfuse.openai wrapper. Tracing failures never break the app.
"""

import logging
import re
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any, Optional

from app.config import get_settings
from app.observability.events import record_event

log = logging.getLogger(__name__)

_client: Any = None
_client_failed = False
_known_names: set[str] = set()

_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def register_names(names: list[str]) -> None:
    """Patient names to redact from traces."""
    for name in names:
        for part in [name, *name.split()]:
            if len(part) >= 3:
                _known_names.add(part)


def mask_text(text: str) -> str:
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    for name in sorted(_known_names, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(name)}\b", "[NAME]", text, flags=re.IGNORECASE)
    return text


def mask(data: Any = None, **_: Any) -> Any:
    """Langfuse mask callback: recursively redact names, phones, emails."""
    try:
        if isinstance(data, str):
            return mask_text(data)
        if isinstance(data, dict):
            return {k: mask(v) for k, v in data.items()}
        if isinstance(data, list):
            return [mask(v) for v in data]
        return data
    except Exception:  # noqa: BLE001
        return "[MASKING_FAILED]"


def get_langfuse() -> Any:
    global _client, _client_failed
    if _client is not None or _client_failed:
        return _client
    s = get_settings()
    if not s.langfuse_enabled:
        _client_failed = True
        return None
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
            mask=mask,
        )
        log.info("Langfuse tracing enabled (%s)", s.langfuse_host)
    except Exception:  # noqa: BLE001
        log.exception("Langfuse init failed; continuing without tracing")
        _client_failed = True
        _client = None
    return _client


def encounter_trace_id(encounter_id: int) -> str:
    lf = get_langfuse()
    if lf is None:
        return ""
    try:
        from langfuse import Langfuse

        return Langfuse.create_trace_id(seed=f"kuralmd-encounter-{encounter_id}")
    except Exception:  # noqa: BLE001
        return ""


def trace_url(trace_id: str) -> str:
    lf = get_langfuse()
    if lf is None or not trace_id:
        return ""
    try:
        return lf.get_trace_url(trace_id=trace_id) or ""
    except Exception:  # noqa: BLE001
        return f"{get_settings().langfuse_host.rstrip('/')}/trace/{trace_id}"


_OBS_TYPES = {"agent": "agent", "guardrail": "guardrail", "extract": "chain", "stt": "tool", "tts": "tool", "vision": "chain"}


class StepHandle:
    """Collects the result of a pipeline step for Langfuse + PipelineEvent."""

    def __init__(self, step: str, model: str, input_summary: Any) -> None:
        self.step = step
        self.model = model
        self.input_summary = input_summary
        self.output_summary: Any = None
        self.detail: Any = None
        self.status = "ok"
        self.observation_id = ""
        self.metadata: dict[str, Any] = {}

    def set(self, output: Any = None, status: Optional[str] = None, detail: Any = None, **metadata: Any) -> None:
        if output is not None:
            self.output_summary = output
        if status:
            self.status = status
        if detail is not None:
            self.detail = detail
        self.metadata.update(metadata)


@contextmanager
def traced_step(
    encounter_id: Optional[int],
    step: str,
    input_summary: Any = None,
    model: str = "",
) -> Iterator[StepHandle]:
    """Span in Langfuse + PipelineEvent row. Exceptions from the body propagate
    (after being recorded as status=error); tracing errors never do."""
    handle = StepHandle(step, model, input_summary)
    lf = get_langfuse()
    stack = ExitStack()
    span = None
    if lf is not None and encounter_id is not None:
        try:
            from langfuse import propagate_attributes

            stack.enter_context(
                propagate_attributes(
                    session_id=f"encounter-{encounter_id}",
                    trace_name=f"encounter-{encounter_id}",
                    tags=["kuralmd"],
                )
            )
            span = stack.enter_context(
                lf.start_as_current_observation(
                    name=step,
                    as_type=_OBS_TYPES.get(step, "span"),
                    trace_context={"trace_id": encounter_trace_id(encounter_id)},
                    input=input_summary,
                    metadata={"model": model} if model else None,
                )
            )
            handle.observation_id = getattr(span, "id", "") or ""
        except Exception:  # noqa: BLE001
            log.warning("langfuse span start failed", exc_info=True)
            stack.close()
            stack, span = ExitStack(), None

    start = time.perf_counter()
    error: Optional[BaseException] = None
    try:
        yield handle
    except BaseException as exc:
        error = exc
        handle.status = "error"
        handle.output_summary = handle.output_summary or f"{type(exc).__name__}: {exc}"
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        if span is not None:
            try:
                level = {"ok": "DEFAULT", "warn": "WARNING", "error": "ERROR"}.get(handle.status, "DEFAULT")
                span.update(
                    output=handle.output_summary,
                    level=level,
                    status_message=str(error) if error else None,
                    metadata={**handle.metadata, "latency_ms": round(latency_ms), "model": handle.model},
                )
            except Exception:  # noqa: BLE001
                log.warning("langfuse span update failed", exc_info=True)
        try:
            stack.close()
        except Exception:  # noqa: BLE001
            log.warning("langfuse span end failed", exc_info=True)
        record_event(
            encounter_id,
            step,
            status=handle.status,
            latency_ms=latency_ms,
            model=handle.model,
            input_summary=mask(handle.input_summary) if isinstance(handle.input_summary, (str, dict, list)) else handle.input_summary,
            output_summary=handle.output_summary,
            detail=handle.detail,
            observation_id=handle.observation_id,
        )


@contextmanager
def traced_generation(name: str, model: str, input_data: Any) -> Iterator[Any]:
    """Manual Langfuse generation for LLM calls that bypass the OpenAI wrapper
    (e.g. LangExtract -> native Ollama). Yields an object with .update(...) or a no-op."""

    class _Noop:
        def update(self, **_: Any) -> None:
            pass

    lf = get_langfuse()
    if lf is None:
        yield _Noop()
        return
    stack = ExitStack()
    try:
        gen = stack.enter_context(
            lf.start_as_current_observation(name=name, as_type="generation", model=model, input=input_data)
        )
    except Exception:  # noqa: BLE001
        stack.close()
        yield _Noop()
        return
    try:
        yield gen
    except BaseException as exc:
        try:
            gen.update(level="ERROR", status_message=str(exc))
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        try:
            stack.close()
        except Exception:  # noqa: BLE001
            pass


def score(encounter_id: Optional[int], name: str, value: float, comment: str = "", trace_id: str = "") -> None:
    lf = get_langfuse()
    if lf is None:
        return
    try:
        tid = trace_id or (encounter_trace_id(encounter_id) if encounter_id is not None else "")
        if not tid:
            return
        lf.create_score(name=name, value=float(value), trace_id=tid, comment=comment or None)
    except Exception:  # noqa: BLE001
        log.warning("langfuse score failed", exc_info=True)


def flush() -> None:
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.flush()
    except Exception:  # noqa: BLE001
        pass
