"""PipelineEvent rows: the in-app timeline works even when Langfuse is down."""

import json
import logging
from typing import Any, Optional

from sqlmodel import Session

from app.db import engine
from app.models import PipelineEvent

log = logging.getLogger(__name__)


def _short(value: Any, limit: int = 300) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def record_event(
    encounter_id: Optional[int],
    step: str,
    status: str = "ok",
    latency_ms: float = 0,
    model: str = "",
    input_summary: Any = None,
    output_summary: Any = None,
    detail: Any = None,
    observation_id: str = "",
) -> None:
    """Write one PipelineEvent. Never raises."""
    try:
        with Session(engine) as session:
            session.add(
                PipelineEvent(
                    encounter_id=encounter_id,
                    step=step,
                    status=status,
                    latency_ms=int(latency_ms),
                    model=model,
                    input_summary=_short(input_summary),
                    output_summary=_short(output_summary),
                    detail_json=json.dumps(detail, ensure_ascii=False, default=str) if detail is not None else "",
                    langfuse_observation_id=observation_id or "",
                )
            )
            session.commit()
    except Exception:  # noqa: BLE001
        log.exception("failed to record pipeline event")
