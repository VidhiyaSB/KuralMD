"""Prescription / medicine-strip photo -> medicines list (Gemma vision)."""

from pathlib import Path
from typing import Optional

from app.llm import prompts
from app.llm.client import LLMError, chat_json
from app.observability.tracing import traced_step
from app.schemas import PrescriptionExtraction


def read_prescription(encounter_id: Optional[int], image_path: Path) -> Optional[PrescriptionExtraction]:
    with traced_step(encounter_id, "vision", input_summary={"image": image_path.name}) as step:
        try:
            out, meta = chat_json(
                prompts.VISION_SYSTEM,
                "Read this prescription image.",
                PrescriptionExtraction,
                name="prescription-vision",
                images=[image_path],
                temperature=0.0,
            )
            step.model = meta.model
            meds = ", ".join(m.name for m in out.medicines) or "no medicines found"
            step.set(output=meds, status="ok" if out.readable else "warn", detail=out.model_dump())
            return out
        except LLMError as exc:
            step.set(output=str(exc), status="error")
            return None
