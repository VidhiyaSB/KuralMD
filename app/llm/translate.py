"""Per-turn Tamil -> English translation (and English -> spoken Tamil for questions)."""

import re
from typing import Optional

from app.llm import prompts
from app.llm.client import LLMError, chat_json
from app.observability.tracing import traced_step
from app.schemas import TranslationOut

_TAMIL_RE = re.compile(r"[஀-௿]")


def has_tamil(text: str) -> bool:
    return bool(_TAMIL_RE.search(text or ""))


def to_english(encounter_id: Optional[int], text: str, question_en: str = "") -> tuple[str, str]:
    """Returns (english_text, detected_language). Pure-English input skips Gemma."""
    if not has_tamil(text):
        with traced_step(encounter_id, "translate", input_summary=text) as step:
            step.set(output=text, detail={"skipped": "input already English"})
        return text, "en"

    with traced_step(encounter_id, "translate", input_summary=text) as step:
        try:
            out, meta = chat_json(
                prompts.TRANSLATE_SYSTEM,
                prompts.TRANSLATE_USER.format(question=question_en or "-", text=text),
                TranslationOut,
                name="translate-ta-en",
                temperature=0.0,
            )
            step.model = meta.model
            step.set(output=out.english, detail={"attempts": meta.attempts, "tokens": [meta.prompt_tokens, meta.completion_tokens]})
            return out.english.strip(), "ta"
        except LLMError as exc:
            step.set(output=f"translation failed: {exc}", status="error")
            return f"[untranslated Tamil] {text}", "ta"


def to_spoken_tamil(encounter_id: Optional[int], text_en: str) -> Optional[str]:
    with traced_step(encounter_id, "translate", input_summary=text_en) as step:
        try:
            out, meta = chat_json(
                prompts.TRANSLATE_TO_TAMIL_SYSTEM, text_en, TranslationOut, name="translate-en-ta", temperature=0.2
            )
            step.model = meta.model
            step.set(output=out.english, detail={"direction": "en->ta"})
            return out.english.strip() if has_tamil(out.english) else None
        except LLMError as exc:
            step.set(output=str(exc), status="error")
            return None
