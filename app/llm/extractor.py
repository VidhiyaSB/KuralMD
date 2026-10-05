"""Transcript -> grounded OPHistory.

Primary: Google LangExtract on Ollama (span-grounded extractions with char offsets).
Fallback: Gemma JSON output + our own rapidfuzz grounding.
Either way, every value must be backed by a quote found in the patient's English words,
otherwise it is nulled, marked 'unclear' and flagged."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from app.config import get_settings
from app.llm import prompts
from app.llm.client import LLMError, chat_json
from app.llm.guardrails import GROUNDING_THRESHOLD, grounding_score
from app.models import Patient, Turn
from app.observability.tracing import score, traced_generation, traced_step
from app.schemas import HISTORY_FIELDS, ExtractedField, OPHistory, RawExtraction

log = logging.getLogger(__name__)

FIELD_KEYS = list(HISTORY_FIELDS.keys())


@dataclass
class Candidate:
    field: str
    value: str
    quote: str
    confidence: float
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    alignment: Optional[str] = None


@dataclass
class PatientSpan:
    start: int
    end: int
    turn: Turn


def build_document(turns: list[Turn]) -> tuple[str, list[PatientSpan]]:
    """English transcript with char offsets of each patient line."""
    parts: list[str] = []
    spans: list[PatientSpan] = []
    pos = 0
    for t in turns:
        text = (t.text_english or t.text_original or "").strip()
        if not text:
            continue
        prefix = "Patient: " if t.role == "patient" else "Assistant: "
        line = prefix + text + "\n"
        if t.role == "patient":
            spans.append(PatientSpan(pos + len(prefix), pos + len(prefix) + len(text), t))
        parts.append(line)
        pos += len(line)
    return "".join(parts), spans


# ---------------------------------------------------------------- LangExtract

def _examples() -> list[Any]:
    import langextract as lx

    text = (
        "Assistant: What problem brings you here today?\n"
        "Patient: I have fever for the last 4 days and some cough.\n"
        "Assistant: Does the fever go up in the evening?\n"
        "Patient: Yes, it comes more in the evening with shivering.\n"
        "Assistant: Is the cough dry or with phlegm?\n"
        "Patient: There is yellow phlegm in the morning.\n"
        "Assistant: Do you have sugar or BP?\n"
        "Patient: I have sugar for 5 years, I take metformin tablet twice daily.\n"
        "Assistant: Any allergy to medicines?\n"
        "Patient: No allergies.\n"
        "Assistant: Does anyone in the family have similar illness?\n"
        "Patient: My mother has BP.\n"
        "Assistant: Do you smoke or drink?\n"
        "Patient: I smoke 5 beedis a day, no alcohol. Appetite is less.\n"
        "Assistant: What work do you do?\n"
        "Patient: I work in a cotton mill, lot of dust there.\n"
    )
    E = lx.data.Extraction
    return [
        lx.data.ExampleData(
            text=text,
            extractions=[
                E("chief_complaint", "fever for the last 4 days", attributes={"summary": "Fever x 4 days"}),
                E("chief_complaint", "some cough", attributes={"summary": "Cough"}),
                E("history_of_presenting_illness", "it comes more in the evening with shivering", attributes={"summary": "Evening rise of temperature with chills"}),
                E("history_of_presenting_illness", "yellow phlegm in the morning", attributes={"summary": "Productive cough, yellow sputum, worse in morning"}),
                E("past_history", "I have sugar for 5 years", attributes={"summary": "Diabetes mellitus x 5 years"}),
                E("drug_history", "metformin tablet twice daily", attributes={"summary": "Metformin twice daily"}),
                E("allergies", "No allergies", attributes={"summary": "No known drug allergies"}),
                E("family_history", "My mother has BP", attributes={"summary": "Mother - hypertension"}),
                E("personal_history", "I smoke 5 beedis a day, no alcohol", attributes={"summary": "Smoker - 5 beedis/day; no alcohol"}),
                E("personal_history", "Appetite is less", attributes={"summary": "Reduced appetite"}),
                E("occupational_history", "I work in a cotton mill", attributes={"summary": "Cotton mill worker"}),
                E("occupational_hazards", "lot of dust there", attributes={"summary": "Cotton dust exposure"}),
            ],
        )
    ]


def _run_langextract(document: str, fence_output: bool = True) -> tuple[list[Candidate], dict[str, Any]]:
    import langextract as lx

    s = get_settings()
    with traced_generation("langextract", s.model_name, {"document": document[:4000], "fence_output": fence_output}) as gen:
        result = lx.extract(
            text_or_documents=document,
            prompt_description=prompts.EXTRACT_DESCRIPTION,
            examples=_examples(),
            model_id=s.model_name,
            model_url=s.ollama_url,
            fence_output=fence_output,  # Gemma wraps JSON in ```json fences; unfenced parsing returns nothing
            use_schema_constraints=False,
            max_char_buffer=4000,
            max_workers=1,
            batch_length=1,
            extraction_passes=1,
            temperature=0.1,
            show_progress=False,
            language_model_params={"timeout": int(s.llm_timeout_s * 3), **({"api_key": s.ollama_api_key} if s.ollama_api_key else {})},
        )
        cands: list[Candidate] = []
        for ex in result.extractions or []:
            if ex.extraction_class not in HISTORY_FIELDS:
                continue
            attrs = ex.attributes or {}
            summary = attrs.get("summary") or ex.extraction_text
            if isinstance(summary, list):
                summary = "; ".join(summary)
            ci = ex.char_interval
            cands.append(
                Candidate(
                    field=ex.extraction_class,
                    value=str(summary).strip(),
                    quote=ex.extraction_text.strip(),
                    confidence=0.9 if ex.alignment_status and ex.alignment_status.name == "MATCH_EXACT" else 0.7,
                    char_start=ci.start_pos if ci else None,
                    char_end=ci.end_pos if ci else None,
                    alignment=ex.alignment_status.name if ex.alignment_status else None,
                )
            )
        gen.update(output=[c.__dict__ for c in cands])
    return cands, {"extractor": "langextract", "raw_count": len(cands), "fence_output": fence_output}


# ---------------------------------------------------------------- Gemma JSON fallback

def _run_json(document: str) -> tuple[list[Candidate], dict[str, Any]]:
    raw, meta = chat_json(
        prompts.EXTRACT_JSON_SYSTEM,
        prompts.EXTRACT_JSON_USER.format(transcript=document),
        RawExtraction,
        name="extract-json",
        temperature=0.0,
    )
    cands: list[Candidate] = []
    for key in FIELD_KEYS:
        f = getattr(raw, key)
        if f.value:
            cands.append(Candidate(field=key, value=f.value.strip(), quote=(f.source_quote or "").strip(), confidence=f.confidence or 0.6))
    return cands, {"extractor": "gemma-json", "raw_count": len(cands), "attempts": meta.attempts}


# ---------------------------------------------------------------- grounding + assembly

def _locate_turn(c: Candidate, spans: list[PatientSpan]) -> tuple[Optional[PatientSpan], float]:
    """Find the patient turn backing a quote; returns (span, grounding score 0-100)."""
    if c.char_start is not None:
        for sp in spans:
            if sp.start - 2 <= c.char_start <= sp.end + 2:
                return sp, max(grounding_score(c.quote, sp.turn.text_english or ""), 90.0 if c.alignment == "MATCH_EXACT" else 0.0)
    best: tuple[Optional[PatientSpan], float] = (None, 0.0)
    for sp in spans:
        sc = grounding_score(c.quote, sp.turn.text_english or "")
        if sc > best[1]:
            best = (sp, sc)
    return best


def _assemble(cands: list[Candidate], spans: list[PatientSpan], patient: Patient) -> tuple[OPHistory, list[str], dict[str, Any]]:
    history = OPHistory()
    history.patient_group = ExtractedField(value=patient.patient_group, source="age", status="filled", confidence=1.0, note=f"From age {patient.age}")
    flags: list[str] = []
    checks: list[dict[str, Any]] = []
    grouped: dict[str, list[tuple[Candidate, Optional[PatientSpan], float]]] = {}

    for c in cands:
        span, sc = _locate_turn(c, spans)
        ok = bool(c.quote) and span is not None and sc >= GROUNDING_THRESHOLD
        checks.append({"field": c.field, "quote": c.quote, "score": round(sc, 1), "grounded": ok})
        if ok:
            grouped.setdefault(c.field, []).append((c, span, sc))
        else:
            grouped.setdefault(f"__ungrounded__{c.field}", []).append((c, span, sc))

    for key in FIELD_KEYS:
        good = grouped.get(key, [])
        bad = grouped.get(f"__ungrounded__{key}", [])
        label = HISTORY_FIELDS[key]["label"]
        if good:
            # de-duplicate values, keep transcript order
            good.sort(key=lambda g: g[1].start if g[1] else 0)
            values, quotes_en, quotes_orig, seen = [], [], [], set()
            for c, sp, _ in good:
                if c.value.lower() not in seen:
                    seen.add(c.value.lower())
                    values.append(c.value)
                quotes_en.append(c.quote)
                orig = sp.turn.text_original if sp else c.quote
                if orig not in quotes_orig:
                    quotes_orig.append(orig)
            first_span = good[0][1]
            field = ExtractedField(
                value="; ".join(values),
                source_quote_en=" … ".join(dict.fromkeys(quotes_en)),
                source_quote_original=" … ".join(quotes_orig),
                source="voice" if first_span and first_span.turn.input_mode != "transcript" else "transcript",
                turn_id=first_span.turn.id if first_span else None,
                confidence=round(sum(c.confidence for c, _, _ in good) / len(good), 2),
                status="filled",
                grounding_score=round(min(sc for _, _, sc in good), 1),
            )
            if key == "pain_score":
                m = re.search(r"\b(10|[0-9])\b", field.value or "")
                if not m:
                    field.status, field.note = "unclear", "Pain score must be a number 0-10 said by the patient."
                    flags.append(f"{label}: not a 0-10 number - please confirm")
                else:
                    field.value = m.group(1)
            setattr(history, key, field)
        elif bad:
            c = bad[0][0]
            setattr(
                history, key,
                ExtractedField(
                    value=None, source_quote_en=c.quote or None, status="unclear", confidence=0.0,
                    grounding_score=round(bad[0][2], 1),
                    note=f"AI suggested '{c.value}' but it could not be found in the patient's words. Not filled.",
                ),
            )
            flags.append(f"{label}: AI answer not grounded in transcript - verify with patient")
        else:
            setattr(history, key, ExtractedField(status="missing"))

    if history.chief_complaint.status != "filled":
        flags.append("Chief complaint missing - please ask the patient")
    grounded = sum(1 for ch in checks if ch["grounded"])
    stats = {
        "candidates": len(checks),
        "grounded": grounded,
        "grounding_pass_rate": round(grounded / len(checks), 3) if checks else 1.0,
        "filled": sum(1 for k in FIELD_KEYS if getattr(history, k).status == "filled"),
        "unclear": sum(1 for k in FIELD_KEYS if getattr(history, k).status == "unclear"),
        "missing": sum(1 for k in FIELD_KEYS if getattr(history, k).status == "missing"),
        "checks": checks,
    }
    return history, flags, stats


def extract_from_document(document: str, spans: list[PatientSpan], patient: Patient, encounter_id: Optional[int]) -> tuple[OPHistory, list[str], dict[str, Any]]:
    s = get_settings()
    with traced_step(encounter_id, "extract", input_summary=document, model=s.model_name) as step:
        cands: list[Candidate] = []
        info: dict[str, Any] = {}
        used = s.extractor
        if s.extractor == "langextract":
            for fence in (True, False):
                try:
                    cands, info = _run_langextract(document, fence_output=fence)
                except Exception as exc:  # noqa: BLE001
                    log.warning("LangExtract (fence=%s) failed: %s", fence, exc)
                    info = {"langextract_error": str(exc)[:300]}
                if cands:
                    break
            if not cands:
                # Zero spans from a real conversation means a parse problem, not an empty history.
                log.warning("LangExtract found nothing; falling back to Gemma JSON extraction")
                info = {**info, "langextract_error": info.get("langextract_error", "0 spans parsed")}
                used = "json"
                step.set(status="warn")
        if used != "langextract":
            try:
                cands, more = _run_json(document)
                info.update(more)
            except LLMError as exc:
                step.set(output=f"extraction failed: {exc}", status="error")
                raise
        step.set(output=f"{len(cands)} candidate spans via {info.get('extractor')}", detail={**info, "candidates": [c.__dict__ for c in cands]})

    with traced_step(encounter_id, "guardrail", input_summary={"check": "grounding", "candidates": len(cands)}) as g:
        history, flags, stats = _assemble(cands, spans, patient)
        stats["extractor"] = info.get("extractor")
        g.set(
            output=f"grounding pass rate {stats['grounding_pass_rate']:.0%} ({stats['grounded']}/{stats['candidates']}); "
            f"filled {stats['filled']}, unclear {stats['unclear']}, missing {stats['missing']}",
            status="warn" if stats["unclear"] else "ok",
            detail=stats,
        )
    score(encounter_id, "grounding_pass_rate", stats["grounding_pass_rate"], f"{stats['grounded']}/{stats['candidates']} grounded")
    score(encounter_id, "fields_filled", stats["filled"])
    return history, flags, stats


def extract_history(encounter_id: Optional[int], patient: Patient, turns: list[Turn]) -> tuple[OPHistory, list[str], dict[str, Any]]:
    document, spans = build_document(turns)
    return extract_from_document(document, spans, patient, encounter_id)
