"""Interview orchestration: STT -> translate -> agent -> TTS, then extraction."""

import json
import logging
import re
from datetime import datetime
from typing import Any, Optional

from sqlmodel import Session, select

from app.config import RECORDINGS_DIR
from app.db import engine
from app.llm import intake_agent
from app.llm.extractor import extract_history
from app.llm.translate import has_tamil, to_english
from app.models import Attachment, Encounter, Patient, Turn
from app.observability.tracing import encounter_trace_id, flush, score, traced_step
from app.schemas import HISTORY_FIELDS, AgentTurn
from app.voice.stt import STTError, get_stt
from app.voice.tts import TTSError, get_tts

log = logging.getLogger(__name__)


STABLE_FIELDS = [
    "past_history", "drug_history", "allergies", "family_history",
    "personal_history", "occupational_history", "occupational_hazards",
]


def prior_history(session: Session, encounter: Encounter) -> Optional[dict[str, Any]]:
    """Stable history from the patient's latest doctor-approved visit (for returning patients)."""
    prev = session.exec(
        select(Encounter)
        .where(Encounter.patient_id == encounter.patient_id, Encounter.id != encounter.id, Encounter.status == "approved")
        .order_by(Encounter.approved_at.desc(), Encounter.id.desc())
    ).first()
    if prev is None:
        return None
    hist = prev.get_extracted()
    fields = {
        k: (hist.get(k) or {}).get("value")
        for k in STABLE_FIELDS
        if (hist.get(k) or {}).get("status") == "filled" and (hist.get(k) or {}).get("value")
    }
    if not fields:
        return None
    return {"encounter_id": prev.id, "date": prev.created_at.strftime("%d %b %Y"), "fields": fields}


def get_turns(session: Session, encounter_id: int) -> list[Turn]:
    return list(session.exec(select(Turn).where(Turn.encounter_id == encounter_id).order_by(Turn.id)).all())


def speak(encounter_id: Optional[int], text: str, language: str) -> Optional[str]:
    """TTS with cache. Returns audio url or None (browser speechSynthesis fallback)."""
    tts = get_tts()
    if tts is None:
        return None
    with traced_step(encounter_id, "tts", input_summary=text) as step:
        try:
            url, cached = tts.synthesize(text, language)
            provider = tts.used.name if tts.used else ""
            step.model = f"{provider}:{tts.model}"
            step.set(
                output=url,
                status="warn" if tts.errors else "ok",
                detail={"provider": provider, "cache_hit": cached, "language": language, "fallback_from": tts.errors or None},
                cache_hit=cached,
            )
            return url
        except TTSError as exc:
            step.set(output=str(exc), status="error")
            return None


def _turn_payload(t: Turn) -> dict[str, Any]:
    return {
        "id": t.id,
        "role": t.role,
        "text_original": t.text_original,
        "text_english": t.text_english,
        "language": t.language,
        "audio_url": t.audio_path or None,
        "stt_confidence": t.stt_confidence,
        "input_mode": t.input_mode,
    }


def progress(encounter: Encounter) -> dict[str, Any]:
    covered = encounter.get_agent_state().get("fields_covered", [])
    total = len(HISTORY_FIELDS) - 1  # pain_score is conditional
    done = len([f for f in covered if f != "pain_score"])
    return {
        "covered": covered,
        "fields": [{"key": k, "label": v["label"], "ta": v["ta"], "covered": k in covered} for k, v in HISTORY_FIELDS.items()],
        "percent": min(100, round(100 * done / total)),
    }


def _save_agent_turn(session: Session, encounter: Encounter, agent: AgentTurn) -> Turn:
    # Commit pending changes first: otherwise reading encounter.language autoflushes an UPDATE and
    # holds the SQLite write lock through the slow TTS call, blocking pipeline-event writes (30 s waits).
    session.add(encounter)
    session.commit()
    eid, language = encounter.id, encounter.language
    audio = speak(eid, agent.next_question_patient_lang, language)
    turn = Turn(
        encounter_id=encounter.id,
        role="agent",
        text_original=agent.next_question_patient_lang,
        language=encounter.language if has_tamil(agent.next_question_patient_lang) else "en",
        text_english=agent.next_question_en,
        audio_path=audio or "",
        input_mode="tts",
    )
    session.add(turn)
    session.commit()
    session.refresh(turn)
    return turn


def state_payload(session: Session, encounter: Encounter) -> dict[str, Any]:
    turns = get_turns(session, encounter.id)
    agent_state = encounter.get_agent_state()
    return {
        "encounter_id": encounter.id,
        "status": encounter.status,
        "language": encounter.language,
        "urgent": encounter.urgent,
        "red_flag_reason": encounter.red_flag_reason,
        "done": encounter.status != "interview_in_progress",
        "questions_asked": encounter.questions_asked,
        "turns": [_turn_payload(t) for t in turns],
        "progress": progress(encounter),
        "fallback": agent_state.get("fallback", False),
        "prior": {
            "date": agent_state["prior"]["date"],
            "fields": [HISTORY_FIELDS[k]["label"] for k in agent_state["prior"]["fields"]],
        } if agent_state.get("prior") else None,
    }


def start_interview(session: Session, encounter: Encounter) -> dict[str, Any]:
    if not encounter.langfuse_trace_id:
        encounter.langfuse_trace_id = encounter_trace_id(encounter.id)
    # Commit before the slow TTS call so this request never holds a SQLite write lock
    # while pipeline events are being written from other connections.
    session.add(encounter)
    session.commit()
    turns = get_turns(session, encounter.id)
    if not turns:
        agent = intake_agent.first_question(encounter.language)
        _save_agent_turn(session, encounter, agent)
        encounter.questions_asked = 1
        encounter.agent_state_json = json.dumps({"fields_covered": [], "prior": prior_history(session, encounter)}, ensure_ascii=False)
    encounter.updated_at = datetime.now()
    session.add(encounter)
    session.commit()
    session.refresh(encounter)
    # Langfuse exports in a background thread; no blocking flush in the request path.
    return state_payload(session, encounter)


def set_language(session: Session, encounter: Encounter, language: str) -> dict[str, Any]:
    """Switch language; re-ask the last question in the new language."""
    encounter.language = language
    session.add(encounter)
    session.commit()
    turns = get_turns(session, encounter.id)
    last_agent = next((t for t in reversed(turns) if t.role == "agent"), None)
    if last_agent and encounter.status == "interview_in_progress":
        from app.llm.translate import to_spoken_tamil

        if language == "ta" and not has_tamil(last_agent.text_original):
            ta = to_spoken_tamil(encounter.id, last_agent.text_english)
            if ta:
                agent = AgentTurn(next_question_en=last_agent.text_english, next_question_patient_lang=ta)
                _save_agent_turn(session, encounter, agent)
        elif language == "en" and has_tamil(last_agent.text_original):
            agent = AgentTurn(next_question_en=last_agent.text_english, next_question_patient_lang=last_agent.text_english)
            _save_agent_turn(session, encounter, agent)
    session.refresh(encounter)
    return state_payload(session, encounter)


def transcribe_audio(encounter: Encounter, audio: bytes, filename: str, language: Optional[str]) -> dict[str, Any]:
    """Run STT only. Returns {"ok", "text", "language", "confidence", "audio_path"|"error"}."""
    n = len(list(RECORDINGS_DIR.glob(f"enc{encounter.id}_*"))) + 1
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "webm")[:5]
    path = RECORDINGS_DIR / f"enc{encounter.id}_{n:02d}.{ext}"
    path.write_bytes(audio)
    rel = f"/static/audio/recordings/{path.name}"

    stt = get_stt()
    if stt is None:
        return {"ok": False, "error": "Speech-to-text is turned off (STT_PROVIDER=none). Please type the answer."}
    with traced_step(encounter.id, "stt", input_summary={"audio": path.name, "bytes": len(audio), "language_hint": language}, model=stt.name) as step:
        try:
            result = stt.transcribe(audio, path.name, language)
            step.model = result.model or stt.name
            status = "ok"
            if not result.text:
                status = "warn"
            elif result.confidence is not None and result.confidence < intake_agent.LOW_STT_CONFIDENCE:
                status = "warn"
            step.set(
                output=result.text or "(no speech detected)",
                status=status,
                detail={"language": result.language, "confidence": result.confidence, "provider": result.provider},
            )
        except STTError as exc:
            step.set(output=str(exc), status="error")
            return {"ok": False, "error": f"Could not transcribe: {exc}. Please type the answer.", "audio_path": rel}
    return {
        "ok": True,
        "text": result.text,
        "language": result.language if result.language in ("ta", "en") else (language or "ta"),
        "confidence": result.confidence,
        "audio_path": rel,
    }


def log_browser_stt(encounter_id: int, text: str, language: str, confidence: Optional[float], reason: str) -> None:
    """Record a pipeline step for STT done in the browser (Chrome Web Speech API, Google)."""
    with traced_step(encounter_id, "stt", input_summary={"source": "browser", "language": language}, model="google-web-speech") as step:
        step.set(
            output=text,
            status="warn" if reason else "ok",
            detail={"provider": "browser (Google Web Speech)", "confidence": confidence, "fallback_from": reason or None},
        )


def handle_answer(
    session: Session,
    encounter: Encounter,
    text: str,
    language: str,
    *,
    audio_path: str = "",
    stt_confidence: Optional[float] = None,
    input_mode: str = "text",
) -> dict[str, Any]:
    """Patient answered (already transcribed or typed). Translate, decide next question, speak."""
    if encounter.status != "interview_in_progress":
        return state_payload(session, encounter)
    patient = session.get(Patient, encounter.patient_id)
    turns = get_turns(session, encounter.id)
    tamil = has_tamil(text)
    p_turn = Turn(
        encounter_id=encounter.id,
        role="patient",
        text_original=text,
        language="ta" if tamil else "en",
        text_english="" if tamil else text,  # Tamil is translated by the same Gemma call as the next question
        audio_path=audio_path,
        stt_confidence=stt_confidence,
        input_mode=input_mode,
    )
    session.add(p_turn)
    session.commit()
    turns.append(p_turn)

    agent, answer_en, new_state = intake_agent.next_turn(encounter, patient, turns, p_turn)
    if tamil and answer_en:
        p_turn.text_english = answer_en
        session.add(p_turn)
        session.commit()
    encounter.agent_state_json = json.dumps({**new_state, "last_agent": agent.model_dump()}, ensure_ascii=False)
    _save_agent_turn(session, encounter, agent)
    encounter.questions_asked += 0 if agent.done else 1
    encounter.updated_at = datetime.now()

    if agent.red_flag:
        encounter.urgent = True
        encounter.red_flag_reason = agent.red_flag_reason
        flags = encounter.get_flags()
        flags.append(f"RED FLAG: {agent.red_flag_reason}")
        encounter.flags = json.dumps(flags, ensure_ascii=False)
        score(encounter.id, "red_flag", 1, agent.red_flag_reason)
    if agent.done:
        encounter.status = "interview_complete"
        encounter.completed_at = datetime.now()
    session.add(encounter)
    session.commit()
    session.refresh(encounter)
    # (no blocking Langfuse flush here - it batches in the background)
    payload = state_payload(session, encounter)
    payload["red_flag"] = agent.red_flag
    return payload


def finish_interview(session: Session, encounter: Encounter) -> None:
    """Mark complete (extraction runs in background via run_extraction)."""
    if encounter.status == "interview_in_progress":
        encounter.status = "interview_complete"
        encounter.completed_at = datetime.now()
        session.add(encounter)
        session.commit()


def run_extraction(encounter_id: int) -> None:
    """Background job: transcript -> grounded OPHistory -> encounter. Never raises."""
    try:
        with Session(engine) as session:
            encounter = session.get(Encounter, encounter_id)
            if encounter is None:
                return
            patient = session.get(Patient, encounter.patient_id)
            turns = get_turns(session, encounter_id)
            backfill_translations(session, turns)
            attachments = list(session.exec(select(Attachment).where(Attachment.encounter_id == encounter_id)).all())
            history, flags, stats = extract_history(encounter_id, patient, turns)
            merge_attachments(history, attachments)
            carry_over_prior(history, encounter.get_agent_state())

            existing = [f for f in encounter.get_flags() if f.startswith("RED FLAG")]
            encounter.extracted_json = history.model_dump_json()
            encounter.flags = json.dumps(existing + flags, ensure_ascii=False)
            encounter.status = "needs_clarification" if flags else "doctor_review"
            encounter.updated_at = datetime.now()
            encounter.agent_state_json = json.dumps(
                {**encounter.get_agent_state(), "extraction_stats": stats}, ensure_ascii=False
            )
            session.add(encounter)
            session.commit()
    except Exception:  # noqa: BLE001
        log.exception("extraction failed for encounter %s", encounter_id)
        with Session(engine) as session:
            encounter = session.get(Encounter, encounter_id)
            if encounter:
                flags = encounter.get_flags()
                flags.append("Extraction failed - please fill the history manually or retry.")
                encounter.flags = json.dumps(flags)
                encounter.status = "needs_clarification"
                session.add(encounter)
                session.commit()
    finally:
        flush()


def carry_over_prior(history: Any, state: dict[str, Any]) -> None:
    """Fill stable fields not discussed today from the last approved visit.
    Confirmed by the patient ('no change') -> filled; otherwise -> unclear for the doctor to verify."""
    from app.schemas import ExtractedField

    prior = state.get("prior")
    if not prior:
        return
    covered = set(state.get("fields_covered", []))
    for key, value in prior["fields"].items():
        current = getattr(history, key)
        if current.status == "filled":
            continue
        confirmed = key in covered
        setattr(history, key, ExtractedField(
            value=value,
            source=f"previous_visit:OP #{prior['encounter_id']}",
            status="filled" if confirmed else "unclear",
            confidence=0.8 if confirmed else 0.5,
            note=(f"From previous approved visit ({prior['date']}). "
                  + ("Patient confirmed no change." if confirmed else "Not re-confirmed today - please verify.")),
        ))


def backfill_translations(session: Session, turns: list[Turn]) -> None:
    """Translate patient turns whose translation was skipped (e.g. Gemma timed out mid-interview)."""
    last_q = ""
    for t in turns:
        if t.role == "agent":
            last_q = t.text_english
        elif has_tamil(t.text_original) and (not t.text_english or t.text_english.startswith("[untranslated")):
            english, _ = to_english(t.encounter_id, t.text_original, last_q)
            if not english.startswith("[untranslated"):
                t.text_english = english
                session.add(t)
                session.commit()


def merge_attachments(history: Any, attachments: list[Attachment]) -> None:
    """Add medicines read from uploaded prescriptions into drug_history."""
    from app.schemas import ExtractedField

    meds: list[str] = []
    sources: list[str] = []
    for a in attachments:
        if not a.extracted_json:
            continue
        data = json.loads(a.extracted_json)
        for m in data.get("medicines", []):
            parts = [m.get("name"), m.get("dose"), m.get("frequency"), m.get("duration")]
            meds.append(" ".join(p for p in parts if p))
        sources.append(a.filename)
    if not meds:
        return
    upload_text = "; ".join(meds)
    current = history.drug_history
    if current.status == "filled" and current.value:
        current.value = f"{current.value}. From prescription: {upload_text}"
        current.note = f"Also from upload: {', '.join(sources)}"
    else:
        history.drug_history = ExtractedField(
            value=upload_text,
            source_quote_en=upload_text,
            source_quote_original=upload_text,
            source=f"upload:{', '.join(sources)}",
            confidence=0.7,
            status="filled",
            note="Read from uploaded prescription by Gemma vision - verify against the image.",
        )


_LINE_RE = re.compile(r"^\s*(doctor|dr|assistant|agent|q|patient|pt|a)\s*[:\-]\s*(.+)$", re.I)


def import_transcript(session: Session, encounter: Encounter, transcript: str, language: str) -> int:
    """Paste-a-transcript path: 'Doctor: ...' / 'Patient: ...' lines become turns."""
    count = 0
    last_q = ""
    for raw in transcript.splitlines():
        m = _LINE_RE.match(raw)
        if not m:
            continue
        who, text = m.group(1).lower(), m.group(2).strip()
        role = "patient" if who in ("patient", "pt", "a") else "agent"
        if role == "patient":
            english, _ = to_english(encounter.id, text, last_q)
        else:
            english = text if not has_tamil(text) else to_english(encounter.id, text)[0]
            last_q = english
        session.add(
            Turn(
                encounter_id=encounter.id, role=role, text_original=text,
                language="ta" if has_tamil(text) else "en", text_english=english, input_mode="transcript",
            )
        )
        count += 1
    encounter.status = "interview_complete"
    encounter.language = language
    encounter.completed_at = datetime.now()
    if not encounter.langfuse_trace_id:
        encounter.langfuse_trace_id = encounter_trace_id(encounter.id)
    session.add(encounter)
    session.commit()
    return count
