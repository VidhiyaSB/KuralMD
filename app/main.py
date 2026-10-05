import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, func, select

from app.config import BASE_DIR, STATIC_DIR, UPLOADS_DIR, get_settings
from app.db import get_session, init_db
from app.llm.client import ollama_status, warm_up
from app.models import Attachment, Encounter, EvalRun, Patient, PipelineEvent, Turn
from app.observability.tracing import get_langfuse, register_names, trace_url
from app.schemas import DOCTOR_ONLY_FIELDS, HISTORY_FIELDS, ExtractedField, OPHistory, TextAnswerIn, TranscriptIn
from app.seed import seed
from app.services import interview as svc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("kuralmd")

app = FastAPI(title="KuralMD")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

STATUS_META: dict[str, tuple[str, str]] = {
    "interview_in_progress": ("Interview in progress", "bg-sky-100 text-sky-800"),
    "interview_complete": ("Extracting record", "bg-indigo-100 text-indigo-800"),
    "needs_clarification": ("Needs clarification", "bg-amber-100 text-amber-800"),
    "doctor_review": ("Doctor review", "bg-teal-100 text-teal-800"),
    "approved": ("Approved", "bg-emerald-100 text-emerald-800"),
}
STEP_META: dict[str, tuple[str, str]] = {
    "stt": ("Speech to text", "🎙️"),
    "translate": ("Translation", "🌐"),
    "agent": ("Intake agent", "🧠"),
    "extract": ("Extraction", "📋"),
    "guardrail": ("Guardrail", "🛡️"),
    "tts": ("Text to speech", "🔊"),
    "vision": ("Prescription vision", "📷"),
}
REVIEW_STATUSES = ["interview_complete", "needs_clarification", "doctor_review"]

templates.env.globals.update(
    STATUS_META=STATUS_META,
    STEP_META=STEP_META,
    HISTORY_FIELDS=HISTORY_FIELDS,
    DOCTOR_ONLY_FIELDS=DOCTOR_ONLY_FIELDS,
)


def _fmt_dt(value: Optional[datetime]) -> str:
    if not value:
        return ""
    today = datetime.now().date()
    if value.date() == today:
        return "Today " + value.strftime("%I:%M %p").lstrip("0")
    return value.strftime("%d %b %Y")


templates.env.filters["dt"] = _fmt_dt
templates.env.filters["fromjson"] = lambda s: json.loads(s) if s else None


def _model_label(model: str) -> str:
    """Friendly model name for the UI ('gemma4:...' -> 'Gemma 4'); other providers unchanged."""
    return "Gemma 4" if (model or "").lower().startswith("gemma") else model


templates.env.filters["model_label"] = _model_label


@app.on_event("startup")
def _startup() -> None:
    init_db()
    seed()
    from sqlmodel import Session as S

    from app.db import engine

    with S(engine) as session:
        register_names([p.name for p in session.exec(select(Patient)).all()])
    get_langfuse()
    import threading

    threading.Thread(target=warm_up, daemon=True).start()


def page(request: Request, session: Session, name: str, **ctx: Any) -> HTMLResponse:
    review_count = session.exec(select(func.count(Encounter.id)).where(Encounter.status.in_(REVIEW_STATUSES))).one()
    urgent_count = session.exec(
        select(func.count(Encounter.id)).where(Encounter.urgent == True, Encounter.status != "approved")  # noqa: E712
    ).one()
    return templates.TemplateResponse(
        request,
        name,
        {"review_count": review_count, "urgent_count": urgent_count, "settings": get_settings(), **ctx},
    )


def get_encounter_or_404(session: Session, encounter_id: int) -> Encounter:
    e = session.get(Encounter, encounter_id)
    if e is None:
        raise HTTPException(404, "Encounter not found")
    return e


# ================================================================ pages

@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse("/patients")


@app.get("/patients", response_class=HTMLResponse)
def patients_page(request: Request, tab: str = "clinic", q: str = "", session: Session = Depends(get_session)) -> HTMLResponse:
    patients = list(session.exec(select(Patient).order_by(Patient.pid)).all())
    latest: dict[int, Encounter] = {}
    for e in session.exec(select(Encounter).order_by(Encounter.created_at)).all():
        latest[e.patient_id] = e
    if q:
        ql = q.lower()
        patients = [p for p in patients if ql in p.name.lower() or ql in p.pid.lower()]
    today = datetime.now().date()
    if tab == "today":
        patients = [p for p in patients if p.id in latest and latest[p.id].created_at.date() == today]
    elif tab == "history":
        patients = [p for p in patients if p.id in latest and latest[p.id].status == "approved"]
    return page(request, session, "patients.html", patients=patients, latest=latest, tab=tab, q=q, nav="patients")


@app.get("/patients/new", response_class=HTMLResponse)
def new_patient_page(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    return page(request, session, "patient_new.html", nav="patients")


def _new_patient(session: Session, **fields: Any) -> Patient:
    count = session.exec(select(func.count(Patient.id))).one()
    p = Patient(pid=f"KMD-{count + 1:04d}", **fields)
    session.add(p)
    session.commit()
    session.refresh(p)
    register_names([p.name])
    return p


@app.post("/patients")
def create_patient(
    name: str = Form(...), age: int = Form(...), sex: str = Form(...), blood_group: str = Form(""),
    preferred_language: str = Form("ta"), assigned_student: str = Form(""), known_allergies: str = Form(""),
    session: Session = Depends(get_session),
) -> RedirectResponse:
    p = _new_patient(
        session, name=name.strip(), age=age, sex=sex, blood_group=blood_group,
        preferred_language=preferred_language, assigned_student=assigned_student, known_allergies=known_allergies,
    )
    return RedirectResponse(f"/patients/{p.id}", status_code=303)


@app.post("/walkin")
def quick_walkin(
    name: str = Form(...), age: int = Form(...), sex: str = Form(...), preferred_language: str = Form("ta"),
    session: Session = Depends(get_session),
) -> RedirectResponse:
    """Register a walk-in patient and jump straight into the voice interview."""
    if preferred_language not in ("ta", "en"):
        preferred_language = "ta"
    p = _new_patient(
        session, name=name.strip(), age=age, sex=sex, preferred_language=preferred_language, assigned_student="Divya R",
    )
    e = Encounter(patient_id=p.id, language=p.preferred_language)
    session.add(e)
    session.commit()
    session.refresh(e)
    return RedirectResponse(f"/interview/{e.id}", status_code=303)


@app.get("/patients/{patient_id}", response_class=HTMLResponse)
def patient_detail(request: Request, patient_id: int, session: Session = Depends(get_session)) -> HTMLResponse:
    p = session.get(Patient, patient_id)
    if p is None:
        raise HTTPException(404)
    encounters = list(session.exec(select(Encounter).where(Encounter.patient_id == p.id).order_by(Encounter.created_at.desc())).all())
    alerts: list[dict[str, str]] = []
    if p.known_allergies:
        alerts.append({"kind": "allergy", "text": f"Allergy: {p.known_allergies}"})
    for e in encounters:
        hist = e.get_extracted()
        al = (hist.get("allergies") or {}).get("value")
        if al and "no " not in al.lower() and al not in p.known_allergies:
            alerts.append({"kind": "allergy", "text": f"Allergy (reported {_fmt_dt(e.created_at)}): {al}"})
        if e.urgent and e.status != "approved":
            alerts.append({"kind": "red", "text": f"Red flag: {e.red_flag_reason}"})
    return page(request, session, "patient_detail.html", patient=p, encounters=encounters, alerts=alerts, nav="patients")


@app.post("/patients/{patient_id}/encounters")
def create_encounter(patient_id: int, department: str = Form("Community Medicine"), session: Session = Depends(get_session)) -> RedirectResponse:
    p = session.get(Patient, patient_id)
    if p is None:
        raise HTTPException(404)
    e = Encounter(patient_id=p.id, department=department, language=p.preferred_language)
    session.add(e)
    session.commit()
    session.refresh(e)
    return RedirectResponse(f"/interview/{e.id}", status_code=303)


@app.get("/interview/{encounter_id}", response_class=HTMLResponse)
def interview_page(request: Request, encounter_id: int, session: Session = Depends(get_session)) -> HTMLResponse:
    e = get_encounter_or_404(session, encounter_id)
    p = session.get(Patient, e.patient_id)
    return page(request, session, "interview.html", encounter=e, patient=p, nav="interviews")


@app.get("/interviews", response_class=HTMLResponse)
def interviews_page(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    rows = session.exec(select(Encounter, Patient).join(Patient).order_by(Encounter.created_at.desc()).limit(50)).all()
    return page(request, session, "interviews.html", rows=rows, nav="interviews")


@app.get("/review", response_class=HTMLResponse)
def review_queue(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    rows = session.exec(
        select(Encounter, Patient).join(Patient).where(Encounter.status.in_(REVIEW_STATUSES))
        .order_by(Encounter.urgent.desc(), Encounter.completed_at.desc())
    ).all()
    items = []
    for e, p in rows:
        hist = e.get_extracted()
        unclear = sum(1 for k in HISTORY_FIELDS if (hist.get(k) or {}).get("status") == "unclear")
        missing = sum(1 for k in HISTORY_FIELDS if (hist.get(k) or {}).get("status") == "missing")
        filled = sum(1 for k in HISTORY_FIELDS if (hist.get(k) or {}).get("status") == "filled")
        items.append({"e": e, "p": p, "unclear": unclear, "missing": missing, "filled": filled})
    return page(request, session, "review.html", items=items, nav="review")


@app.get("/encounters/{encounter_id}/record", response_class=HTMLResponse)
def record_review(request: Request, encounter_id: int, session: Session = Depends(get_session)) -> HTMLResponse:
    e = get_encounter_or_404(session, encounter_id)
    p = session.get(Patient, e.patient_id)
    hist = e.get_extracted() or OPHistory().model_dump()
    turns = {t.id: t for t in svc.get_turns(session, e.id)}
    attachments = list(session.exec(select(Attachment).where(Attachment.encounter_id == e.id)).all())
    return page(
        request, session, "record_review.html",
        encounter=e, patient=p, hist=hist, doctor=e.get_doctor_fields(), flags=e.get_flags(),
        turns=turns, attachments=attachments, langfuse_url=trace_url(e.langfuse_trace_id), nav="review",
    )


@app.post("/encounters/{encounter_id}/record")
async def save_record(request: Request, encounter_id: int, session: Session = Depends(get_session)) -> RedirectResponse:
    e = get_encounter_or_404(session, encounter_id)
    form = await request.form()
    action = str(form.get("action", "save"))
    hist = e.get_extracted() or OPHistory().model_dump()
    for key in ["patient_group", *HISTORY_FIELDS.keys()]:
        new_val = str(form.get(f"h_{key}", "")).strip()
        cur = hist.get(key) or ExtractedField().model_dump()
        if new_val != (cur.get("value") or ""):
            cur["value"] = new_val or None
            cur["status"] = "filled" if new_val else "missing"
            cur["source"] = "doctor"
            cur["note"] = "Edited by doctor"
        hist[key] = cur
    doctor = {k: str(form.get(f"d_{k}", "")).strip() for group in DOCTOR_ONLY_FIELDS.values() for k, _ in group}
    e.extracted_json = json.dumps(hist, ensure_ascii=False)
    e.doctor_fields_json = json.dumps(doctor, ensure_ascii=False)
    e.doctor_notes = str(form.get("doctor_notes", ""))
    e.updated_at = datetime.now()
    if action == "approve":
        e.status = "approved"
        e.approved_by = str(form.get("approved_by") or "Doctor")
        e.approved_at = datetime.now()
    elif e.status in ("needs_clarification", "interview_complete"):
        e.status = "doctor_review"
    session.add(e)
    session.commit()
    return RedirectResponse(f"/encounters/{e.id}/record?saved=1", status_code=303)


@app.post("/encounters/{encounter_id}/followup")
async def ask_followup(request: Request, encounter_id: int, session: Session = Depends(get_session)) -> RedirectResponse:
    e = get_encounter_or_404(session, encounter_id)
    form = await request.form()
    fields = [f for f in form.getlist("fields") if f in HISTORY_FIELDS]
    note = str(form.get("note", ""))
    state = e.get_agent_state()
    state.update({"focus_fields": fields, "focus_note": note, "rephrased_last": False})
    e.agent_state_json = json.dumps(state, ensure_ascii=False)
    e.status = "interview_in_progress"
    e.questions_asked = max(0, get_settings().max_questions - max(3, 2 * len(fields) + 1))
    session.add(e)
    session.commit()
    # Ask the first clarification question right away.
    p = session.get(Patient, e.patient_id)
    from app.llm import intake_agent

    agent, _, new_state = intake_agent.next_turn(e, p, svc.get_turns(session, e.id))
    if not agent.done:
        e.agent_state_json = json.dumps(new_state, ensure_ascii=False)
        svc._save_agent_turn(session, e, agent)
        e.questions_asked += 1
        session.add(e)
        session.commit()
    return RedirectResponse(f"/interview/{e.id}", status_code=303)


@app.post("/encounters/{encounter_id}/transcript")
def paste_transcript(
    encounter_id: int, background: BackgroundTasks, transcript: str = Form(...), language: str = Form("en"),
    session: Session = Depends(get_session),
) -> RedirectResponse:
    e = get_encounter_or_404(session, encounter_id)
    n = svc.import_transcript(session, e, transcript, language)
    if n == 0:
        raise HTTPException(400, "No 'Doctor:' / 'Patient:' lines found in transcript")
    background.add_task(svc.run_extraction, e.id)
    return RedirectResponse(f"/encounters/{e.id}/record", status_code=303)


@app.post("/encounters/{encounter_id}/upload")
async def upload_prescription(encounter_id: int, file: UploadFile = File(...), session: Session = Depends(get_session)) -> RedirectResponse:
    e = get_encounter_or_404(session, encounter_id)
    ext = (file.filename or "upload.png").rsplit(".", 1)[-1].lower()
    if ext not in ("png", "jpg", "jpeg", "webp"):
        raise HTTPException(400, "Please upload a PNG/JPG image")
    name = f"enc{e.id}_{uuid.uuid4().hex[:8]}.{ext}"
    path = UPLOADS_DIR / name
    path.write_bytes(await file.read())
    from app.llm.vision import read_prescription

    result = read_prescription(e.id, path)
    att = Attachment(encounter_id=e.id, filename=file.filename or name, path=f"/static/uploads/{name}",
                     extracted_json=result.model_dump_json() if result else "")
    session.add(att)
    session.commit()
    if e.extracted_json and result:
        hist = OPHistory.model_validate(e.get_extracted())
        svc.merge_attachments(hist, [att])
        e.extracted_json = hist.model_dump_json()
        session.add(e)
        session.commit()
    return RedirectResponse(f"/encounters/{e.id}/record", status_code=303)


@app.post("/encounters/{encounter_id}/reextract")
def reextract(encounter_id: int, background: BackgroundTasks, session: Session = Depends(get_session)) -> RedirectResponse:
    e = get_encounter_or_404(session, encounter_id)
    e.status = "interview_complete"
    session.add(e)
    session.commit()
    background.add_task(svc.run_extraction, e.id)
    return RedirectResponse(f"/encounters/{e.id}/record", status_code=303)


@app.get("/encounters/{encounter_id}/pipeline", response_class=HTMLResponse)
def pipeline_page(request: Request, encounter_id: int, session: Session = Depends(get_session)) -> HTMLResponse:
    e = get_encounter_or_404(session, encounter_id)
    p = session.get(Patient, e.patient_id)
    events = list(session.exec(select(PipelineEvent).where(PipelineEvent.encounter_id == e.id).order_by(PipelineEvent.id)).all())
    totals: dict[str, dict[str, Any]] = {}
    for ev in events:
        t = totals.setdefault(ev.step, {"count": 0, "ms": 0, "warn": 0, "error": 0})
        t["count"] += 1
        t["ms"] += ev.latency_ms
        if ev.status in ("warn", "error"):
            t[ev.status] += 1
    lf_url = trace_url(e.langfuse_trace_id)
    return page(request, session, "pipeline.html", encounter=e, patient=p, events=events, totals=totals,
                langfuse_url=lf_url, langfuse_host=get_settings().langfuse_host, nav="pipeline")


@app.get("/pipeline", response_class=HTMLResponse)
def pipeline_index(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    counts = dict(session.exec(select(PipelineEvent.encounter_id, func.count(PipelineEvent.id)).group_by(PipelineEvent.encounter_id)).all())
    rows = session.exec(select(Encounter, Patient).join(Patient).order_by(Encounter.updated_at.desc()).limit(50)).all()
    rows = [(e, p, counts.get(e.id, 0)) for e, p in rows if counts.get(e.id)]
    return page(request, session, "pipeline_index.html", rows=rows, nav="pipeline")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    start = datetime.combine(datetime.now().date(), datetime.min.time())
    encs = list(session.exec(select(Encounter)).all())
    today = [e for e in encs if e.created_at >= start]
    durations = [(e.completed_at - e.created_at).total_seconds() / 60 for e in encs if e.completed_at and e.completed_at > e.created_at and e.questions_asked > 0]
    agent_lat = session.exec(select(func.avg(PipelineEvent.latency_ms)).where(PipelineEvent.step.in_(["agent", "translate"]), PipelineEvent.status != "error", PipelineEvent.model != "")).one()
    step_lat = session.exec(select(PipelineEvent.step, func.avg(PipelineEvent.latency_ms), func.count(PipelineEvent.id)).group_by(PipelineEvent.step)).all()
    filled = unclear = missing = 0
    for e in encs:
        h = e.get_extracted()
        for k in HISTORY_FIELDS:
            st = (h.get(k) or {}).get("status")
            filled += st == "filled"
            unclear += st == "unclear"
            missing += st == "missing"
    guard = session.exec(select(PipelineEvent).where(PipelineEvent.step == "guardrail")).all()
    rates = [d.get("grounding_pass_rate") for d in (g.get_detail() for g in guard) if isinstance(d, dict) and "grounding_pass_rate" in d]
    evals = list(session.exec(select(EvalRun).order_by(EvalRun.created_at.desc()).limit(5)).all())
    status_counts = {s: sum(1 for e in encs if e.status == s) for s in STATUS_META}
    week = []
    for i in range(6, -1, -1):
        d = (datetime.now() - timedelta(days=i)).date()
        week.append({"label": d.strftime("%a"), "count": sum(1 for e in encs if e.created_at.date() == d)})
    return page(
        request, session, "dashboard.html", nav="dashboard",
        interviews_today=len(today), avg_minutes=round(sum(durations) / len(durations), 1) if durations else None,
        avg_gemma_ms=round(agent_lat) if agent_lat else None, step_lat=step_lat,
        filled=filled, unclear=unclear, missing=missing,
        grounding=round(100 * sum(rates) / len(rates)) if rates else None,
        evals=evals, status_counts=status_counts, week=week,
        urgent=sum(1 for e in encs if e.urgent),
    )


# ================================================================ JSON API (voice interview)

@app.get("/api/health")
def health() -> dict[str, Any]:
    s = get_settings()
    status = ollama_status()
    return {
        "ollama": {"ok": status["ok"], "model_present": status["model_present"]},
        "model": s.model_label,
        "elevenlabs": s.elevenlabs_enabled,
        "stt_provider": s.stt_provider,
        "langfuse": get_langfuse() is not None,
        "extractor": s.extractor,
    }


@app.post("/api/interview/{encounter_id}/start")
def api_start(encounter_id: int, background: BackgroundTasks, session: Session = Depends(get_session)) -> dict[str, Any]:
    background.add_task(warm_up)  # load Gemma + cache the agent prompt while the first question plays
    return svc.start_interview(session, get_encounter_or_404(session, encounter_id))


@app.get("/api/interview/{encounter_id}/state")
def api_state(encounter_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    return svc.state_payload(session, get_encounter_or_404(session, encounter_id))


@app.post("/api/interview/{encounter_id}/language")
def api_language(encounter_id: int, body: dict[str, str], session: Session = Depends(get_session)) -> dict[str, Any]:
    lang = body.get("language", "ta")
    if lang not in ("ta", "en"):
        raise HTTPException(400, "language must be ta or en")
    return svc.set_language(session, get_encounter_or_404(session, encounter_id), lang)


@app.post("/api/interview/{encounter_id}/transcribe")
async def api_transcribe(encounter_id: int, audio: UploadFile = File(...), language: str = Form("ta"), session: Session = Depends(get_session)) -> JSONResponse:
    e = get_encounter_or_404(session, encounter_id)
    data = await audio.read()
    if len(data) < 1000:
        return JSONResponse({"ok": False, "error": "Recording too short - hold the mic button while speaking."})
    result = svc.transcribe_audio(e, data, audio.filename or "answer.webm", language)
    return JSONResponse(result)


@app.post("/api/interview/{encounter_id}/answer")
def api_answer(encounter_id: int, body: dict[str, Any], background: BackgroundTasks, session: Session = Depends(get_session)) -> dict[str, Any]:
    e = get_encounter_or_404(session, encounter_id)
    parsed = TextAnswerIn(text=str(body.get("text", "")).strip(), language=body.get("language", e.language))
    if not parsed.text:
        raise HTTPException(400, "Empty answer")
    if body.get("stt_source") == "browser":
        svc.log_browser_stt(e.id, parsed.text, parsed.language, body.get("stt_confidence"), str(body.get("stt_fallback_reason") or ""))
    payload = svc.handle_answer(
        session, e, parsed.text, parsed.language,
        audio_path=str(body.get("audio_path") or ""),
        stt_confidence=body.get("stt_confidence"),
        input_mode=str(body.get("input_mode") or "text"),
    )
    if payload["status"] == "interview_complete":
        background.add_task(svc.run_extraction, e.id)
    return payload


@app.post("/api/interview/{encounter_id}/finish")
def api_finish(encounter_id: int, background: BackgroundTasks, session: Session = Depends(get_session)) -> dict[str, Any]:
    e = get_encounter_or_404(session, encounter_id)
    svc.finish_interview(session, e)
    background.add_task(svc.run_extraction, e.id)
    return {"ok": True, "status": e.status, "record_url": f"/encounters/{e.id}/record"}


@app.get("/api/encounters/{encounter_id}/status")
def api_status(encounter_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    e = get_encounter_or_404(session, encounter_id)
    return {"status": e.status, "label": STATUS_META.get(e.status, (e.status, ""))[0], "flags": e.get_flags()}


@app.post("/api/encounters/{encounter_id}/transcript")
def api_transcript(encounter_id: int, body: TranscriptIn, background: BackgroundTasks, session: Session = Depends(get_session)) -> dict[str, Any]:
    e = get_encounter_or_404(session, encounter_id)
    n = svc.import_transcript(session, e, body.transcript, body.language)
    background.add_task(svc.run_extraction, e.id)
    return {"ok": True, "turns": n}
