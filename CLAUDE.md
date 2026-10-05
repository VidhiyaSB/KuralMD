# CLAUDE.md — KuralMD

**KuralMD** (Kural / குரல் = "voice" in Tamil): a Tamil + English **voice interview** that takes a patient's medical history and fills the outpatient (OP) record in a sample hospital portal, with all reasoning running locally on Gemma. Every step of the AI pipeline is visible and traceable.

## What we are building
Built for a friend who is a final-year MBBS student. Today she manually interviews each outpatient, asks many follow-up questions (fever: how long, how high, evening rise? cough: dry or productive, sputum colour?), and then types everything into a hospital web portal template. Many patients are more comfortable speaking Tamil.

KuralMD has three parts, all with a real UI:
1. **Patient voice interview:** speaks each question in Tamil or English, listens to the answer, asks clinically sensible follow-ups.
2. **Sample hospital portal:** patient list, patient profile, case records, and the OP record form, where the extracted history lands for the doctor to review, edit, and approve.
3. **Observability:** every AI step (speech-to-text, translation, agent decision, extraction, guardrail check) is traced in **Langfuse** and also shown inside the portal as a step-by-step pipeline timeline per encounter.

The AI only does **history-taking and record extraction**. It never diagnoses, never suggests treatment, and never fills examination findings.

## Hackathon constraints (important)
- DEV "Hacktoberfest Weekend Challenge: Build for a Friend". Must have **open-source AI at its core**.
- **Deadline: Monday 5 Oct 2026, 12:29 PM IST.** Scope ruthlessly. One working Tamil voice interview end to end beats many half features.
- The DEV write-up is weighted most heavily; the demo video must look clean.
- Partner categories: **Gemma** (core reasoning), **ElevenLabs** (Tamil speech in/out).
- **No hosting for now.** Everything runs locally (Ollama on this laptop); the demo is a screen recording. Do not spend time on deployment.
- **Only synthetic data.** Never use real patient names, photos, or IDs. Seed the portal with fictional patients.

## Environment
- Windows, PowerShell, VS Code. Python 3.11+.
- **Ollama** at `http://localhost:11434`, models on `X:\Ollama\models`.
- LLM: **`gemma4:e4b`** (text + images). Fallback: `gemma4:e2b` if slow.
- `.env` (never commit): `OLLAMA_URL`, `MODEL_NAME`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `STT_PROVIDER`, `DATABASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.

## Tech stack (keep it simple)
- **Backend:** FastAPI + Pydantic v2 + SQLModel on **SQLite** (`app.db`).
- **Frontend:** Jinja2 templates + vanilla JS + Tailwind via CDN + Alpine.js (CDN) for small interactivity. No build step.
- **Voice in browser:** `MediaRecorder` push-to-talk, upload audio to backend, play TTS audio via `<audio>`.
- **Speech-to-text:** ElevenLabs Speech-to-Text (Scribe), Tamil (`ta`) / English / auto-detect. Verify current model id and params in docs.
- **Text-to-speech:** ElevenLabs multilingual TTS model that supports Tamil. Verify model id in docs. Cache audio for repeated questions.
- **LLM calls:** Ollama. Prefer Ollama's **OpenAI-compatible endpoint** (`{OLLAMA_URL}/v1`) through the Langfuse OpenAI wrapper (`from langfuse.openai import OpenAI`) so every Gemma call is traced automatically. Use JSON-schema structured output (`response_format`) where supported; otherwise native `/api/chat` with `format` and trace it manually.
- **Extraction + grounding:** Google **LangExtract** with Ollama (check README for current params). If it fights us, fall back to Gemma JSON output + our own fuzzy grounding check (rapidfuzz).
- **Observability:** **Langfuse** Python SDK (check current docs; v3 uses `@observe()` decorator and `get_client()`). Langfuse Cloud free tier is fastest to set up; self-hosting via Docker Compose is the fully private option if time allows. Enable input/output **masking** so phone numbers/names are redacted in traces.
- No Playwright; we own the portal and write directly to the DB.
- Keep STT behind an interface (`stt.py`) so a local model can be swapped in later for fully offline use.

## Suggested structure
```
app/
  main.py              # FastAPI app, routes
  config.py            # env settings
  db.py, models.py     # SQLModel tables
  schemas.py           # Pydantic: OPHistory, ExtractedField, AgentTurn, PipelineEvent
  seed.py              # fictional patients + past encounters for the portal
  llm/
    client.py          # Gemma client (OpenAI-compatible via Langfuse wrapper)
    prompts.py         # all prompts as constants
    intake_agent.py    # decides next question in patient's language
    translate.py       # Tamil -> English per turn
    extractor.py       # transcript -> grounded OPHistory JSON
    guardrails.py      # grounding, red flags, schema validation
  voice/
    stt.py, tts.py     # ElevenLabs implementations behind interfaces
  observability/
    tracing.py         # Langfuse setup, masking, helpers
    events.py          # writes PipelineEvent rows for the in-app timeline
  templates/
    base.html          # app shell: top bar, search, nav
    patients.html      # patient list
    patient_detail.html
    interview.html     # patient voice interview
    record_review.html # OP record form for doctor
    pipeline.html      # per-encounter AI timeline
    dashboard.html     # simple stats
  static/              # js (recorder, player), css, cached audio
samples/               # synthetic prescriptions + ground_truth.json
eval/run_eval.py       # extraction accuracy vs ground truth (logged as Langfuse scores)
```

## Data model
- **Patient:** id, pid (e.g. `KMD-0001`), name, age, sex, blood_group, payer (`Self Pay`), preferred_language (`ta`/`en`), assigned_student, created_at.
- **Encounter:** id, patient_id, department (e.g. Community Medicine), status, urgent, extracted_json, flags, doctor_notes, approved_by, langfuse_trace_id, timestamps.
- **Turn:** encounter_id, role (`agent`/`patient`), text_original, language, text_english, audio_path, created_at.
- **PipelineEvent:** encounter_id, step (`stt` | `translate` | `agent` | `extract` | `guardrail` | `tts` | `vision`), status (`ok` | `warn` | `error`), latency_ms, model, input_summary, output_summary, langfuse_observation_id, created_at.

Encounter status flow: `interview_in_progress → interview_complete → needs_clarification → doctor_review → approved`.

## Fields the AI fills (history section of the OP template)
Each field: `{ value, source_quote_en, source_quote_original, source ("voice" | "upload:<file>"), turn_id, confidence, status: "filled" | "missing" | "unclear" }`.

- `patient_group` (Adult / Paediatric / Geriatric, from age)
- `chief_complaint`: complaint + duration
- `history_of_presenting_illness`: onset, duration, character, progression, aggravating/relieving factors, associated symptoms
- `past_history`, `drug_history`, `allergies` (added for safety), `family_history`
- `personal_history`: diet, bowel/bladder, sleep, appetite, smoking, alcohol
- `occupational_history`, `occupational_hazards`
- `pain_score` only if the patient gives a 0–10 number

## Fields the AI must NEVER fill (doctor only)
General examination, level of consciousness, build/nourishment, PICCLE signs, posture, gait, hygiene, hydration, signs of distress, CNS/CVS/RS/abdomen findings, departmental screenings and approvals, provisional diagnosis, ICD code, plan, next review date. Shown as empty inputs on the review form.

## UI (this matters as much as the backend)
Clean, mobile-first, card-based, similar in spirit to a modern hospital EMR. Calm blue/teal palette, white cards, rounded corners, clear status badges, large tap targets. Must look good in a phone-sized browser window and on desktop. Tamil text uses Noto Sans Tamil (Google Fonts).

1. **App shell:** top bar with KuralMD logo, patient search, notifications count, menu. Left nav on desktop / bottom nav on mobile: Patients, Interviews, Review Queue, Pipeline, Dashboard.
2. **Patients list** (`/patients`): tabs Clinic / Today / History; search by name or PID; rows with avatar initials, name, age·sex·blood group·payer, assigned student, status chip, "View".
3. **Patient detail** (`/patients/{id}`): header card (name, age, sex, PID, badges); Alerts card (allergies, red flags); Case Records list of encounters with status badge; "Start voice interview" button.
4. **Voice interview** (`/interview/{encounter_id}`): big mic button with recording animation, language toggle (தமிழ் / English), current question in large text with replay button, chat bubbles (original + small English line), progress bar of history fields covered, "type instead" fallback, Finish button. Red-flag banner if triggered.
5. **Review queue** (`/review`): encounters awaiting review, urgent first, flagged-field counts.
6. **OP record review** (`/encounters/{id}/record`): form laid out like the hospital OP template (two-column field grid). AI-filled fields pre-populated with a "from patient" chip; clicking shows Tamil + English source quotes; flagged fields highlighted amber; doctor-only fields empty and editable; buttons "Ask follow-up" and "Approve". Approved records show a green "Approved" badge.
7. **Pipeline timeline** (`/encounters/{id}/pipeline`): vertical timeline of PipelineEvents: step icon, status colour, latency, model, short input → output, guardrail results, link "Open in Langfuse".
8. **Dashboard** (`/dashboard`): interviews today, avg interview time, avg Gemma latency, fields auto-filled vs flagged, extraction accuracy from eval runs.

## Voice interview behaviour
- One short question at a time in everyday spoken Tamil (not literary) or English; spoken via TTS and shown as text.
- Patient answers by voice (or types). STT → translate to English → agent decides next question.
- Start with chief complaint, dig in with complaint-specific follow-ups, then cover remaining history fields.
- Follow-ups: **Fever** (duration, pattern, evening rise, chills, max temp, body pain, rash, urinary symptoms, travel); **Cough** (duration, dry/productive, sputum colour/quantity, blood, breathlessness, wheeze, night worsening); **Pain** (site, onset, duration, character, radiation, 0–10 severity, aggravating/relieving, effect of medication).
- Unclear answer or low STT confidence → rephrase once, then move on.
- Agent JSON per turn: `{ next_question_en, next_question_patient_lang, fields_still_needed[], red_flag, red_flag_reason, done }`.
- Stop when fields are covered or after ~15 questions; mark the rest `missing`.

## Guardrails
1. **Grounding:** every extracted value needs a source quote that appears in the English transcript (fuzzy match). Otherwise null + `unclear` + flag. Never guess. Log grounding pass rate as a Langfuse score.
2. **No clinical advice:** never diagnose, name conditions, or recommend medicines.
3. **Red flags:** chest pain, severe breathlessness, fainting, blood in vomit/sputum, suicidal thoughts → stop interview, speak and show "Please inform the doctor/nurse immediately", mark urgent.
4. **Translation safety:** review screen always shows original Tamil next to English.
5. **Privacy:** reasoning, translation, extraction run locally on Gemma. Audio goes to ElevenLabs; if Langfuse Cloud is used, traces leave the machine too, so mask names/phone numbers and say this honestly in the UI and write-up. Self-hosted Langfuse + local STT = fully private mode.
6. **Schema validation:** Pydantic validates all LLM JSON; retry once, then flag.

## Observability rules
- One Langfuse **trace per encounter** (session id = encounter id); one span per pipeline step; Gemma calls appear as generations with model, tokens, latency.
- Also write a `PipelineEvent` row for each step so the in-app timeline works even if Langfuse is down. Tracing failures must never break the app.
- Eval runs push per-field accuracy as Langfuse scores.

## Milestones (in order; demo after each)
- **M1 — Skeleton + portal UI:** app shell, seeded fictional patients, patient list, patient detail, empty OP record form. Langfuse client wired up.
- **M2 — Core extraction:** pasted English transcript → Gemma extractor → grounded JSON → saved → shown on OP record review; trace visible in Langfuse and in-app pipeline timeline.
- **M3 — Tamil voice interview (heart of the demo):** mic → ElevenLabs STT → Gemma translate → agent → ElevenLabs TTS, ending with extraction into the record. Test a full scripted Tamil fever + cough interview.
- **M4 — Review polish:** source quotes, flags, approve flow, urgent path, dashboard numbers.
- **M5 (stretch):** prescription uploads via Gemma vision + eval scores; local STT for offline mode.
- Reserve the last ~2 hours for the DEV write-up and demo video (Tamil interview → record filled → pipeline timeline → Langfuse trace).

## Coding rules for Claude Code
- Small, working increments; run the app after each change and check the UI in the browser.
- All prompts in `app/llm/prompts.py`.
- Handle slow/unavailable Ollama, ElevenLabs, or Langfuse with timeouts, clear UI errors, and fallbacks (text input when voice fails).
- Type hints everywhere; no heavy frameworks beyond the stack above.
- Never add real patient data. Never commit `.env`, `app.db`, or recorded audio.
- Commands (PowerShell):
  - `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
  - `pip install fastapi uvicorn[standard] sqlmodel jinja2 python-multipart httpx pydantic-settings langextract rapidfuzz openai langfuse`
  - `uvicorn app.main:app --reload`
  - Check model: `ollama list` and `ollama run gemma4:e4b "hello"`