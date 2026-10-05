# KuralMD · குரல்

A bilingual (Tamil + English) **voice interview** that takes an outpatient's medical history and fills the history section of the OP record in a sample hospital portal. Reasoning runs on **Gemma 4** through Ollama, and every AI step is traced, both in an in-app pipeline timeline and in **Langfuse**.

> The AI only takes the history and drafts the record. It never diagnoses, never suggests treatment and never fills examination findings. The doctor reviews, edits and approves. All patient data in this repo is synthetic.

Built for the DEV Hacktoberfest Weekend Challenge: *Build for a Friend*.

## What it does

- **Voice interview.** The patient answers by voice in Tamil or English. One short question at a time, spoken aloud, with complaint-specific follow-ups (fever, cough, pain).
- **OP record.** Chief complaint, presenting illness, past, drug, allergy, family, personal and occupational history are filled from the transcript. Each field links to the patient's exact words in Tamil and English.
- **Doctor review.** Unclear or ungrounded fields are highlighted. The doctor can edit fields, ask the patient a targeted follow-up, or approve.
- **Returning patients.** History from the last approved visit is read back for confirmation instead of being asked again.
- **Old prescriptions.** A photo is read by Gemma 4 vision into drug history.
- **Observability.** One Langfuse trace per encounter with names and phone numbers masked. The same steps appear as a timeline inside the portal.

## Pipeline
```
mic → ElevenLabs Scribe (STT) ─┐  fallback: Chrome speech recognition
                               ▼
Gemma 4: translate the answer + choose the next question (one call)
  → guardrails: red flags (EN + Tamil keywords + model), no-clinical-advice check
  → ElevenLabs TTS (Tamil)      fallback: Google TTS (gTTS), then browser speech
finish → LangExtract on Gemma 4 → grounding check (rapidfuzz) → OP record → doctor approves
```

## Guardrails
- **Grounding:** every extracted value needs a quote that is found in the patient's words. Otherwise the field is left blank, marked *unclear* and flagged.
- **Red flags:** chest pain, severe breathlessness, fainting, blood in vomit or sputum, and suicidal thoughts stop the interview and mark the encounter urgent.
- **No advice:** questions that contain diagnosis or treatment language are replaced with a safe question.
- **Schema validation:** every Gemma JSON output is validated with Pydantic and retried once.

## Run it (Windows / PowerShell)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # add ELEVENLABS_API_KEY and LANGFUSE_* keys
ollama pull gemma4:e4b
uvicorn app.main:app
```
Open http://127.0.0.1:8000 in Chrome. Fictional patients are seeded on first start. Use **Quick walk-in** to start an interview right away.

### Model options (`.env`)
| Setup | Settings |
|---|---|
| Local (default) | `OLLAMA_URL=http://localhost:11434`, `MODEL_NAME=gemma4:e4b` |
| Faster local | `MODEL_NAME=gemma4:e2b` (Tamil phrasing is weaker) |

Everything runs against your local Ollama. On a CPU-only laptop, e4b takes roughly 45 to 55 s per question; a GPU makes it much faster. `OLLAMA_URL` can also point at another Ollama server on your network (set `OLLAMA_API_KEY` if that server requires auth).

### Other switches
- `EXTRACTOR=json` uses Gemma JSON with rapidfuzz grounding instead of LangExtract. LangExtract automatically falls back to this if it returns nothing.
- `TTS_PROVIDER=auto|elevenlabs|google|none` and `STT_PROVIDER=elevenlabs|local|browser|none`. `local` requires `pip install faster-whisper`.
- Without Langfuse keys, the in-app pipeline timeline still works.

## Extras
- `python samples/make_prescriptions.py` generates synthetic prescription images (printed and handwritten-style) for the upload demo.
- `python -m eval.run_eval` measures extraction accuracy against `samples/ground_truth.json`. Results show on the dashboard and as Langfuse scores.

## Privacy
Voice audio goes to ElevenLabs. Traces go to Langfuse with names, phone numbers and emails masked. For a fully private setup, run Gemma locally, self-host Langfuse and use `STT_PROVIDER=local`.

## Stack
FastAPI · SQLModel/SQLite · Jinja2 · Tailwind (CDN) · Alpine.js · Ollama · Gemma 4 · ElevenLabs · Google LangExtract · Langfuse · rapidfuzz

## License
MIT
