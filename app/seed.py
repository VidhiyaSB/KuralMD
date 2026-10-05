"""Seed the portal with FICTIONAL patients and past encounters (synthetic data only)."""

import json
from datetime import datetime, timedelta

from sqlmodel import Session, select

from app.db import engine, init_db
from app.models import Encounter, Patient, Turn
from app.schemas import ExtractedField, OPHistory

PATIENTS = [
    # name, age, sex, blood, lang, student, allergies
    ("Meenakshi Sundaram", 34, "F", "B+", "ta", "Divya R", ""),
    ("Arul Selvan", 52, "M", "O+", "ta", "Divya R", "Sulfa drugs"),
    ("Kavitha Ramasamy", 27, "F", "A+", "ta", "Harini K", ""),
    ("Murugan Pandian", 67, "M", "AB+", "ta", "Divya R", ""),
    ("Priya Natarajan", 9, "F", "O-", "ta", "Harini K", "Peanuts"),
    ("Rahul Menon", 41, "M", "B-", "en", "Divya R", ""),
    ("Selvi Annamalai", 73, "F", "A-", "ta", "Harini K", "Penicillin"),
    ("Karthik Raja", 23, "M", "O+", "en", "Divya R", ""),
    ("Lakshmi Venkatesan", 58, "F", "B+", "ta", "Harini K", ""),
    ("Dinesh Kumar", 31, "M", "A+", "ta", "Divya R", ""),
]


def _f(value: str, quote: str, quote_ta: str = "") -> dict:
    return ExtractedField(
        value=value, source_quote_en=quote, source_quote_original=quote_ta or quote,
        source="voice", confidence=0.9, status="filled", grounding_score=100.0,
    ).model_dump()


def _past_record(age: int, group: str) -> str:
    h = OPHistory().model_dump()
    h["patient_group"] = ExtractedField(value=group, source="age", status="filled", confidence=1.0).model_dump()
    h["chief_complaint"] = _f("Headache x 2 days", "I have had headache for two days", "ரெண்டு நாளா தலைவலி இருக்கு")
    h["history_of_presenting_illness"] = _f(
        "Gradual onset frontal headache, worse in sun, relieved by rest",
        "it gets worse when I go out in the sun and better when I rest",
        "வெயில்ல போனா அதிகமாகுது, ரெஸ்ட் எடுத்தா சரியாகுது",
    )
    h["past_history"] = _f("No DM / HTN", "no sugar no BP", "சுகர், BP எதுவும் இல்ல")
    h["allergies"] = _f("No known allergies", "no allergies", "அலர்ஜி எதுவும் இல்ல")
    for key in ("drug_history", "family_history", "personal_history", "occupational_history", "occupational_hazards", "pain_score"):
        h[key] = ExtractedField(status="missing").model_dump()
    return json.dumps(h, ensure_ascii=False)


def seed(force: bool = False) -> None:
    init_db()
    with Session(engine) as session:
        if session.exec(select(Patient)).first() and not force:
            return
        now = datetime.now()
        for i, (name, age, sex, blood, lang, student, allergy) in enumerate(PATIENTS, start=1):
            p = Patient(
                pid=f"KMD-{i:04d}", name=name, age=age, sex=sex, blood_group=blood,
                preferred_language=lang, assigned_student=student, known_allergies=allergy,
                phone=f"9{(i * 7919) % 1000000000:09d}",  # synthetic
                created_at=now - timedelta(days=60 - i * 3),
            )
            session.add(p)
            session.commit()
            session.refresh(p)

            # Past approved encounter for every other patient.
            if i % 2 == 0:
                done = now - timedelta(days=20 + i)
                e = Encounter(
                    patient_id=p.id, status="approved", language=lang, created_at=done, updated_at=done,
                    completed_at=done + timedelta(minutes=6), approved_by="Dr. S. Ilango (Faculty)",
                    approved_at=done + timedelta(hours=1), questions_asked=8,
                    extracted_json=_past_record(age, p.patient_group),
                    doctor_fields_json=json.dumps({"provisional_diagnosis": "—", "plan": "Review if persists"}),
                )
                session.add(e)
                session.commit()
                session.refresh(e)
                session.add(Turn(encounter_id=e.id, role="agent", text_original="இன்னைக்கு என்ன பிரச்சனைக்காக வந்திருக்கீங்க?", language="ta", text_english="What problem brings you here today?", created_at=done))
                session.add(Turn(encounter_id=e.id, role="patient", text_original="ரெண்டு நாளா தலைவலி இருக்கு", language="ta", text_english="I have had headache for two days", created_at=done))
            # Today's walk-in waiting for interview.
            if i in (1, 3, 6):
                session.add(Encounter(patient_id=p.id, status="interview_in_progress", language=lang, created_at=now - timedelta(minutes=10 * i)))
        session.commit()


if __name__ == "__main__":
    seed(force=False)
    print("seeded")
