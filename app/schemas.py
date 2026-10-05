from typing import Literal, Optional

from pydantic import BaseModel, Field

FieldStatus = Literal["filled", "missing", "unclear"]

# History fields the AI is allowed to fill, in interview order.
HISTORY_FIELDS: dict[str, dict[str, str]] = {
    "chief_complaint": {
        "label": "Chief complaint",
        "ta": "முக்கிய பிரச்சனை",
        "hint": "Main complaint with duration, e.g. 'Fever for 3 days'.",
    },
    "history_of_presenting_illness": {
        "label": "History of presenting illness",
        "ta": "தற்போதைய நோயின் வரலாறு",
        "hint": "Onset, duration, character, progression, aggravating/relieving factors, associated symptoms.",
    },
    "past_history": {
        "label": "Past history",
        "ta": "முந்தைய நோய்கள்",
        "hint": "Previous illnesses (diabetes, hypertension, TB, asthma), surgeries, hospital admissions.",
    },
    "drug_history": {
        "label": "Drug history",
        "ta": "மருந்து வரலாறு",
        "hint": "Current and recent medicines the patient takes.",
    },
    "allergies": {
        "label": "Allergies",
        "ta": "ஒவ்வாமை",
        "hint": "Drug or food allergies, or 'no known allergies'.",
    },
    "family_history": {
        "label": "Family history",
        "ta": "குடும்ப வரலாறு",
        "hint": "Similar illness or chronic disease in family members.",
    },
    "personal_history": {
        "label": "Personal history",
        "ta": "தனிப்பட்ட வரலாறு",
        "hint": "Diet, bowel/bladder habits, sleep, appetite, smoking, alcohol.",
    },
    "occupational_history": {
        "label": "Occupational history",
        "ta": "தொழில்",
        "hint": "Patient's job / work.",
    },
    "occupational_hazards": {
        "label": "Occupational hazards",
        "ta": "தொழில் சார்ந்த ஆபத்துகள்",
        "hint": "Dust, chemicals, smoke, heat, animals or other exposure at work.",
    },
    "pain_score": {
        "label": "Pain score (0-10)",
        "ta": "வலி அளவு",
        "hint": "Only a number 0-10 the patient said themselves.",
    },
}

# Fields that only the doctor may fill. Shown empty on the review form.
DOCTOR_ONLY_FIELDS: dict[str, list[tuple[str, str]]] = {
    "General examination": [
        ("level_of_consciousness", "Level of consciousness"),
        ("build_nourishment", "Build / nourishment"),
        ("piccle", "Pallor / Icterus / Cyanosis / Clubbing / Lymphadenopathy / Edema"),
        ("posture", "Posture"),
        ("gait", "Gait"),
        ("hygiene", "Hygiene"),
        ("hydration", "Hydration"),
        ("signs_of_distress", "Signs of distress"),
    ],
    "Vitals": [
        ("temperature", "Temperature"),
        ("pulse", "Pulse"),
        ("bp", "Blood pressure"),
        ("rr", "Respiratory rate"),
        ("spo2", "SpO2"),
    ],
    "Systemic examination": [
        ("cns", "CNS"),
        ("cvs", "CVS"),
        ("rs", "RS"),
        ("abdomen", "Abdomen"),
    ],
    "Screening & approvals": [
        ("departmental_screening", "Departmental screenings"),
        ("approvals", "Approvals"),
    ],
    "Assessment & plan": [
        ("provisional_diagnosis", "Provisional diagnosis"),
        ("icd_code", "ICD code"),
        ("plan", "Plan"),
        ("next_review_date", "Next review date"),
    ],
}


class ExtractedField(BaseModel):
    value: Optional[str] = None
    source_quote_en: Optional[str] = None
    source_quote_original: Optional[str] = None
    source: str = "voice"  # "voice" | "upload:<file>"
    turn_id: Optional[int] = None
    confidence: float = 0.0
    status: FieldStatus = "missing"
    grounding_score: Optional[float] = None
    note: Optional[str] = None


class OPHistory(BaseModel):
    patient_group: ExtractedField = Field(default_factory=ExtractedField)
    chief_complaint: ExtractedField = Field(default_factory=ExtractedField)
    history_of_presenting_illness: ExtractedField = Field(default_factory=ExtractedField)
    past_history: ExtractedField = Field(default_factory=ExtractedField)
    drug_history: ExtractedField = Field(default_factory=ExtractedField)
    allergies: ExtractedField = Field(default_factory=ExtractedField)
    family_history: ExtractedField = Field(default_factory=ExtractedField)
    personal_history: ExtractedField = Field(default_factory=ExtractedField)
    occupational_history: ExtractedField = Field(default_factory=ExtractedField)
    occupational_hazards: ExtractedField = Field(default_factory=ExtractedField)
    pain_score: ExtractedField = Field(default_factory=ExtractedField)


# ---------- LLM output schemas ----------


class AgentTurn(BaseModel):
    next_question_en: str = ""
    next_question_patient_lang: str = ""
    fields_still_needed: list[str] = Field(default_factory=list)
    fields_covered: list[str] = Field(default_factory=list)
    red_flag: bool = False
    red_flag_reason: str = ""
    answer_unclear: bool = False
    done: bool = False


class AgentFastOut(BaseModel):
    """Compact single-call agent output (translation + next question)."""

    answer_en: str = ""
    covered: list[str] = Field(default_factory=list)
    q_en: str = ""
    q_ta: str = ""
    unclear: bool = False
    red_flag: str = ""
    done: bool = False


class TranslationOut(BaseModel):
    english: str
    detected_language: str = "ta"


class RawField(BaseModel):
    value: Optional[str] = None
    source_quote: Optional[str] = None
    confidence: float = 0.0


class RawExtraction(BaseModel):
    chief_complaint: RawField = Field(default_factory=RawField)
    history_of_presenting_illness: RawField = Field(default_factory=RawField)
    past_history: RawField = Field(default_factory=RawField)
    drug_history: RawField = Field(default_factory=RawField)
    allergies: RawField = Field(default_factory=RawField)
    family_history: RawField = Field(default_factory=RawField)
    personal_history: RawField = Field(default_factory=RawField)
    occupational_history: RawField = Field(default_factory=RawField)
    occupational_hazards: RawField = Field(default_factory=RawField)
    pain_score: RawField = Field(default_factory=RawField)


class Medicine(BaseModel):
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None


class PrescriptionExtraction(BaseModel):
    medicines: list[Medicine] = Field(default_factory=list)
    prescriber_or_hospital: Optional[str] = None
    date: Optional[str] = None
    readable: bool = True
    notes: Optional[str] = None


class AdviceCheck(BaseModel):
    contains_clinical_advice: bool = False
    reason: str = ""


# ---------- API schemas ----------


class TextAnswerIn(BaseModel):
    text: str
    language: Literal["ta", "en"] = "ta"


class TranscriptIn(BaseModel):
    transcript: str
    language: Literal["ta", "en"] = "en"
