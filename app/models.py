import json
from datetime import datetime
from typing import Any, Optional

from pydantic import NaiveDatetime
from sqlmodel import Field, SQLModel


def now() -> datetime:
    return datetime.now()


ENCOUNTER_STATUSES = [
    "interview_in_progress",
    "interview_complete",
    "needs_clarification",
    "doctor_review",
    "approved",
]


class Patient(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    pid: str = Field(index=True, unique=True)
    name: str
    age: int
    sex: str  # M / F / O
    blood_group: str = ""
    payer: str = "Self Pay"
    phone: str = ""
    preferred_language: str = "ta"  # ta / en
    assigned_student: str = ""
    known_allergies: str = ""
    created_at: NaiveDatetime = Field(default_factory=now)

    @property
    def initials(self) -> str:
        parts = [p for p in self.name.split() if p]
        return "".join(p[0] for p in parts[:2]).upper() or "?"

    @property
    def patient_group(self) -> str:
        if self.age < 18:
            return "Paediatric"
        if self.age >= 60:
            return "Geriatric"
        return "Adult"


class Encounter(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patient.id", index=True)
    department: str = "Community Medicine"
    status: str = "interview_in_progress"
    urgent: bool = False
    red_flag_reason: str = ""
    language: str = "ta"
    extracted_json: str = ""  # OPHistory as JSON
    doctor_fields_json: str = ""  # doctor-only fields typed on review
    flags: str = "[]"  # JSON list of flag strings
    doctor_notes: str = ""
    approved_by: str = ""
    approved_at: Optional[NaiveDatetime] = None
    langfuse_trace_id: str = ""
    agent_state_json: str = ""  # last AgentTurn
    questions_asked: int = 0
    created_at: NaiveDatetime = Field(default_factory=now)
    updated_at: NaiveDatetime = Field(default_factory=now)
    completed_at: Optional[NaiveDatetime] = None

    def get_extracted(self) -> dict[str, Any]:
        return json.loads(self.extracted_json) if self.extracted_json else {}

    def get_doctor_fields(self) -> dict[str, Any]:
        return json.loads(self.doctor_fields_json) if self.doctor_fields_json else {}

    def get_flags(self) -> list[str]:
        try:
            return json.loads(self.flags or "[]")
        except json.JSONDecodeError:
            return []

    def get_agent_state(self) -> dict[str, Any]:
        return json.loads(self.agent_state_json) if self.agent_state_json else {}


class Turn(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    encounter_id: int = Field(foreign_key="encounter.id", index=True)
    role: str  # agent / patient
    text_original: str
    language: str = "en"
    text_english: str = ""
    audio_path: str = ""
    stt_confidence: Optional[float] = None
    input_mode: str = "voice"  # voice / text
    created_at: NaiveDatetime = Field(default_factory=now)


class PipelineEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    encounter_id: Optional[int] = Field(default=None, foreign_key="encounter.id", index=True)
    step: str  # stt | translate | agent | extract | guardrail | tts | vision
    status: str = "ok"  # ok | warn | error
    latency_ms: int = 0
    model: str = ""
    input_summary: str = ""
    output_summary: str = ""
    detail_json: str = ""
    langfuse_observation_id: str = ""
    created_at: NaiveDatetime = Field(default_factory=now)

    def get_detail(self) -> Any:
        try:
            return json.loads(self.detail_json) if self.detail_json else None
        except json.JSONDecodeError:
            return None


class Attachment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    encounter_id: int = Field(foreign_key="encounter.id", index=True)
    filename: str
    path: str
    kind: str = "prescription"
    extracted_json: str = ""
    created_at: NaiveDatetime = Field(default_factory=now)


class EvalRun(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    extractor: str = ""
    model: str = ""
    cases: int = 0
    accuracy: float = 0.0
    per_field_json: str = ""
    created_at: NaiveDatetime = Field(default_factory=now)
