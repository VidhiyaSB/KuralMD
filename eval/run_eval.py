"""Extraction accuracy vs synthetic ground truth. Logs an EvalRun row + Langfuse scores.

Usage:  python -m eval.run_eval
"""

import json
from pathlib import Path
from typing import Any

from sqlmodel import Session

from app.config import ROOT_DIR, get_settings
from app.db import engine, init_db
from app.llm.extractor import FIELD_KEYS, PatientSpan, extract_from_document
from app.models import EvalRun, Patient, Turn
from app.observability.tracing import flush, get_langfuse

GT_PATH = ROOT_DIR / "samples" / "ground_truth.json"


def _build(case: dict[str, Any]) -> tuple[str, list[PatientSpan]]:
    parts: list[str] = []
    spans: list[PatientSpan] = []
    pos = 0
    for i, (role, original, english) in enumerate(case["transcript"], start=1):
        prefix = "Patient: " if role == "patient" else "Assistant: "
        line = prefix + english + "\n"
        if role == "patient":
            turn = Turn(id=i, encounter_id=0, role=role, text_original=original, text_english=english,
                        language="ta" if original != english else "en", input_mode="transcript")
            spans.append(PatientSpan(pos + len(prefix), pos + len(prefix) + len(english), turn))
        parts.append(line)
        pos += len(line)
    return "".join(parts), spans


def score_field(expected: Any, field: dict[str, Any]) -> bool:
    value = (field.get("value") or "").lower()
    if expected is None:
        return field.get("status") != "filled"
    return field.get("status") == "filled" and all(k.lower() in value for k in expected)


def main() -> None:
    init_db()
    s = get_settings()
    cases = json.loads(Path(GT_PATH).read_text(encoding="utf-8"))["cases"]
    per_field: dict[str, list[bool]] = {k: [] for k in FIELD_KEYS}
    lf = get_langfuse()
    for case in cases:
        doc, spans = _build(case)
        patient = Patient(pid="EVAL", name="Eval Case", age=case["age"], sex=case["sex"])
        history, flags, stats = extract_from_document(doc, spans, patient, None)
        h = history.model_dump()
        print(f"\n== {case['id']}  (grounding {stats['grounding_pass_rate']:.0%}, extractor {stats.get('extractor')})")
        for key, expected in case["expect"].items():
            ok = score_field(expected, h[key])
            per_field[key].append(ok)
            print(f"  {'PASS' if ok else 'FAIL'}  {key:32s} -> {h[key].get('value')!r}  (expected {expected})")

    field_acc = {k: sum(v) / len(v) for k, v in per_field.items() if v}
    total = [x for v in per_field.values() for x in v]
    accuracy = sum(total) / len(total) if total else 0.0
    print(f"\nOverall accuracy: {accuracy:.1%}")

    with Session(engine) as session:
        session.add(EvalRun(extractor=s.extractor, model=s.model_name, cases=len(cases), accuracy=accuracy,
                            per_field_json=json.dumps(field_acc)))
        session.commit()

    if lf is not None:
        try:
            from langfuse import Langfuse

            trace_id = Langfuse.create_trace_id()
            with lf.start_as_current_observation(name="extraction-eval", trace_context={"trace_id": trace_id},
                                                 input={"cases": len(cases), "extractor": s.extractor}) as span:
                span.update(output={"accuracy": accuracy, "per_field": field_acc})
            lf.create_score(name="eval_accuracy", value=accuracy, trace_id=trace_id)
            for k, v in field_acc.items():
                lf.create_score(name=f"eval_{k}", value=v, trace_id=trace_id)
        except Exception as exc:  # noqa: BLE001
            print("Langfuse logging failed:", exc)
        flush()


if __name__ == "__main__":
    main()
