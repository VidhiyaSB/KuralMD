"""Intake agent: ONE Gemma call per patient answer that both translates the answer
and decides the next single history question in the patient's language."""

from typing import Any, Optional

from app.config import get_settings
from app.llm import prompts
from app.llm.client import LLMError, chat_json
from app.llm.guardrails import QUESTION_BANK, contains_clinical_advice, detect_red_flags
from app.llm.translate import has_tamil, to_spoken_tamil
from app.models import Encounter, Patient, Turn
from app.observability.tracing import traced_step
from app.schemas import HISTORY_FIELDS, AgentFastOut, AgentTurn

ALL_FIELDS = list(HISTORY_FIELDS.keys())
CODE_FOR = {v: k for k, v in prompts.FIELD_CODES.items()}
LOW_STT_CONFIDENCE = 0.5
HISTORY_TURNS = 16  # only the most recent turns go into the prompt


def format_conversation(turns: list[Turn]) -> str:
    lines: list[str] = []
    for t in turns[-HISTORY_TURNS:]:
        who = "Q" if t.role == "agent" else "A"
        lines.append(f"{who}: {t.text_english or t.text_original}")
    return "\n".join(lines) or "(start)"


def first_question(language: str) -> AgentTurn:
    return AgentTurn(
        next_question_en=prompts.AGENT_FIRST_QUESTION_EN,
        next_question_patient_lang=prompts.AGENT_FIRST_QUESTION.get(language, prompts.AGENT_FIRST_QUESTION_EN),
        fields_still_needed=ALL_FIELDS.copy(),
    )


def _remaining(covered: list[str], skip: Optional[list[str]] = None) -> list[str]:
    skip = skip or []
    return [f for f in ALL_FIELDS if f not in covered and f not in skip and f != "pain_score"]


def _bank_turn(language: str, covered: list[str], asked: list[str]) -> tuple[AgentTurn, Optional[str]]:
    """Offline fallback: next field that is neither covered nor already asked."""
    remaining = _remaining(covered, asked)
    if not remaining:
        return _terminal("done", language, covered), None
    field = remaining[0]
    q = QUESTION_BANK[field]
    turn = AgentTurn(
        next_question_en=q["en"],
        next_question_patient_lang=q["ta"] if language == "ta" else q["en"],
        fields_still_needed=[f for f in ALL_FIELDS if f not in covered],
        fields_covered=covered,
    )
    return turn, field


def _terminal(kind: str, language: str, covered: list[str], reason: str = "") -> AgentTurn:
    msg = prompts.RED_FLAG_MESSAGE if kind == "red_flag" else prompts.DONE_MESSAGE
    return AgentTurn(
        next_question_en=msg["en"],
        next_question_patient_lang=msg.get(language, msg["en"]),
        fields_still_needed=[f for f in ALL_FIELDS if f not in covered],
        fields_covered=covered,
        red_flag=kind == "red_flag",
        red_flag_reason=reason,
        done=True,
    )


def next_turn(
    encounter: Encounter, patient: Patient, turns: list[Turn], answer: Optional[Turn] = None
) -> tuple[AgentTurn, Optional[str], dict[str, Any]]:
    """Decide the next question after the patient's `answer` (already saved in `turns`).

    Returns (agent_turn, answer_english or None, new_agent_state). Applies all guardrails."""
    s = get_settings()
    state = encounter.get_agent_state()
    covered: list[str] = list(state.get("fields_covered", []))
    bank_asked: list[str] = list(state.get("bank_asked", []))
    language = encounter.language
    eid = encounter.id

    # A question-bank field that the patient has now answered counts as covered.
    pending = state.get("pending_field")
    if pending and answer is not None and pending not in covered:
        covered.append(pending)

    base_state = {**state, "fields_covered": covered, "bank_asked": bank_asked, "pending_field": None}

    # 1. Deterministic red-flag guardrail on the latest answer.
    if answer is not None:
        with traced_step(eid, "guardrail", input_summary={"check": "red_flags", "text": answer.text_original}) as g:
            hits = detect_red_flags(answer.text_english, answer.text_original)
            if hits:
                reason = "; ".join(f"{h.category} ('{h.matched}')" for h in hits)
                g.set(output=f"RED FLAG: {reason}", status="warn", detail={"hits": [h.__dict__ for h in hits]})
                return _terminal("red_flag", language, covered, reason), None, base_state
            g.set(output="no red flags")

    # 2. Question limit.
    if encounter.questions_asked >= s.max_questions:
        with traced_step(eid, "guardrail", input_summary={"check": "question_limit"}) as g:
            g.set(output=f"question limit {s.max_questions} reached; ending interview", status="warn")
        return _terminal("done", language, covered), None, base_state

    # 3. Extra instructions: unclear answer / doctor follow-up focus.
    extra = ""
    unclear = False
    if answer is not None:
        conf = answer.stt_confidence
        if len(answer.text_original.strip()) < 2 or (conf is not None and conf < LOW_STT_CONFIDENCE):
            unclear = True
            extra += (
                "\nThe answer was unclear. Ask a different question now."
                if state.get("rephrased_last")
                else "\nThe answer was unclear (low audio confidence). Rephrase the last question more simply."
            )
    prior = state.get("prior")
    if prior:
        pending_prior = {k: v for k, v in prior["fields"].items() if k not in covered}
        if pending_prior:
            known = "; ".join(f"{CODE_FOR.get(k, k)}: {v}" for k, v in pending_prior.items())
            extra += (
                f"\nReturning patient. Known from the last doctor-approved visit ({prior['date']}): {known}. "
                "After the current complaint is clear, ask ONE question that reads these back briefly and asks if anything changed. "
                "If the patient says no change, add all those codes to covered. Do not ask about them one by one."
            )
    focus = state.get("focus_fields") or []
    if focus:
        codes = ", ".join(CODE_FOR.get(f, f) for f in focus)
        extra += f"\nThe DOCTOR asked to clarify only: {codes}. {state.get('focus_note', '')} Ask about these only, then done=true."

    prior = turns[:-1] if answer is not None and turns and turns[-1] is answer else turns
    tamil = answer is not None and has_tamil(answer.text_original)

    # 4. One Gemma call: translate the answer + choose the next question.
    with traced_step(eid, "agent", input_summary=answer.text_original if answer else "(start)") as step:
        try:
            out, meta = chat_json(
                prompts.AGENT_FAST_SYSTEM,
                prompts.AGENT_FAST_USER.format(
                    conversation=format_conversation(prior),
                    language="Tamil" if tamil else "English",
                    answer=answer.text_original if answer else "(none yet)",
                    covered=", ".join(CODE_FOR.get(f, f) for f in covered) or "none",
                    asked=encounter.questions_asked + 1,
                    max_questions=s.max_questions,
                    extra=extra,
                ),
                AgentFastOut,
                name="intake-agent",
                temperature=0.3,
                max_tokens=320,
                metadata={"encounter_id": eid, "question_no": encounter.questions_asked + 1},
            )
            step.model = meta.model
            step.set(
                output=out.q_en,
                detail={"decision": out.model_dump(), "attempts": meta.attempts, "tokens": [meta.prompt_tokens, meta.completion_tokens]},
                status="ok" if meta.attempts == 1 else "warn",
            )
        except LLMError as exc:
            step.set(output=f"Gemma unavailable, using question bank: {exc}", status="error")
            out = None

    if out is None:
        fallback, field = _bank_turn(language, covered, bank_asked)
        if field:
            bank_asked.append(field)
        return fallback, None, {**base_state, "bank_asked": bank_asked, "pending_field": field, "rephrased_last": False, "fallback": True}

    answer_en = out.answer_en.strip() or None
    if answer is not None and answer_en and tamil:
        # Record the translation as its own pipeline step (it came from the same Gemma call).
        with traced_step(eid, "translate", input_summary=answer.text_original, model=meta.model) as t:
            t.set(output=answer_en, detail={"via": "intake-agent (single call)"})

    covered = sorted(set(covered) | {prompts.FIELD_CODES.get(c.strip().lower(), c) for c in out.covered} & set(ALL_FIELDS))
    new_state = {**base_state, "fields_covered": covered, "rephrased_last": unclear and not state.get("rephrased_last", False), "fallback": False}
    agent = AgentTurn(
        next_question_en=out.q_en.strip(),
        next_question_patient_lang=out.q_ta.strip() if language == "ta" else out.q_en.strip(),
        fields_covered=covered,
        fields_still_needed=[f for f in ALL_FIELDS if f not in covered],
        red_flag=bool(out.red_flag.strip()),
        red_flag_reason=out.red_flag.strip(),
        answer_unclear=out.unclear,
        done=out.done,
    )

    if agent.red_flag:
        with traced_step(eid, "guardrail", input_summary={"check": "agent_red_flag"}) as g:
            g.set(output=f"RED FLAG (agent): {agent.red_flag_reason}", status="warn")
        return _terminal("red_flag", language, covered, agent.red_flag_reason), answer_en, new_state

    if agent.done or not agent.next_question_en:
        if _remaining(covered) and not agent.done:
            fallback, field = _bank_turn(language, covered, bank_asked)
            return fallback, answer_en, {**new_state, "pending_field": field}
        return _terminal("done", language, covered), answer_en, new_state

    # 5. No-clinical-advice guardrail on what we are about to say.
    with traced_step(eid, "guardrail", input_summary={"check": "no_clinical_advice", "text": agent.next_question_en}) as g:
        hit = contains_clinical_advice(agent.next_question_en) or contains_clinical_advice(agent.next_question_patient_lang)
        if hit:
            g.set(output=f"blocked clinical advice ('{hit}'); replaced with safe question", status="warn")
            safe, field = _bank_turn(language, covered, bank_asked)
            agent.next_question_en, agent.next_question_patient_lang = safe.next_question_en, safe.next_question_patient_lang
            new_state["pending_field"] = field
        else:
            g.set(output="passed")

    # 6. Make sure a Tamil patient hears Tamil.
    if language == "ta" and not has_tamil(agent.next_question_patient_lang):
        agent.next_question_patient_lang = to_spoken_tamil(eid, agent.next_question_en) or agent.next_question_en

    return agent, answer_en, new_state
