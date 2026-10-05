"""Deterministic guardrails: red flags, no-clinical-advice, grounding, question bank."""

import re
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz

# ---------------------------------------------------------------- red flags

RED_FLAG_PATTERNS: dict[str, list[str]] = {
    "Chest pain": [r"chest pain", r"pain in (?:my |the )?chest", r"chest (?:is )?(?:tight|heavy)", r"நெஞ்சு ?வலி", r"நெஞ்சுல வலி", r"நெஞ்சு அடைப்பு"],
    "Severe breathlessness": [
        r"can(?:'|no)?t breathe", r"cannot breathe", r"unable to breathe", r"severe(?:ly)? breathless",
        r"gasping", r"மூச்சு ?விட ?முடியல", r"மூச்சு ?திணறல்", r"மூச்சு வாங்குது ரொம்ப",
    ],
    "Fainting": [r"faint(?:ed|ing)?", r"passed out", r"lost consciousness", r"unconscious", r"blacked out", r"மயக்கம்", r"மயங்கி"],
    "Blood in vomit": [r"vomit(?:ing|ed)? blood", r"blood in (?:the |my )?vomit", r"ரத்த ?வாந்தி", r"வாந்தில ரத்தம்"],
    "Blood in sputum": [
        r"cough(?:ing|ed)? (?:up )?blood", r"blood in (?:the |my )?(?:sputum|phlegm|cough)", r"blood(?:y)? (?:sputum|phlegm)",
        r"சளியில ?ரத்தம்", r"சளில ரத்தம்", r"இருமல்ல ரத்தம்", r"ரத்தமா வருது",
    ],
    "Suicidal thoughts": [
        r"suicid", r"kill myself", r"end my life", r"want to die", r"self[- ]harm", r"தற்கொலை", r"சாகணும்", r"செத்துடலாம்",
    ],
}

_NEG_BEFORE = re.compile(r"\b(?:no|not|never|without|denies|deny|don'?t|doesn'?t|didn'?t|nil)\b[\w\s,]{0,25}$", re.I)
_NEG_AFTER_TA = re.compile(r"^[\s\w]{0,12}(?:இல்ல|இல்லை|கிடையாது)")


@dataclass
class RedFlagHit:
    category: str
    matched: str


def detect_red_flags(*texts: Optional[str]) -> list[RedFlagHit]:
    hits: list[RedFlagHit] = []
    for text in texts:
        if not text:
            continue
        for category, patterns in RED_FLAG_PATTERNS.items():
            if any(h.category == category for h in hits):
                continue
            for pat in patterns:
                for m in re.finditer(pat, text, flags=re.IGNORECASE):
                    before = text[max(0, m.start() - 30) : m.start()]
                    after = text[m.end() : m.end() + 20]
                    if _NEG_BEFORE.search(before) or _NEG_AFTER_TA.search(after):
                        continue
                    hits.append(RedFlagHit(category, m.group(0)))
                    break
                if any(h.category == category for h in hits):
                    break
    return hits


# ---------------------------------------------------------------- no clinical advice

_ADVICE_PATTERNS = [
    r"\byou (?:may|might|probably|likely|could) have (?:a |an )?(?:\w+ )?(?:infection|disease|typhoid|dengue|malaria|tb|tuberculosis|pneumonia|covid|flu|asthma|cancer|ulcer|gastritis|migraine)\b",
    r"\bdiagnos", r"\bit (?:is|could be|may be|might be|sounds like)\b.*\b(?:infection|fever|disease|typhoid|dengue|malaria|tb|tuberculosis|pneumonia|covid|flu)\b",
    r"\b(?:take|start|stop|continue) (?:a |the |this )?(?:tablet|medicine|paracetamol|antibiotic|dolo|crocin)",
    r"\b\d+\s?mg\b", r"\bprescri(?:be|ption for you)", r"\byou should (?:take|get|do) (?:a |an )?(?:test|scan|x-?ray|blood test)",
    r"\bdon'?t worry\b", r"\bnothing serious\b",
]


def contains_clinical_advice(text: str) -> Optional[str]:
    for pat in _ADVICE_PATTERNS:
        m = re.search(pat, text or "", flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None


# ---------------------------------------------------------------- grounding

GROUNDING_THRESHOLD = 80.0


def grounding_score(quote: Optional[str], transcript: str) -> float:
    if not quote or not transcript:
        return 0.0
    q = re.sub(r"\s+", " ", quote.strip().lower())
    t = re.sub(r"\s+", " ", transcript.lower())
    if q in t:
        return 100.0
    return float(fuzz.partial_ratio(q, t))


def is_grounded(quote: Optional[str], transcript: str, threshold: float = GROUNDING_THRESHOLD) -> bool:
    return grounding_score(quote, transcript) >= threshold


# ---------------------------------------------------------------- fallback question bank
# Used when Gemma is unavailable or a generated question fails a guardrail.

QUESTION_BANK: dict[str, dict[str, str]] = {
    "chief_complaint": {"en": "What problem brings you here today, and since how many days?", "ta": "இன்னைக்கு என்ன பிரச்சனைக்காக வந்திருக்கீங்க? எத்தனை நாளா இருக்கு?"},
    "history_of_presenting_illness": {"en": "How did it start, and is it getting better or worse?", "ta": "இது எப்படி ஆரம்பிச்சது? கூடுதா, குறையுதா?"},
    "past_history": {"en": "Do you have sugar, BP, asthma, TB or any old illness?", "ta": "உங்களுக்கு சுகர், BP, ஆஸ்துமா, TB மாதிரி ஏதாவது முன்னாடி இருந்துச்சா?"},
    "drug_history": {"en": "Are you taking any tablets or medicines now?", "ta": "இப்போ ஏதாவது மாத்திரை, மருந்து சாப்பிடறீங்களா?"},
    "allergies": {"en": "Are you allergic to any medicine or food?", "ta": "ஏதாவது மருந்து இல்ல சாப்பாடு உங்களுக்கு ஒத்துக்காதா? அலர்ஜி இருக்கா?"},
    "family_history": {"en": "Does anyone in your family have a similar problem or long-term illness?", "ta": "உங்க வீட்டுல யாருக்காவது இதே மாதிரி இல்ல சுகர், BP மாதிரி நோய் இருக்கா?"},
    "personal_history": {"en": "How are your appetite, sleep, and do you smoke or drink alcohol?", "ta": "பசி, தூக்கம் எல்லாம் சரியா இருக்கா? சிகரெட், மது பழக்கம் இருக்கா?"},
    "occupational_history": {"en": "What work do you do?", "ta": "நீங்க என்ன வேலை பாக்கறீங்க?"},
    "occupational_hazards": {"en": "At work, are you exposed to dust, smoke, chemicals or heat?", "ta": "வேலை இடத்துல தூசி, புகை, கெமிக்கல், வெயில் மாதிரி ஏதாவது படுதா?"},
    "pain_score": {"en": "From 0 to 10, how bad is the pain?", "ta": "0 முதல் 10 வரை சொன்னா, வலி எவ்வளவு இருக்கு?"},
}
