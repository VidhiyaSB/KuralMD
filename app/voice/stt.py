"""Speech-to-text behind an interface so a local model can be swapped in for offline mode."""

import io
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)


class STTError(RuntimeError):
    pass


@dataclass
class STTResult:
    text: str
    language: str  # ta / en / other ISO code
    confidence: Optional[float] = None
    provider: str = ""
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class STTProvider(Protocol):
    name: str

    def transcribe(self, audio: bytes, filename: str, language: Optional[str]) -> STTResult: ...


def _norm_lang(code: Optional[str]) -> str:
    code = (code or "").lower()
    if code in ("ta", "tam", "tamil"):
        return "ta"
    if code in ("en", "eng", "english"):
        return "en"
    return code or "unknown"


class ElevenLabsSTT:
    name = "elevenlabs"
    URL = "https://api.elevenlabs.io/v1/speech-to-text"

    def transcribe(self, audio: bytes, filename: str, language: Optional[str]) -> STTResult:
        s = get_settings()
        if not s.elevenlabs_api_key:
            raise STTError("ELEVENLABS_API_KEY is not set")
        data: dict[str, str] = {"model_id": s.elevenlabs_stt_model, "tag_audio_events": "false"}
        if language in ("ta", "en"):
            data["language_code"] = language
        try:
            r = httpx.post(
                self.URL,
                headers={"xi-api-key": s.elevenlabs_api_key},
                data=data,
                files={"file": (filename, audio, "audio/webm")},
                timeout=60,
            )
        except httpx.HTTPError as exc:
            raise STTError(f"ElevenLabs STT network error: {exc}") from exc
        if r.status_code >= 400:
            raise STTError(f"ElevenLabs STT {r.status_code}: {r.text[:300]}")
        body = r.json()
        words = [w for w in body.get("words", []) if w.get("type", "word") == "word"]
        logprobs = [w["logprob"] for w in words if isinstance(w.get("logprob"), (int, float))]
        if logprobs:
            confidence = sum(math.exp(lp) for lp in logprobs) / len(logprobs)
        else:
            confidence = body.get("language_probability")
        return STTResult(
            text=(body.get("text") or "").strip(),
            language=_norm_lang(body.get("language_code") or language),
            confidence=confidence,
            provider=self.name,
            model=s.elevenlabs_stt_model,
            raw={k: v for k, v in body.items() if k != "words"},
        )


class LocalWhisperSTT:
    """Fully offline STT via faster-whisper (pip install faster-whisper)."""

    name = "local"
    _model: Any = None

    def transcribe(self, audio: bytes, filename: str, language: Optional[str]) -> STTResult:
        s = get_settings()
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise STTError("faster-whisper is not installed (pip install faster-whisper)") from exc
        if LocalWhisperSTT._model is None:
            LocalWhisperSTT._model = WhisperModel(s.local_stt_model, device="cpu", compute_type="int8")
        segments, info = LocalWhisperSTT._model.transcribe(
            io.BytesIO(audio), language=language if language in ("ta", "en") else None
        )
        segs = list(segments)
        text = " ".join(seg.text.strip() for seg in segs).strip()
        conf = None
        if segs:
            conf = sum(math.exp(seg.avg_logprob) for seg in segs) / len(segs)
        return STTResult(
            text=text,
            language=_norm_lang(info.language),
            confidence=conf,
            provider=self.name,
            model=f"faster-whisper-{s.local_stt_model}",
        )


def get_stt() -> Optional[STTProvider]:
    provider = get_settings().stt_provider.lower()
    if provider == "elevenlabs":
        return ElevenLabsSTT()
    if provider == "local":
        return LocalWhisperSTT()
    return None
