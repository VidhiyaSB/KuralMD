"""Text-to-speech with an on-disk cache.

Providers: ElevenLabs multilingual (supports Tamil) and Google (gTTS, no key needed).
TTS_PROVIDER=auto tries ElevenLabs first and falls back to Google.
If every provider fails, the browser falls back to speechSynthesis."""

import hashlib
import io
import logging
from typing import Optional, Protocol

import httpx

from app.config import TTS_CACHE_DIR, get_settings

log = logging.getLogger(__name__)

DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # "Sarah" premade voice (free-plan OK); override with ELEVENLABS_VOICE_ID


class TTSError(RuntimeError):
    pass


class TTSProvider(Protocol):
    name: str
    model: str

    def synthesize(self, text: str, language: str) -> tuple[str, bool]:
        """Returns (static url path, cache_hit)."""
        ...


def _cached(key_src: str) -> tuple[str, "object", bool]:
    key = hashlib.sha1(key_src.encode()).hexdigest()[:20]
    path = TTS_CACHE_DIR / f"{key}.mp3"
    return f"/static/audio/tts/{key}.mp3", path, path.exists() and path.stat().st_size > 0


class ElevenLabsTTS:
    name = "elevenlabs"

    @property
    def model(self) -> str:
        return get_settings().elevenlabs_tts_model

    def synthesize(self, text: str, language: str) -> tuple[str, bool]:
        s = get_settings()
        if not s.elevenlabs_api_key:
            raise TTSError("ELEVENLABS_API_KEY is not set")
        voice = s.elevenlabs_voice_id or DEFAULT_VOICE_ID
        url, path, hit = _cached(f"{s.elevenlabs_tts_model}|{voice}|{text}")
        if hit:
            return url, True
        try:
            r = httpx.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
                params={"output_format": "mp3_44100_128"},
                headers={"xi-api-key": s.elevenlabs_api_key, "accept": "audio/mpeg"},
                json={
                    "text": text,
                    "model_id": s.elevenlabs_tts_model,
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                },
                timeout=60,
            )
        except httpx.HTTPError as exc:
            raise TTSError(f"ElevenLabs TTS network error: {exc}") from exc
        if r.status_code >= 400:
            raise TTSError(f"ElevenLabs TTS {r.status_code}: {r.text[:300]}")
        path.write_bytes(r.content)
        return url, False


class GoogleTTS:
    """Google Translate TTS via gTTS: free, no key, supports Tamil ('ta')."""

    name = "google"
    model = "gtts"

    def synthesize(self, text: str, language: str) -> tuple[str, bool]:
        lang = "ta" if language == "ta" else "en"
        tld = "co.in"  # Indian English accent for English questions
        url, path, hit = _cached(f"gtts|{lang}|{tld}|{text}")
        if hit:
            return url, True
        try:
            from gtts import gTTS

            buf = io.BytesIO()
            gTTS(text, lang=lang, tld=tld).write_to_fp(buf)
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"Google TTS failed: {exc}") from exc
        path.write_bytes(buf.getvalue())
        return url, False


class FallbackTTS:
    """Try providers in order; remembers the last provider used."""

    name = "auto"

    def __init__(self, providers: list[TTSProvider]) -> None:
        self.providers = providers
        self.used: Optional[TTSProvider] = None
        self.errors: list[str] = []

    @property
    def model(self) -> str:
        return self.used.model if self.used else ""

    def synthesize(self, text: str, language: str) -> tuple[str, bool]:
        self.errors = []
        for p in self.providers:
            try:
                result = p.synthesize(text, language)
                self.used = p
                return result
            except TTSError as exc:
                self.errors.append(f"{p.name}: {exc}")
                log.warning("TTS provider %s failed: %s", p.name, exc)
        raise TTSError(" | ".join(self.errors))


def get_tts() -> Optional[FallbackTTS]:
    s = get_settings()
    provider = s.tts_provider.lower()
    if provider == "none":
        return None
    if provider == "google":
        return FallbackTTS([GoogleTTS()])
    if provider == "elevenlabs":
        return FallbackTTS([ElevenLabsTTS()])
    chain: list[TTSProvider] = []
    if s.elevenlabs_enabled:
        chain.append(ElevenLabsTTS())
    chain.append(GoogleTTS())
    return FallbackTTS(chain)
