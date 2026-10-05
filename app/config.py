from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    ollama_url: str = "http://localhost:11434"  # or https://ollama.com for Ollama Cloud direct
    ollama_api_key: str = ""  # only for https://ollama.com (Ollama Cloud) or a proxied Ollama
    model_name: str = "gemma4:e4b"
    fallback_model_name: str = "gemma4:e2b"
    llm_timeout_s: float = 120.0

    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    elevenlabs_stt_model: str = "scribe_v1"
    elevenlabs_tts_model: str = "eleven_multilingual_v2"

    stt_provider: str = "elevenlabs"  # elevenlabs | local | browser | none
    tts_provider: str = "auto"  # auto (ElevenLabs -> Google) | elevenlabs | google | none
    local_stt_model: str = "small"

    database_url: str = f"sqlite:///{(ROOT_DIR / 'app.db').as_posix()}"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_HOST", "LANGFUSE_BASE_URL"),
    )

    extractor: str = "langextract"
    max_questions: int = 15

    @property
    def ollama_v1(self) -> str:
        return self.ollama_url.rstrip("/") + "/v1"

    @property
    def is_cloud_model(self) -> bool:
        return "cloud" in self.model_name or "ollama.com" in self.ollama_url

    @property
    def model_label(self) -> str:
        """Model name shown in the UI. Says 'local' only when it really runs locally."""
        return "Gemma 4 · local" if not self.is_cloud_model else "Gemma 4"

    @property
    def ollama_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.ollama_api_key}"} if self.ollama_api_key else {}

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def elevenlabs_enabled(self) -> bool:
        return bool(self.elevenlabs_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR = STATIC_DIR / "audio"
TTS_CACHE_DIR = AUDIO_DIR / "tts"
RECORDINGS_DIR = AUDIO_DIR / "recordings"
UPLOADS_DIR = STATIC_DIR / "uploads"
for _d in (TTS_CACHE_DIR, RECORDINGS_DIR, UPLOADS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
