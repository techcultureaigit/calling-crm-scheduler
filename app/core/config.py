from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

def guess_gender_by_name(name: str) -> str:
    name_lower = name.lower().strip()
    male_names = {"jarvis", "david", "marcus", "peter", "echo", "onyx", "alloy", "fable"}
    female_names = {"sophia", "nova", "sarah", "elena", "shimmer", "ballad", "coral", "sage"}
    if name_lower in male_names:
        return "male"
    if name_lower in female_names:
        return "female"
    return None

class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(default="not_set")
    MONGODB_URL: str = Field(default="mongodb://localhost:27017")
    DATABASE_NAME: str = Field(default="voice_survey")
    ASSISTANT_NAME: str = Field(default="Jarvis")
    VOICE_GENDER: str = Field(default="female")
    TTS_VOICE: str = Field(default="nova")  # alloy, echo, fable, onyx, nova, shimmer
    PORT: int = Field(default=8000)
    CHAT_MODEL: str = Field(default="gpt-5.6-luna")
    STT_PROVIDER: str = Field(default="openai")  # openai or elevenlabs
    STT_MODEL: str = Field(default="whisper-1")      # whisper-1 or scribe_v2
    TTS_MODEL: str = Field(default="tts-1")
    REALTIME_MODEL: str = Field(default="gpt-realtime-2.1-mini") # gpt-realtime-2.1 or gpt-realtime-2.1-mini
    WS_AUTH_TOKEN: str = Field(default="secure-survey-token-12345")
    TTS_PROVIDER: str = Field(default="openai")
    ELEVENLABS_API_KEY: str = Field(default="not_set")
    ELEVENLABS_VOICE_ID: str = Field(default="21m0aVPtns59J8wQ5913")
    ELEVENLABS_MODEL_ID: str = Field(default="eleven_flash_v2_5")
    TTS_SPEECH_SPEED: float = Field(default=1.0)
    USD_TO_INR_RATE: float = Field(default=95.0)
    
    SMARTFLO_EMAIL: str = Field(default="not_set")
    SMARTFLO_PASSWORD: str = Field(default="not_set")
    SMARTFLO_BASE_URL: str = Field(default="https://api-smartflo.tatateleservices.com")
    SMARTFLO_API_TOKEN: str = Field(default="not_set")
    
    CLOUDINARY_CLOUD_NAME: str = Field(default="not_set")
    CLOUDINARY_API_KEY: str = Field(default="not_set")
    CLOUDINARY_API_SECRET: str = Field(default="not_set")
    SARVAM_API_KEY: str = Field(default="not_set")
    GOOGLE_APPLICATION_CREDENTIALS: str = Field(default="voice-survey-505211-674591af47e3.json")
    GOOGLE_API_KEY: str = Field(default="not_set")
    GEMINI_API_KEY: str = Field(default="not_set")
    DEEPGRAM_API_KEY: str = Field(default="not_set")
    DEEPSEEK_API_KEY: str = Field(default="not_set")
    GROK_API_KEY: str = Field(default="not_set")
    XAI_API_KEY: str = Field(default="not_set")
    CALL_WORKFLOW_URLS: list[str] = Field(default=["http://127.0.0.1:8000"])
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def model_post_init(self, __context):
        # Try to guess voice gender automatically based on the assistant name
        guessed_gender = guess_gender_by_name(self.ASSISTANT_NAME)
        if guessed_gender:
            self.VOICE_GENDER = guessed_gender

        # Resolve voice default based on resolved VOICE_GENDER if not explicitly configured differently
        if self.VOICE_GENDER.lower() == "male" and self.TTS_VOICE == "nova":
            self.TTS_VOICE = "onyx"
        elif self.VOICE_GENDER.lower() == "female" and self.TTS_VOICE == "onyx":
            self.TTS_VOICE = "nova"

settings = Settings()
