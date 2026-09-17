from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List

class SurveyStep(BaseModel):
    id: str
    field: str
    question: str
    instruction: str
    skip_if: Optional[str] = None
    type: str = "str"  # "str", "int", "list"
    description: Optional[str] = None

class SurveyConfig(BaseModel):
    survey_id: str
    assistant_name: str = "Aisha"
    system_prompt: str = ""
    llm_provider: str = "openai"  # "openai", "google", "deepseek", "grok"
    chat_model: str = "gpt-5.6-luna"
    stt_provider: str = "elevenlabs"  # "google", "elevenlabs", "openai"
    stt_model: str = "scribe_v2"
    tts_provider: str = "elevenlabs"  # "google", "elevenlabs", "openai", "sarvam"
    tts_voice_id: str = "EXAVITQu4vr4xnSDxMaL"
    tts_model_id: str = "eleven_flash_v2_5"
    initial_greeting: str = "hello how are you?"
    survey_steps: List[SurveyStep] = Field(default_factory=list)

class SurveySession(BaseModel):
    session_id: str
    survey_id: str = "default"
    status: str = "IN_PROGRESS"  # IN_PROGRESS, COMPLETED
    current_question_index: int = 0
    tts_provider: Optional[str] = None
    voice: Optional[str] = None
    total_cost: float = 0.0
    cost_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ConversationTurn(BaseModel):
    session_id: str
    speaker: str  # SYSTEM, CUSTOMER
    text_content: str
    audio_path: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SurveyResult(BaseModel):
    session_id: str
    call_sid: Optional[str] = None
    extracted_data: Dict[str, Any]
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

# This is the target structured data we want the AI to extract from the conversation transcript
class TargetSurveyData(BaseModel):
    full_name: Optional[str] = Field(None, description="The customer's full name. Must be written in English.")
    age: Optional[int] = Field(None, description="The customer's age in years.")
    employment_status: Optional[str] = Field(None, description="Employment status, e.g., Employed, Self-employed, Unemployed, Retired, Student. Must be written in English.")
    annual_income: Optional[int] = Field(None, description="The customer's annual income in Indian Rupees (INR). Convert spoken terms like '15 lakh' to a numeric value (1500000).")
    investment_preferences: Optional[List[str]] = Field(None, description="Preferred investment avenues mentioned by the customer, e.g., Mutual Funds, Stocks, Gold, FDs, etc. Must be written in English.")
    additional_feedback: Optional[str] = Field(None, description="Any other feedback or remarks provided by the customer during the conversation. Must be translated to and written in English.")


class SurveyResponse(BaseModel):
    extracted_data: TargetSurveyData = Field(..., description="The updated extracted survey data based on the entire conversation history including the latest utterance.")
    conversational_response: str = Field(..., description="The next conversational response/question to say to the customer (exactly 1 sentence maximum). Follow the survey rules and prompt instructions.")


