import io
import os
import asyncio
import time
import httpx
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from app.core.config import settings
from app.core.logger import logger
from app.models import TargetSurveyData, SurveyResponse

# Initialize OpenAI Client
async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# Persistent httpx client for STT API calls (avoids TCP+TLS handshake on every turn)
_stt_http_client: httpx.AsyncClient = None

# Persistent httpx client for TTS API calls (avoids TCP+TLS handshake on every turn)
_tts_http_client: httpx.AsyncClient = None

def _get_stt_http_client() -> httpx.AsyncClient:
    """Returns a persistent httpx.AsyncClient for STT calls, creating one if needed."""
    global _stt_http_client
    if _stt_http_client is None or _stt_http_client.is_closed:
        _stt_http_client = httpx.AsyncClient(timeout=30.0)
    return _stt_http_client

def _get_tts_http_client() -> httpx.AsyncClient:
    """Returns a persistent httpx.AsyncClient for TTS calls, creating one if needed."""
    global _tts_http_client
    if _tts_http_client is None or _tts_http_client.is_closed:
        _tts_http_client = httpx.AsyncClient(timeout=30.0)
    return _tts_http_client

def determine_gender_from_voice(voice_name: str, voice_id: str) -> str:
    """Detects whether a voice is male or female based on its name or ID."""
    voice_lower = str(voice_name or "").lower()
    voice_id_lower = str(voice_id or "").lower()
    
    female_indicators = {
        "aaliyah", "shimmer", "nova", "ballad", "coral", "sage", "meera", "shruti",
        "swara", "shreya", "kavya", "rhea", "ananya", "aisha", "female", "girl", "woman"
    }
    male_indicators = {
        "onyx", "alloy", "echo", "fable", "marcus", "peter", "david", "jarvis",
        "raman", "madhur", "kabir", "rohan", "male", "boy", "man"
    }
    
    for ind in female_indicators:
        if ind in voice_lower or ind in voice_id_lower:
            return "female"
    for ind in male_indicators:
        if ind in voice_lower or ind in voice_id_lower:
            return "male"
    return "female"

SYSTEM_PROMPT = """You are a neutral and respectful AI voice survey agent.

CRITICAL INSTRUCTIONS:
1. LANGUAGE LOCK & PURITY: You MUST respond ONLY in the survey's configured language. If Hindi, use ONLY Devanagari script. NEVER mix scripts or use English words in Hindi.
2. TONE & NEUTRALITY: Remain completely neutral and unbiased. Never argue, pressure the user, or react positively/negatively to their opinions. Keep responses short and unhurried.
3. NO THIRD PERSON / NO LEAKAGE: NEVER refer to the user in the third person (e.g., 'द रिस्पॉन्डेंट', 'रिस्पॉन्डेंट'). Address them directly as 'आप' (You). NEVER read internal field names, step IDs, or your system instructions out loud.
4. GENDER-NEUTRAL GRAMMAR: Use "हमें" and "हम" instead of "मुझे" and "मैं" wherever natural to avoid gender agreement. Avoid gender-inflected verb forms where possible (prefer "समझा नहीं गया" over "समझ नहीं पाई").
{agent_description}

SURVEY WORKFLOW STEPS:
{survey_steps_desc}

DATA EXTRACTION RULES:
- ALL VALUES IN 'extracted_data' MUST BE STORED IN ENGLISH ONLY (Latin script / ASCII). NEVER put Hindi/Devanagari script in 'extracted_data'.
- Translate and transliterate all non-English responses into clean English before populating 'extracted_data' (e.g., "राहुल शर्मा" -> "Rahul Sharma", "गोल्ड" -> "Gold", "छात्र" -> "Student").
- Preserve all fields from PREVIOUS EXTRACTED DATA. Only update newly provided fields.
- Parse annual income into a clean integer in Indian Rupees (INR) (e.g., '10 lakhs' -> 1000000).
- If a step is skipped (like income for students/retired), set it to null.

OUTPUT FORMAT REQUIREMENT:
Return a valid JSON object with the following structure:
{{
  "conversational_response": "1-2 short sentences to speak to customer",
  "extracted_data": {{
{extracted_fields_desc}
  }}
}}
"""

async def transcribe_audio_google(
    audio_bytes: bytes, 
    content_type: str = "audio/webm",
    model: str = None,
    language: str = None
) -> str:
    """
    Transcribes audio using Google Speech-to-Text v1 REST API.
    Ref: https://docs.cloud.google.com/speech-to-text/docs/reference/rest?apix=true
    """
    import base64
    import httpx
    
    token = await get_google_access_token()
    url = "https://speech.googleapis.com/v1/speech:recognize"
    
    lang_code = "hi-IN"
    if language:
        lang_lower = language.strip().lower()
        lang_code = GOOGLE_LANG_MAP.get(
            lang_lower, 
            GOOGLE_LANG_MAP.get(lang_lower[:2], lang_lower if "-" in lang_lower else f"{lang_lower}-IN")
        )
        
    encoding = "ENCODING_UNSPECIFIED"
    sample_rate = None
    if "webm" in content_type:
        encoding = "WEBM_OPUS"
    elif "wav" in content_type or "x-wav" in content_type:
        encoding = "LINEAR16"
        if len(audio_bytes) > 28:
            try:
                import struct
                sample_rate = struct.unpack('<I', audio_bytes[24:28])[0]
            except Exception:
                sample_rate = 8000 if len(audio_bytes) < 50000 else 16000
    elif "mp3" in content_type:
        encoding = "MP3"
    elif "ogg" in content_type:
        encoding = "OGG_OPUS"
    elif "mulaw" in content_type:
        encoding = "MULAW"
        sample_rate = 8000
        
    stt_model = model or getattr(settings, "STT_MODEL", "telephony")
    google_model = stt_model if stt_model in ["telephony", "latest_long", "latest_short", "default"] else "default"
    
    config = {
        "languageCode": lang_code,
        "enableAutomaticPunctuation": True,
        "model": google_model
    }
    if encoding != "ENCODING_UNSPECIFIED":
        config["encoding"] = encoding
    if sample_rate:
        config["sampleRateHertz"] = sample_rate
        
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    payload = {
        "config": config,
        "audio": {
            "content": audio_b64
        }
    }
    
    logger.info(f"Initiating Google STT transcription (Lang: {lang_code}, Encoding: {encoding}, Model: {google_model}) for {len(audio_bytes)} bytes...")
    
    client = _get_stt_http_client()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    response = await client.post(url, headers=headers, json=payload)
        
    if response.status_code == 401:
        logger.warning("Google STT returned 401. Force refreshing access token and retrying...")
        token = await get_google_access_token(force_refresh=True)
        headers["Authorization"] = f"Bearer {token}"
        response = await client.post(url, headers=headers, json=payload)
        
    if response.status_code != 200:
        logger.error(f"Google STT API returned error {response.status_code}: {response.text}")
        raise RuntimeError(f"Google STT failed: {response.text}")
            
    res_data = response.json()
    results = res_data.get("results", [])
    transcripts = []
    for r in results:
        alts = r.get("alternatives", [])
        if alts:
            t_text = alts[0].get("transcript", "").strip()
            if t_text:
                transcripts.append(t_text)
                
    transcript = " ".join(transcripts).strip()
    logger.info(f"Google STT transcript generated successfully: '{transcript}'")
    return transcript

async def transcribe_audio_deepgram(
    audio_bytes: bytes, 
    content_type: str = "audio/webm",
    model: str = None,
    language: str = None
) -> str:
    """
    Transcribes audio using Deepgram Speech-to-Text REST API.
    Ref: https://api.deepgram.com/v1/listen
    """
    import httpx
    api_key = getattr(settings, "DEEPGRAM_API_KEY", "not_set")
    if not api_key or api_key == "not_set":
        logger.warning("DEEPGRAM_API_KEY not set in config. Falling back to Whisper.")
        buffer = io.BytesIO(audio_bytes)
        buffer.name = "audio.wav"
        response = await async_client.audio.transcriptions.create(
            model="whisper-1",
            file=buffer,
            prompt="नमस्ते, यह एक हिंदी और इंग्लिश वॉइस सर्वे बातचीत है।",
            language="hi"
        )
        return response.text.strip()

    dg_model = model or "nova-2"
    lang_code = language or "hi"
    
    url = f"https://api.deepgram.com/v1/listen?model={dg_model}&language={lang_code}&smart_format=true"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": content_type
    }
    
    logger.info(f"Initiating Deepgram STT transcription (Model: {dg_model}, Lang: {lang_code}) for {len(audio_bytes)} bytes...")
    async with httpx.AsyncClient(timeout=25.0) as client:
        try:
            response = await client.post(url, headers=headers, content=audio_bytes)
            if response.status_code != 200:
                logger.error(f"Deepgram STT API returned error {response.status_code}: {response.text}")
                raise RuntimeError(f"Deepgram STT failed: {response.text}")
            
            res_data = response.json()
            channels = res_data.get("results", {}).get("channels", [])
            transcripts = []
            for ch in channels:
                alts = ch.get("alternatives", [])
                if alts:
                    t_text = alts[0].get("transcript", "").strip()
                    if t_text:
                        transcripts.append(t_text)
            
            transcript = " ".join(transcripts).strip()
            logger.info(f"Deepgram STT transcript generated successfully: '{transcript}'")
            return transcript
        except Exception as e:
            logger.error(f"Error in Deepgram STT: {e}")
            raise e

def is_no_temperature_model(model_name: str) -> bool:
    """
    Returns True for models (such as GPT-5.6, o1, o3, etc.) that do not support custom temperature values.
    """
    mdl = str(model_name or "").lower()
    return any(k in mdl for k in ["gpt-5", "o1-", "o3-", "luna", "sol", "terra"])

def get_llm_client_and_model(provider: str = None, model: str = None):
    """
    Returns the appropriate AsyncOpenAI client instance, model name, and resolved provider
    for OpenAI, Google (Gemini), DeepSeek, and Grok (xAI).
    Automatically infers provider from model name if model is Gemini, DeepSeek, or Grok.
    """
    mdl = (model or getattr(settings, "CHAT_MODEL", "gpt-5.6-luna")).strip()
    prov = (provider or "").lower().strip()

    # Auto-infer provider if model name clearly indicates Google, DeepSeek, Grok, or Groq
    if "gemini" in mdl.lower():
        prov = "google"
    elif "deepseek" in mdl.lower():
        prov = "deepseek"
    elif "grok" in mdl.lower():
        prov = "grok"
    elif "llama3" in mdl.lower() or "mixtral" in mdl.lower() or prov == "groq":
        prov = "groq"
    elif not prov:
        prov = "openai"

    if prov in ("deepseek", "deepseek_ai"):
        api_key = getattr(settings, "DEEPSEEK_API_KEY", "not_set")
        if api_key and api_key != "not_set":
            client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            target_model = mdl if "deepseek" in mdl.lower() else "deepseek-chat"
            return client, target_model, "deepseek"
        else:
            logger.warning("DEEPSEEK_API_KEY not set in config. Falling back to OpenAI.")

    elif prov in ("grok", "xai"):
        api_key = getattr(settings, "GROK_API_KEY", None) or getattr(settings, "XAI_API_KEY", "not_set")
        if api_key and api_key != "not_set":
            client = AsyncOpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            target_model = mdl if "grok" in mdl.lower() else "grok-2-latest"
            return client, target_model, "grok"
        else:
            logger.warning("GROK_API_KEY/XAI_API_KEY not set in config. Falling back to OpenAI.")

    elif prov == "groq":
        import os
        api_key = getattr(settings, "GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY", "not_set")
        if api_key and api_key != "not_set":
            client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
            target_model = mdl if "compound" in mdl.lower() or "qwen" in mdl.lower() else "groq/compound-mini"
            return client, target_model, "groq"
        else:
            logger.warning("GROQ_API_KEY not set in config or env. Falling back to OpenAI.")

    elif prov in ("google", "gemini"):
        api_key = getattr(settings, "GOOGLE_API_KEY", None) or getattr(settings, "GEMINI_API_KEY", None)
        if api_key and api_key != "not_set":
            client = AsyncOpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            target_model = mdl if "gemini" in mdl.lower() else "gemini-2.0-flash"
            return client, target_model, "google"
        else:
            logger.warning("GOOGLE_API_KEY or GEMINI_API_KEY is required in .env for Gemini model authentication. Falling back to OpenAI.")

    # Default: OpenAI
    return async_client, mdl, "openai"

async def transcribe_audio(
    audio_bytes: bytes, 
    content_type: str = "audio/webm",
    provider: str = None,
    model: str = None,
    session_id: str = None,
    language: str = None
) -> str:
    """
    Transcribes audio bytes recorded in the browser/telephony into text using the dynamically selected STT provider (Google, ElevenLabs, OpenAI).
    """
    stt_provider = provider
    stt_model = model
    survey_lang = language

    if session_id and (not stt_provider or not survey_lang):
        try:
            from app.core.db import get_collection, get_survey_config
            sessions_col = get_collection("survey_sessions")
            session = await sessions_col.find_one({"session_id": session_id})
            if session:
                survey_id = session.get("survey_id", "default")
                survey_config = await get_survey_config(survey_id)
                if survey_config:
                    stt_provider = stt_provider or survey_config.get("stt_provider")
                    stt_model = stt_model or survey_config.get("stt_model")
                    survey_lang = survey_lang or survey_config.get("language")
        except Exception as ex:
            logger.warning(f"Could not resolve survey STT config for session {session_id}: {ex}")

    stt_provider = (stt_provider or settings.STT_PROVIDER or "elevenlabs").lower().strip()
    stt_model = stt_model or settings.STT_MODEL

    ext = "webm"
    if "wav" in content_type:
        ext = "wav"
    elif "mp3" in content_type:
        ext = "mp3"
    elif "ogg" in content_type:
        ext = "ogg"
    elif "m4a" in content_type:
        ext = "m4a"
        
    if stt_provider in ("google", "google_stt", "google-stt"):
        return await transcribe_audio_google(audio_bytes, content_type=content_type, model=stt_model, language=survey_lang)
    elif stt_provider in ("deepgram", "deepgram_stt"):
        return await transcribe_audio_deepgram(audio_bytes, content_type=content_type, model=stt_model, language=survey_lang)
    elif stt_provider == "elevenlabs":
        logger.info(f"Initiating STT transcription via ElevenLabs ({stt_model}) for {len(audio_bytes)} bytes audio ({content_type})...")
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {
            "xi-api-key": settings.ELEVENLABS_API_KEY
        }
        files = {
            "file": (f"audio.{ext}", audio_bytes, content_type)
        }
        data = {
            "model_id": stt_model,
            "tag_audio_events": "false"
        }
        if survey_lang:
            data["language_code"] = survey_lang
            
        client = _get_stt_http_client()
        try:
            response = await client.post(url, headers=headers, files=files, data=data)
            if response.status_code != 200:
                logger.error(f"ElevenLabs STT API returned status {response.status_code}: {response.text}")
                raise RuntimeError(f"ElevenLabs STT failed: {response.text}")
            res_data = response.json()
            transcript = res_data.get("text", "").strip()
            logger.info(f"ElevenLabs transcript generated successfully: '{transcript}'")
            return transcript
        except Exception as e:
            logger.error(f"Error in ElevenLabs STT: {e}")
            logger.warning(f"Falling back to Google STT due to ElevenLabs error...")
            return await transcribe_audio_google(audio_bytes, content_type=content_type, model="default", language=survey_lang)
    else:
        buffer = io.BytesIO(audio_bytes)
        buffer.name = f"audio.{ext}"
        
        # Use a dynamic prompt based on the survey language
        lang_lower = (survey_lang or "hi").strip().lower()
        if lang_lower.startswith("en"):
            stt_prompt = "Hello, this is an English voice survey conversation."
        else:
            stt_prompt = "नमस्ते, यह एक हिंदी और इंग्लिश वॉइस सर्वे बातचीत है。"
            
        try:
            logger.info(f"Initiating STT transcription via Whisper ({stt_model}) for {len(audio_bytes)} bytes audio ({content_type})...")
            
            kwargs = {
                "model": stt_model,
                "file": buffer,
                "prompt": stt_prompt
            }
            if survey_lang:
                kwargs["language"] = survey_lang
                
            response = await async_client.audio.transcriptions.create(**kwargs)
            transcript = response.text.strip()
            logger.info(f"Whisper transcript generated successfully: '{transcript}'")
            return transcript
        except Exception as e:
            logger.error(f"Error in Whisper speech-to-text transcription: {e}")
            raise e

async def generate_response(
    history: List[Dict[str, Any]], 
    prev_extracted: Dict[str, Any], 
    skip_translation: bool = False,
    state_context: str = None,
    survey_id: str = "default",
    provider: str = None,
    model: str = None,
    survey_config: dict = None,
    last_filler_message: str = None
) -> Dict[str, Any]:
    """
    Generates the next survey response and extracts structured metrics in a single API call using Structured Outputs.
    Supports dynamic LLM providers (Google Gemini, DeepSeek, OpenAI, Grok) with automatic fallback.
    """
    import asyncio
    import json
    
    # 1. Initialize with defaults from settings
    assistant_name = settings.ASSISTANT_NAME
    raw_system_prompt = SYSTEM_PROMPT
    chat_model = model
    llm_provider = provider
    survey_steps = []
    
    # 2. Use pre-fetched survey_config if provided, otherwise fetch from MongoDB
    if not survey_config:
        from app.core.db import get_survey_config
        survey_config = await get_survey_config(survey_id)
    if survey_config:
        assistant_name = survey_config.get("assistant_name") or assistant_name
        raw_system_prompt = survey_config.get("system_prompt") or raw_system_prompt
        llm_provider = llm_provider or survey_config.get("llm_provider") or survey_config.get("chat_provider") or "openai"
        chat_model = chat_model or survey_config.get("chat_model") or survey_config.get("llm_model") or settings.CHAT_MODEL
        survey_steps = survey_config.get("survey_steps", [])
        survey_language = survey_config.get("language") or "hi"
        prompts = survey_config.get("prompts") or {}
        agent_description = prompts.get("description")
    else:
        survey_language = "hi"
        llm_provider = llm_provider or "openai"
        agent_description = None
        
    agent_description_str = ""
    if agent_description:
        agent_description_str = f"5. AGENT PERSONA/DESCRIPTION: {agent_description}\nIf the user asks about your personal information, name, or who you represent, you must reply based on this description."
    
    # 2. Dynamically build the prompt context & output format description if using the template SYSTEM_PROMPT
    if "survey_steps_desc" in raw_system_prompt:
        steps_desc = ""
        extracted_fields = {}
        for idx, step in enumerate(survey_steps, 1):
            field_name = step["field"]
            steps_desc += f"{idx}. Step '{step['id']}' (field: '{field_name}'): {step['instruction']}\n"
            
            type_str = step.get("type", "str")
            if type_str == "int":
                field_val_repr = "integer or null"
            elif type_str == "list":
                field_val_repr = "list of strings or null"
            else:
                field_val_repr = "string or null"
            extracted_fields[f'"{field_name}"'] = field_val_repr
        
        steps_desc += f"{len(survey_steps) + 1}. Conclude the survey warmly, thank them for their time, and say goodbye."
        
        extracted_fields_lines = []
        for field_name, val_repr in extracted_fields.items():
            extracted_fields_lines.append(f"    {field_name}: {val_repr}")
        extracted_fields_desc = ",\n".join(extracted_fields_lines)
        
        system_content = raw_system_prompt.format(
            assistant_name=assistant_name,
            survey_steps_desc=steps_desc,
            extracted_fields_desc=extracted_fields_desc,
            agent_description=agent_description_str
        )
    else:
        try:
            system_content = raw_system_prompt.format(assistant_name=assistant_name, agent_description=agent_description_str)
        except Exception:
            system_content = raw_system_prompt
            if agent_description_str:
                system_content = f"{agent_description_str}\n\n{system_content}"
            
        # Append survey questions list to custom system prompt if not present
        steps_summary = "\n\nSURVEY QUESTIONS AND STEPS LIST:\n"
        for idx, step in enumerate(survey_steps, 1):
            q_text = step.get("question") or step.get("description") or ""
            steps_summary += f"{idx}. Field ID '{step['field']}': Question: '{q_text}'\n"
        system_content += steps_summary
        
    # Resolve voice details and voice gender dynamically
    voice_id = settings.ELEVENLABS_VOICE_ID
    voice_name = settings.TTS_VOICE
    if survey_config:
        voice_id = survey_config.get("tts_voice_id") or voice_id
        voice_name = survey_config.get("tts_voice_name") or voice_name
        
    voice_gender = None
    if voice_id:
        from app.core.db import get_collection
        from bson import ObjectId
        voices_col = get_collection("voices")
        voice_doc = None
        
        if ObjectId.is_valid(str(voice_id)):
            voice_doc = await voices_col.find_one({"_id": ObjectId(str(voice_id))})
        if not voice_doc:
            voice_doc = await voices_col.find_one({"_id": str(voice_id)})
        if not voice_doc:
            voice_doc = await voices_col.find_one({"voiceId": str(voice_id)})
            
        if voice_doc and voice_doc.get("gender"):
            voice_gender = str(voice_doc.get("gender")).lower()
            
    if not voice_gender:
        voice_gender = determine_gender_from_voice(voice_name, voice_id)
    
    if voice_gender == "female":
        gender_instruction = (
            "GRAMMATICAL GENDER (CRITICAL): You MUST speak in the first-person female grammatical gender in Hindi/Hinglish. "
            "Use female verb inflections ending in 'आई', 'ई', 'रही हूँ', 'गई', 'करूँगी' (e.g. say 'समझ गई' or 'मैं समझ गई' "
            "instead of 'समझ गया', and 'सुन रही हूँ' instead of 'सुन रहा हूँ'). Never use male grammatical endings."
        )
    else:
        gender_instruction = (
            "GRAMMATICAL GENDER (CRITICAL): You MUST speak in the first-person male grammatical gender in Hindi/Hinglish. "
            "Use male verb inflections ending in 'आ', 'रहा हूँ', 'गया', 'करूँगा' (e.g. say 'समझ गया' or 'मैं समझ गया')."
        )
        
    system_content = gender_instruction + "\n\n" + system_content
            
    # 3. Dynamically build the required output JSON schema description with exact question mappings
    extracted_fields = {}
    for idx, step in enumerate(survey_steps, 1):
        field_name = step["field"]
        q_text = step.get("question") or step.get("description") or ""
        type_str = step.get("type", "str")
        options = step.get("options") or []

        if options:
            opts_str = ", ".join([f"'{o.get('label')}' (value: '{o.get('value')}')" if isinstance(o, dict) else f"'{o}'" for o in options])
            field_val_repr = f"string or null (Answer to Question {idx}: '{q_text}'. MUST MATCH ONE VALUE IN: [{opts_str}]. If out of context or invalid, set to null!)"
        elif type_str == "int":
            field_val_repr = f"integer or null (Answer to Question {idx}: '{q_text}')"
        elif type_str == "list":
            field_val_repr = f"list of strings or null (Answer to Question {idx}: '{q_text}')"
        else:
            field_val_repr = f"string or null (Answer to Question {idx}: '{q_text}')"

        extracted_fields[f'"{field_name}"'] = field_val_repr
        
    extracted_fields_lines = []
    for field_name, val_repr in extracted_fields.items():
        extracted_fields_lines.append(f"    {field_name}: {val_repr}")
    extracted_fields_desc = ",\n".join(extracted_fields_lines)
    
    # Map language codes to human-readable names
    LANGUAGE_MAP = {
        "hi": "Hindi", "en": "English", "ta": "Tamil", "te": "Telugu",
        "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
        "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu", "or": "Odia",
    }
    language_name = LANGUAGE_MAP.get(survey_language, survey_language)
    
    filler_constraint = ""
    if last_filler_message:
        filler_constraint = f"Do NOT reuse the previous filler: '{last_filler_message}'. "
        
    output_format_instruction = (
        f"OUTPUT FORMAT REQUIREMENT:\n"
        f"You MUST return a valid JSON object containing exactly the following two keys:\n"
        f"1. \"conversational_response\": A string containing 1 to 2 short sentences in {language_name} to continue the survey conversation. ALWAYS respond in {language_name}.\n"
        f"   CRITICAL CONTEXT RULE: The conversational_response acts as a brief filler acknowledging the user's PREVIOUS answer. If the user gives a short factual answer (like 'Male' or a number), do NOT use generic empathetic fillers like 'I understand' or 'samajh sakti hu'. Instead, use a very brief acknowledgment like 'Okay', 'Got it', or no filler at all. {filler_constraint}\n"
        f"2. \"extracted_data\": A JSON object containing the extracted survey data with the following fields:\n"
        f"{{\n{extracted_fields_desc}\n}}\n"
    )
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "system", "content": output_format_instruction}
    ]
    
    # Removed state_context insertion from here to move it to the end
    
    # Map database turn models to OpenAI message formats
    for turn in history:
        role = "assistant" if turn["speaker"] == "SYSTEM" else "user"
        messages.append({"role": role, "content": turn["text_content"]})
        
    if state_context:
        messages.append({"role": "system", "content": state_context})
        
    # Append the previous extraction state at the end so the LLM has the baseline context
    messages.append({
        "role": "system",
        "content": f"PREVIOUS EXTRACTED DATA:\n{json.dumps(prev_extracted, ensure_ascii=False)}"
    })
    
    # Enforce JSON formatting constraint to comply with OpenAI JSON mode API requirements
    messages.append({
        "role": "system",
        "content": "Return the output strictly in JSON format matching the schema."
    })
        
    # Primary model and fallback model options
    target_client, target_model_name, target_provider = get_llm_client_and_model(llm_provider, chat_model)
    models_to_try = [(target_client, target_model_name, target_provider)]
    fallback_model = getattr(settings, "CHAT_MODEL", "gpt-5.6-luna")
    if target_provider != "openai":
        models_to_try.append((async_client, fallback_model, "openai"))
    elif target_model_name != fallback_model:
        models_to_try.append((async_client, fallback_model, "openai"))

    last_error = None
    
    for client_inst, model_name, provider_name in models_to_try:
        try:
            logger.info(f"Generating unified response and extraction via {provider_name}:{model_name} (JSON mode)...")
            create_kwargs = {
                "model": model_name,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "max_completion_tokens": 1024,
                "timeout": 12.0
            }
            if not is_no_temperature_model(model_name):
                create_kwargs["temperature"] = 0.0

            response = await client_inst.chat.completions.create(**create_kwargs)
            raw_content = response.choices[0].message.content.strip()
            parsed = json.loads(raw_content)
            
            ai_response = parsed.get("conversational_response", "").strip()
            raw_extracted = parsed.get("extracted_data", {})
            if not isinstance(raw_extracted, dict):
                raw_extracted = {}
                
            # Populate fields dynamically based on the current survey steps config
            extracted_dict = {}
            for step in survey_steps:
                field_name = step["field"]
                val = raw_extracted.get(field_name)
                if isinstance(val, str) and val.strip().lower() in ["null", "none", "n/a", "undefined"]:
                    val = None

                # Strict Option Validation
                options = step.get("options") or []
                if val is not None and options:
                    if isinstance(val, str) and val.startswith("FALLBACK:"):
                        # Bypass strict validation after 2 failed attempts
                        val = val.replace("FALLBACK:", "", 1).strip()
                    else:
                        val_str = str(val).strip().lower()
                        matched_val = None
                        for opt in options:
                            if isinstance(opt, dict):
                                opt_lbl = str(opt.get("label", "")).strip().lower()
                                opt_val = str(opt.get("value", "")).strip().lower()
                                if val_str == opt_val or val_str == opt_lbl:
                                    matched_val = opt.get("value") or opt.get("label")
                                    break
                            elif isinstance(opt, str):
                                if val_str == opt.strip().lower():
                                    matched_val = opt
                                    break
                        val = matched_val

                extracted_dict[field_name] = val
            
            # Translate extracted data only if caller didn't request to skip
            if not skip_translation:
                # Resolve session_id from calling context if possible (stored in history turns if any)
                session_id = history[0].get("session_id") if (history and "session_id" in history[0]) else None
                extracted_dict = await translate_extracted_data(extracted_dict, session_id=session_id)
            
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            
            logger.info(f"Unified generation via {provider_name}:{model_name} successful. Response: '{ai_response}', Extracted: {extracted_dict}")
            return {
                "conversational_response": ai_response,
                "extracted_data": extracted_dict,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "model": model_name,
                    "provider": provider_name
                }
            }
        except Exception as e:
            last_error = e
            logger.warning(f"Failed unified generation using provider/model {provider_name}:{model_name}: {e}. Retrying with fallback...")
            await asyncio.sleep(0.1)
            
    logger.error(f"All chat completion models failed. Last error: {last_error}")
    raise last_error

async def extract_user_intent_structured(
    user_transcript: str, 
    current_step: dict, 
    provider: str = None, 
    model: str = None, 
    survey_config: dict = None
) -> dict:
    """
    Pass 1: Ultra-fast state extraction using JSON mode.
    Takes only the user's transcript and the current active step context.
    Returns: {"state": "valid" | "unclear", "value": "extracted value or null"}
    """
    llm_provider = provider or (survey_config.get("llm_provider") if survey_config else "openai")
    chat_model = model or (survey_config.get("chat_model") if survey_config else settings.CHAT_MODEL)
    
    sys_prompt = (
        "You are a strict data extraction engine for a voice survey.\n"
        "Your ONLY job is to evaluate if the user's spoken transcript is a valid answer for the CURRENT STEP.\n\n"
        "RULES:\n"
        "1. If the user answers the question, or gives a contextual synonym, state='valid'. Extract the formal value.\n"
        "2. If the user's transcript matches ANY of the positive/negative synonyms in the INSTRUCTIONS & MAPPING RULES, you MUST forcefully extract that value and set state='valid', even if it sounds like they are just giving consent to ask questions.\n"
        "3. If the user is completely unclear or off-topic and doesn't match any mapping rules, state='unclear', value='null'.\n\n"
        "OUTPUT FORMAT (Strict JSON):\n"
        '{"state": "valid" | "unclear", "value": "<extracted string>"}'
    )
    
    q_text = current_step.get("question", "")
    q_instruction = current_step.get("instruction", "")
    q_options = [opt.get("label", opt) if isinstance(opt, dict) else opt for opt in current_step.get("options", [])]
    
    user_prompt = f"CURRENT STEP: {q_text}\n"
    if q_instruction:
        user_prompt += f"INSTRUCTIONS & MAPPING RULES: {q_instruction}\n"
    if q_options:
        user_prompt += f"ALLOWED OPTIONS: {', '.join(q_options)}\n"
    user_prompt += f"\nUSER TRANSCRIPT: {user_transcript}"
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    target_client, target_model_name, target_provider = get_llm_client_and_model(llm_provider, chat_model)
    
    try:
        response = await target_client.chat.completions.create(
            model=target_model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=50,
            timeout=5.0
        )
        import json
        result = json.loads(response.choices[0].message.content)
        return {
            "state": str(result.get("state", "unclear")).lower(),
            "value": str(result.get("value", "null"))
        }
    except Exception as e:
        logger.error(f"Failed to extract structured intent: {e}")
        return {"state": "unclear", "value": "null"}

async def evaluate_turn_state(
    current_step: Dict[str, Any],
    user_transcript: str,
    provider: str = None,
    model: str = None,
    survey_config: dict = None
) -> Dict[str, Any]:
    """
    Pass 1 of the two-pass architecture.
    Evaluates the user's answer against the CURRENT STEP to extract state and value.
    """
    import json
    
    llm_provider = provider or "groq"
    chat_model = model or "groq/compound-mini"
    
    system_prompt = f"""You are a survey data extraction evaluator.
Analyze the user's response to the current survey question.
Current Question: '{current_step.get("question")}'
Instructions for extraction: '{current_step.get("instruction")}'

Determine if the user's response provides a valid answer (even if contextual or synonymous) or if it's completely unclear/off-topic.
If valid, extract the value according to the instructions. If unclear, set state to "unclear" and extracted_value to null.
If the step has predefined options: {current_step.get("options", "None")}, you MUST map the user's response to the exact value or label of one of those options. For example, if it's a Yes/No question and the user says "हाँ" (Yes), output "Yes".

Output strictly as a JSON object:
{{
    "state": "valid" or "unclear",
    "extracted_value": "extracted string, integer, or null"
}}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User Response: {user_transcript}"}
    ]
    
    target_client, target_model_name, target_provider = get_llm_client_and_model(llm_provider, chat_model)
    
    logger.info(f"Initiating Pass 1 evaluation via {target_provider}:{target_model_name}...")
    
    try:
        create_kwargs = {
            "model": target_model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 256
        }
        if not is_no_temperature_model(target_model_name):
            create_kwargs["temperature"] = 0.0
            
        create_kwargs["timeout"] = 5.0

        response = await target_client.chat.completions.create(**create_kwargs)
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        return parsed
    except Exception as e:
        logger.error(f"Error in evaluate_turn_state: {e}")
        return {"state": "unclear", "extracted_value": None}


async def generate_response_stream(
    history: list,
    state_context: str,
    survey_id: str = "default",
    provider: str = None,
    model: str = None,
    survey_config: dict = None,
    last_filler_message: str = None
):
    import asyncio
    import json
    """
    Streams the plain-text conversational response from the LLM.
    Does not use JSON mode or extract fields, achieving minimal time-to-first-token.
    """
    
    # 1. Initialize defaults
    assistant_name = settings.ASSISTANT_NAME
    raw_system_prompt = SYSTEM_PROMPT
    chat_model = model
    llm_provider = provider
    
    # 2. Use survey_config
    if survey_config:
        assistant_name = survey_config.get("assistant_name") or assistant_name
        raw_system_prompt = survey_config.get("system_prompt") or raw_system_prompt
        llm_provider = llm_provider or survey_config.get("llm_provider") or survey_config.get("chat_provider") or "openai"
        chat_model = chat_model or survey_config.get("chat_model") or survey_config.get("llm_model") or settings.CHAT_MODEL
        survey_language = survey_config.get("language") or "hi"
        farewell_msg = survey_config.get("farewell_message") or survey_config.get("farewell") or "Thank you for your time. Goodbye."
        prompts = survey_config.get("prompts") or {}
        agent_description = prompts.get("description")
    else:
        survey_language = "hi"
        llm_provider = llm_provider or "openai"
        farewell_msg = "Thank you for your time. Goodbye."
        agent_description = None
        
    agent_desc_instruction = ""
    if agent_description:
        agent_desc_instruction = f"10. AGENT PERSONA/DESCRIPTION: {agent_description}\nIf the user asks about your personal information, name, or who you represent, you MUST reply based on this description."
        
    # Map language codes to human-readable names
    LANGUAGE_MAP = {
        "hi": "Hindi", "en": "English", "ta": "Tamil", "te": "Telugu",
        "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
        "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu", "or": "Odia",
    }
    language_name = LANGUAGE_MAP.get(survey_language, survey_language)
    
    # Resolve voice details and voice gender dynamically
    voice_id = settings.ELEVENLABS_VOICE_ID
    voice_name = settings.TTS_VOICE
    if survey_config:
        voice_id = survey_config.get("tts_voice_id") or voice_id
        voice_name = survey_config.get("tts_voice_name") or voice_name
        
    voice_gender = determine_gender_from_voice(voice_name, voice_id)
    
    if voice_gender == "female":
        gender_instruction = (
            "GRAMMATICAL GENDER (CRITICAL): You are a FEMALE voice assistant. You MUST use female verb inflections for yourself in Hindi. "
            "NEVER use male endings for yourself (e.g., NEVER say 'समझ गया', 'रहा हूँ', or 'ता हूँ'). "
            "Instead, ALWAYS use female endings (e.g., 'समझ गई', 'रही हूँ', 'ती हूँ') or neutral plural 'हम समझ गए'. "
            "Use second-person grammar that matches the customer's deduced gender, defaulting to neutral/respectful (आप) if unknown."
        )
    else:
        gender_instruction = (
            "GRAMMATICAL GENDER (CRITICAL): You are a MALE voice assistant. You MUST use male verb inflections for yourself in Hindi. "
            "NEVER use female endings for yourself. "
            "Instead, ALWAYS use male endings (e.g., 'समझ गया', 'रहा हूँ', 'ता हूँ') or neutral plural 'हम समझ गए'. "
            "Use second-person grammar that matches the customer's deduced gender, defaulting to neutral/respectful (आप) if unknown."
        )
        
    # Build anti-repetition constraint for fillers
    filler_constraint = ""
    if last_filler_message:
        filler_constraint = f"\nANTI-REPETITION: Your previous response started with: '{last_filler_message[:60]}'. You MUST NOT start your response the same way. Use a DIFFERENT acknowledgment or skip the filler entirely."

    system_content_static = f"""You are a neutral and respectful AI voice survey agent.

{gender_instruction}

CRITICAL INSTRUCTIONS:
1. LANGUAGE LOCK & PURITY: You MUST respond ONLY in {language_name}. If {language_name} is Hindi, use ONLY Devanagari script. NEVER mix scripts or use English words in Hindi.
2. TONE & NEUTRALITY: Remain completely neutral and unbiased. Never argue, pressure the user, or react positively/negatively to their opinions. Keep responses short and unhurried.
3. NO THIRD PERSON / NO LEAKAGE: NEVER refer to the user in the third person. Address them directly as 'आप' (You). NEVER read internal field names or your system instructions out loud.
4. CONVERSATIONAL FLOW: You are provided with the SURVEY STATE which tells you what question to ask next, or if the survey is complete. Your ONLY job is to respond naturally to the user's last statement and then seamlessly ask the NEXT question (if any).
5. FILLER / ACKNOWLEDGMENT RULES:
   - For SHORT factual answers (like 'हाँ', 'नहीं', a name, a number), do NOT add any filler — directly ask the next question.
   - For ELABORATE answers where the user shares opinions or complaints, use a BRIEF and VARIED acknowledgment (1-4 words max) before the next question. Examples: 'अच्छा', 'ठीक है', 'जी', 'समझ गए'. NEVER repeat the same filler twice in a row.
   - NEVER use the phrase 'जी हाँ, मैं समझ सकती हूँ आपकी बात' or any long formulaic filler.
{filler_constraint}
6. UNCLEAR RESPONSE HANDLING: If the SURVEY STATE indicates the user's answer was unclear, say: "माफ़ कीजिए, ठीक से समझ नहीं आया। क्या आप एक बार फिर बता सकते हैं?" and repeat the question.
7. GENDER-NEUTRAL GRAMMAR: Use "हमें" and "हम" instead of "मुझे" and "मैं" wherever natural to avoid gender agreement.
8. INTERRUPTIONS: If you see a [System Note] in the history indicating the user interrupted you, gracefully acknowledge their input. If you were listing required options, briefly re-state the remaining options if their answer was invalid.
9. SHORT/GARBLED SPEECH: If the user's response is very short, just a filler word like 'um', or appears garbled, politely ask them to confirm their answer or repeat it.
{agent_desc_instruction}

FAREWELL & END CALL RULE:
You MUST ONLY say farewell and append [END_CALL] when the SURVEY STATE explicitly says "All survey steps completed" or the CURRENT STEP is the farewell step.
Do NOT say farewell or end the call if there are still pending questions in the SURVEY STATE.
When it IS time to say farewell, output the exact farewell message: "{farewell_msg}" and append [END_CALL] at the very end.

OUTPUT FORMAT:
Output ONLY your natural conversational spoken response in {language_name}. Do NOT output any internal JSON, brackets, or state metadata. Speak directly to the user.
"""
    
    messages = [
        {"role": "system", "content": system_content_static}
    ]
    
    # Add the latest user response and system response for context
    latest_user_text = ""
    latest_system_text = ""
    for turn in reversed(history):
        if turn["speaker"] == "CUSTOMER" and not latest_user_text:
            latest_user_text = turn["text_content"]
        elif turn["speaker"] == "SYSTEM" and not latest_system_text:
            latest_system_text = turn["text_content"]
            
        if latest_user_text and latest_system_text:
            break
            
    if latest_system_text:
        messages.append({"role": "assistant", "content": f"What You Just Said: {latest_system_text}"})
        
    if latest_user_text:
        messages.append({"role": "user", "content": f"Latest User Response: {latest_user_text}"})
        
    if state_context:
        messages.append({"role": "system", "content": f"SURVEY STATE AND WORKFLOW:\n{state_context}"})
        
    # Resolve target client
    target_client, target_model_name, target_provider = get_llm_client_and_model(llm_provider, chat_model)
    
    logger.info(f"Initiating streaming unified completion via {target_provider}:{target_model_name}...")
    import json
    logger.info(f"--- LLM PROMPT PAYLOAD ---\n{json.dumps(messages, indent=2, ensure_ascii=False)}\n--------------------------")
    
    create_kwargs = {
        "model": target_model_name,
        "messages": messages,
        "stream": True,
        "max_completion_tokens": 1024,
        "timeout": 10.0
    }
    if target_provider == "openai":
        # Request stream options to get exact usage tokens in the final chunk
        create_kwargs["stream_options"] = {"include_usage": True}
        
    if not is_no_temperature_model(target_model_name):
        create_kwargs["temperature"] = 0.5
        
    create_kwargs["timeout"] = 5.0
        
    response = await target_client.chat.completions.create(**create_kwargs)
    return response, target_model_name, target_provider

import hashlib

import time

# In-memory TTS audio cache (md5(text:voice) -> bytes)
TTS_PCM_CACHE = {}

_google_token_lock = None
_google_access_token = {"token": None, "expires_at": 0}

def _get_google_token_lock():
    global _google_token_lock
    if _google_token_lock is None:
        _google_token_lock = asyncio.Lock()
    return _google_token_lock

def get_google_access_token_sync(force_refresh: bool = False) -> str:
    """
    Synchronously returns a valid Google Access Token using the Google Service Account JSON key.
    """
    now = time.time()
    if not force_refresh and _google_access_token["token"] and now < _google_access_token["expires_at"]:
        return _google_access_token["token"]

    sa_file = getattr(settings, "GOOGLE_APPLICATION_CREDENTIALS", "voice-survey-505211-674591af47e3.json")
    if not sa_file or not os.path.exists(sa_file):
        if os.path.exists("voice-survey-505211-674591af47e3.json"):
            sa_file = "voice-survey-505211-674591af47e3.json"

    if not sa_file or not os.path.exists(sa_file):
        logger.error("Google Service Account JSON key file not found!")
        raise RuntimeError("Google Service Account JSON key file not found.")

    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request as GoogleAuthRequest

        logger.info(f"Generating Google Access Token using Service Account JSON ({sa_file})...")
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        creds = service_account.Credentials.from_service_account_file(sa_file, scopes=scopes)
        req = GoogleAuthRequest()
        creds.refresh(req)
        token = creds.token

        _google_access_token["token"] = token
        _google_access_token["expires_at"] = time.time() + 3300
        logger.info(f"Successfully generated Google Access Token via Service Account ({creds.service_account_email}).")
        return token
    except Exception as sa_err:
        logger.error(f"Service Account token generation failed: {sa_err}")
        raise RuntimeError(f"Google Service Account token generation failed: {sa_err}")

async def get_google_access_token(force_refresh: bool = False) -> str:
    """
    Returns a valid Google Access Token using the Google Service Account JSON key.
    Uses async lock and proactive expiration buffer to prevent race conditions during concurrent calls.
    """
    now = time.time()
    if not force_refresh and _google_access_token["token"] and now < _google_access_token["expires_at"]:
        return _google_access_token["token"]

    async with _get_google_token_lock():
        return get_google_access_token_sync(force_refresh=force_refresh)

GOOGLE_LANG_MAP = {
    "hi": "hi-IN",
    "en": "en-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "or": "or-IN",
    "pa": "pa-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "as": "as-IN",
}

async def synthesize_speech(
    text: str, 
    provider: str = None, 
    voice: str = None, 
    model_id: str = None, 
    session_id: str = None,
    language: str = None,
    speed: float = None,
    survey_config: dict = None
) -> bytes:
    """
    Converts a response text into speech audio bytes using ElevenLabs, Sarvam AI, or Google TTS, with 0-cost caching.
    """
    clean_text = text.strip()
    survey_speed = None

    if session_id and (not provider or not voice or not language or speed is None) and not survey_config:
        try:
            from app.core.db import get_collection, get_survey_config
            sessions_col = get_collection("survey_sessions")
            logger.info(f"[DB-READ] Reading survey_sessions from MongoDB in synthesize_speech for session_id: {session_id}")
            session = await sessions_col.find_one({"session_id": session_id})
            if session:
                survey_id = session.get("survey_id", "default")
                logger.info(f"[DB-READ] Reading survey_config from MongoDB in synthesize_speech for survey_id: {survey_id}")
                survey_config = await get_survey_config(survey_id)
        except Exception as ex:
            logger.warning(f"Could not resolve TTS config for session {session_id}: {ex}")

    if survey_config:
        provider = provider or survey_config.get("tts_provider")
        voice = voice or survey_config.get("tts_voice_id")
        model_id = model_id or survey_config.get("tts_model_id")
        language = language or survey_config.get("language")
        persona_obj = survey_config.get("persona") or {}
        tts_obj = persona_obj.get("tts") if isinstance(persona_obj, dict) else {}
        if not isinstance(tts_obj, dict): tts_obj = {}
        survey_speed = (
            survey_config.get("tts_speed")
            or tts_obj.get("tts_speed")
            or tts_obj.get("speed")
            or (persona_obj.get("tts_speed") if isinstance(persona_obj, dict) else None)
            or survey_config.get("speech_speed")
            or survey_config.get("speed")
        )

    tts_provider = (provider or settings.TTS_PROVIDER or "openai").lower().strip()
    
    # Resolve default voice for provider if voice is not set
    if not voice:
        if tts_provider in ("google", "google_tts"):
            voice = "hi-IN-Chirp3-HD-Achernar"
        elif tts_provider in ("openai", "openai_tts"):
            voice = getattr(settings, "TTS_VOICE", "nova")
        elif tts_provider == "sarvam":
            voice = "meera"
        else:
            voice = getattr(settings, "ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

    tts_voice_id = voice
    tts_speed = speed if speed is not None else (survey_speed if survey_speed is not None else getattr(settings, "TTS_SPEECH_SPEED", 1.0))
    cache_key = hashlib.md5(f"mp3:{tts_provider}:{clean_text.lower()}:{tts_voice_id}:{tts_speed}:{language}".encode()).hexdigest()

    if cache_key in TTS_PCM_CACHE:
        logger.info(f"Using in-memory cached TTS audio for text: '{clean_text[:40]}...' (0ms delay, $0.00 cost)")
        return TTS_PCM_CACHE[cache_key]

    tts_provider = provider or settings.TTS_PROVIDER
    
    if tts_provider in ("google", "google_tts"):
        token = await get_google_access_token()
        url = "https://texttospeech.googleapis.com/v1/text:synthesize"
        # Resolve language code from survey language parameter ('hi', 'en', 'bn', 'gu', etc.)
        lang_code = "hi-IN"
        if language:
            lang_lower = language.strip().lower()
            lang_code = GOOGLE_LANG_MAP.get(lang_lower, GOOGLE_LANG_MAP.get(lang_lower[:2], lang_lower if "-" in lang_lower else f"{lang_lower}-IN"))

        voice_name = voice or "hi-IN-Chirp3-HD-Achernar"
        if ":" in voice_name:
            lang_code, voice_name = voice_name.split(":", 1)
        elif "-" in voice_name and not language:
            parts = voice_name.split("-")
            if len(parts) >= 2:
                lang_code = f"{parts[0]}-{parts[1]}"

        data = {
            "input": {
                "text": clean_text
            },
            "voice": {
                "languageCode": lang_code,
                "name": voice_name
            },
            "audioConfig": {
                "audioEncoding": "MP3"
            }
        }
        logger.info(f"Synthesizing speech via Google TTS (Voice: {voice_name}, Lang: {lang_code})...")
        client = _get_tts_http_client()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        response = await client.post(url, headers=headers, json=data)
            
        # Retry with force token refresh if 401 Unauthorized happens
        if response.status_code == 401:
            logger.warning("Google TTS returned 401. Force refreshing access token and retrying...")
            token = await get_google_access_token(force_refresh=True)
            headers["Authorization"] = f"Bearer {token}"
            response = await client.post(url, headers=headers, json=data)

        if response.status_code == 200:
            res_data = response.json()
            audio_b64 = res_data.get("audioContent", "")
            if audio_b64:
                import base64
                audio_content = base64.b64decode(audio_b64)
                TTS_PCM_CACHE[cache_key] = audio_content
                return audio_content
        logger.error(f"Google TTS API failed with status {response.status_code}: {response.text}")
        raise RuntimeError(f"Google TTS failed status {response.status_code}: {response.text}")

    if tts_provider in ("openai", "openai_tts"):
        openai_voice = voice or getattr(settings, "TTS_VOICE", "nova")
        if len(openai_voice) > 15:
            openai_voice = "nova"
        openai_model = model_id if (model_id and "tts-1" in str(model_id)) else getattr(settings, "TTS_MODEL", "tts-1")
        try:
            logger.info(f"Synthesizing speech via OpenAI TTS (Voice: {openai_voice}, Model: {openai_model})...")
            response = await async_client.audio.speech.create(
                model=openai_model,
                voice=openai_voice,
                input=clean_text,
                response_format="mp3"
            )
            audio_content = response.content
            TTS_PCM_CACHE[cache_key] = audio_content
            return audio_content
        except Exception as e:
            logger.error(f"OpenAI TTS synthesis request failed: {e}")
            raise e

    if tts_provider == "sarvam":
        url = "https://api.sarvam.ai/text-to-speech"
        headers = {
            "api-subscription-key": getattr(settings, "SARVAM_API_KEY", "not_set"),
            "Content-Type": "application/json"
        }
        data = {
            "inputs": [clean_text],
            "target_language_code": "hi-IN",
            "speaker": "meera",
            "model": "bulbul:v1"
        }
        logger.info(f"Synthesizing speech via Sarvam AI (Voice: meera)...")
        client = _get_tts_http_client()
        response = await client.post(url, headers=headers, json=data)
        if response.status_code == 200:
            res_data = response.json()
            audios = res_data.get("audios", [])
            if audios:
                import base64
                audio_content = base64.b64decode(audios[0])
                TTS_PCM_CACHE[cache_key] = audio_content
                return audio_content
        raise RuntimeError(f"Sarvam AI TTS failed with status {response.status_code}: {response.text}")

    # Fallback to ElevenLabs TTS
    tts_model_id = model_id if (model_id and "eleven" in str(model_id)) else settings.ELEVENLABS_MODEL_ID
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{tts_voice_id}"
    headers = {
        "xi-api-key": settings.ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": clean_text,
        "model_id": tts_model_id,
        "voice_settings": {
            "stability": 0.65,
            "similarity_boost": 0.75,
            "speed": float(tts_speed)
        }
    }
    try:
        logger.info(f"Synthesizing response text to speech via ElevenLabs (Voice ID: {tts_voice_id})...")
        client = _get_tts_http_client()
        response = await client.post(url, headers=headers, json=data)
        if response.status_code != 200:
            logger.error(f"ElevenLabs API returned error {response.status_code}: {response.text}")
            raise RuntimeError(f"ElevenLabs TTS failed: {response.text}")
        audio_content = response.content
        TTS_PCM_CACHE[cache_key] = audio_content
        logger.info(f"ElevenLabs speech synthesis successful. Size: {len(audio_content)} bytes")
        return audio_content
    except Exception as e:
        err_msg = str(e) if str(e) else repr(e)
        logger.error(f"ElevenLabs synthesis request failed: {err_msg}")
        raise RuntimeError(f"ElevenLabs synthesis failed: {err_msg}") from e

async def synthesize_speech_pcm(
    text: str, 
    provider: str = None,
    voice: str = None, 
    model_id: str = None, 
    session_id: str = None,
    language: str = None,
    speed: float = None,
    survey_config: dict = None
) -> bytes:
    """
    Synthesizes speech to raw 24kHz 16-bit mono PCM bytes with 0-cost caching.
    """
    clean_text = text.strip()
    survey_speed = None

    if session_id and (not provider or not voice or not language) and not survey_config:
        try:
            from app.core.db import get_collection, get_survey_config
            sessions_col = get_collection("survey_sessions")
            logger.info(f"[DB-READ] Reading survey_sessions from MongoDB in synthesize_speech_pcm for session_id: {session_id}")
            session = await sessions_col.find_one({"session_id": session_id})
            if session:
                survey_id = session.get("survey_id", "default")
                logger.info(f"[DB-READ] Reading survey_config from MongoDB in synthesize_speech_pcm for survey_id: {survey_id}")
                survey_config = await get_survey_config(survey_id)
        except Exception as ex:
            logger.warning(f"Could not resolve TTS config for session {session_id}: {ex}")

    if survey_config:
        provider = provider or survey_config.get("tts_provider")
        voice = voice or survey_config.get("tts_voice_id")
        model_id = model_id or survey_config.get("tts_model_id")
        language = language or survey_config.get("language")
        persona_obj = survey_config.get("persona") or {}
        tts_obj = persona_obj.get("tts") if isinstance(persona_obj, dict) else {}
        if not isinstance(tts_obj, dict): tts_obj = {}
        survey_speed = (
            survey_config.get("tts_speed")
            or tts_obj.get("tts_speed")
            or tts_obj.get("speed")
            or (persona_obj.get("tts_speed") if isinstance(persona_obj, dict) else None)
            or survey_config.get("speech_speed")
            or survey_config.get("speed")
        )

    tts_provider = (provider or settings.TTS_PROVIDER or "openai").lower().strip()

    # Resolve default voice for provider if voice is not set
    if not voice:
        if tts_provider in ("google", "google_tts"):
            voice = "hi-IN-Chirp3-HD-Achernar"
        elif tts_provider in ("openai", "openai_tts"):
            voice = getattr(settings, "TTS_VOICE", "nova")
        elif tts_provider == "sarvam":
            voice = "meera"
        else:
            voice = getattr(settings, "ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

    tts_voice_id = voice
    tts_speed = speed if speed is not None else (survey_speed if survey_speed is not None else getattr(settings, "TTS_SPEECH_SPEED", 1.0))
    cache_key = hashlib.md5(f"pcm:{tts_provider}:{clean_text.lower()}:{tts_voice_id}:{tts_speed}:{language}".encode()).hexdigest()

    if cache_key in TTS_PCM_CACHE:
        logger.info(f"Using in-memory cached PCM audio for text: '{clean_text[:40]}...' (0ms delay, $0.00 cost)")
        return TTS_PCM_CACHE[cache_key]

    if tts_provider in ("google", "google_tts"):
        token = await get_google_access_token()
        url = "https://texttospeech.googleapis.com/v1/text:synthesize"
        lang_code = "hi-IN"
        if language:
            lang_lower = language.strip().lower()
            lang_code = GOOGLE_LANG_MAP.get(lang_lower, GOOGLE_LANG_MAP.get(lang_lower[:2], lang_lower if "-" in lang_lower else f"{lang_lower}-IN"))

        voice_name = voice or "hi-IN-Chirp3-HD-Achernar"
        if ":" in voice_name:
            lang_code, voice_name = voice_name.split(":", 1)
        elif "-" in voice_name and not language:
            parts = voice_name.split("-")
            if len(parts) >= 2:
                lang_code = f"{parts[0]}-{parts[1]}"

        data = {
            "input": {
                "text": clean_text
            },
            "voice": {
                "languageCode": lang_code,
                "name": voice_name
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": 24000
            }
        }
        logger.info(f"Synthesizing PCM speech via Google TTS (Voice: {voice_name}, Lang: {lang_code})...")
        client = _get_tts_http_client()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        response = await client.post(url, headers=headers, json=data)

        # Retry with force token refresh if 401 Unauthorized happens
        if response.status_code == 401:
            logger.warning("Google TTS PCM returned 401. Force refreshing access token and retrying...")
            token = await get_google_access_token(force_refresh=True)
            headers["Authorization"] = f"Bearer {token}"
            response = await client.post(url, headers=headers, json=data)

        if response.status_code == 200:
            res_data = response.json()
            audio_b64 = res_data.get("audioContent", "")
            if audio_b64:
                import base64
                raw_audio = base64.b64decode(audio_b64)
                # Strip 44-byte WAV header if LINEAR16 output returns container header
                if raw_audio.startswith(b"RIFF") and len(raw_audio) > 44:
                    audio_content = raw_audio[44:]
                else:
                    audio_content = raw_audio
                TTS_PCM_CACHE[cache_key] = audio_content
                return audio_content
        logger.error(f"Google TTS PCM synthesis failed with status {response.status_code}: {response.text}")
        raise RuntimeError(f"Google TTS PCM synthesis failed status {response.status_code}: {response.text}")

    if tts_provider in ("openai", "openai_tts"):
        openai_voice = voice or getattr(settings, "TTS_VOICE", "nova")
        if len(openai_voice) > 15:
            openai_voice = "nova"
        openai_model = model_id if (model_id and "tts-1" in str(model_id)) else getattr(settings, "TTS_MODEL", "tts-1")
        try:
            logger.info(f"Synthesizing PCM speech via OpenAI TTS (Voice: {openai_voice}, Model: {openai_model})...")
            response = await async_client.audio.speech.create(
                model=openai_model,
                voice=openai_voice,
                input=clean_text,
                response_format="pcm",
                speed=float(tts_speed)
            )
            audio_content = response.content
            TTS_PCM_CACHE[cache_key] = audio_content
            return audio_content
        except Exception as e:
            logger.error(f"OpenAI PCM TTS synthesis request failed: {e}")
            raise e

    tts_model_id = model_id if (model_id and "eleven" in str(model_id)) else settings.ELEVENLABS_MODEL_ID
    
    # Specify pcm_24000 to get S16LE 24kHz mono PCM with streaming latency optimization level 3
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{tts_voice_id}?output_format=pcm_24000&optimize_streaming_latency=3"
    headers = {
        "xi-api-key": settings.ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": clean_text,
        "model_id": tts_model_id,
        "voice_settings": {
            "stability": 0.65,
            "similarity_boost": 0.75,
            "speed": float(tts_speed)
        }
    }
    try:
        logger.info(f"Synthesizing response text to PCM via ElevenLabs (Voice ID: {tts_voice_id})...")
        client = _get_tts_http_client()
        response = await client.post(url, headers=headers, json=data)
        if response.status_code != 200:
            logger.error(f"ElevenLabs API returned error {response.status_code}: {response.text}")
            raise RuntimeError(f"ElevenLabs PCM synthesis failed: {response.text}")
        audio_content = response.content
        TTS_PCM_CACHE[cache_key] = audio_content
        logger.info(f"ElevenLabs PCM synthesis successful. Size: {len(audio_content)} bytes")
        return audio_content
    except Exception as e:
        err_msg = str(e) if str(e) else repr(e)
        logger.error(f"ElevenLabs PCM synthesis request failed: {err_msg}")
        raise RuntimeError(f"ElevenLabs PCM synthesis failed: {err_msg}") from e

async def extract_survey_data(history: List[Dict[str, str]], session_id: str = None, survey_config: dict = None) -> Any:
    """
    Runs structured extraction at the end of the survey using OpenAI Structured Outputs.
    Parses the conversation transcript into the dynamically constructed Pydantic schema based on survey steps config.
    """
    if not survey_config:
        # 1. Fetch survey config from database based on session
        survey_id = "default"
        if session_id:
            try:
                from app.core.db import get_collection
                sessions_col = get_collection("survey_sessions")
                logger.info(f"[DB-READ] Reading survey_sessions from MongoDB in extract_survey_data for session_id: {session_id}")
                session = await sessions_col.find_one({"session_id": session_id})
                if session:
                    survey_id = session.get("survey_id", "default")
            except Exception as ex:
                logger.error(f"Failed to get survey_id in extract_survey_data: {ex}")
                
        from app.core.db import get_survey_config
        logger.info(f"[DB-READ] Reading survey_config from MongoDB in extract_survey_data for survey_id: {survey_id}")
        survey_config = await get_survey_config(survey_id)
        
    survey_steps = survey_config.get("survey_steps", []) if survey_config else []
    
    
    # 2. Dynamically build schema using pydantic.create_model
    from pydantic import create_model, Field
    from typing import Literal
    
    fields = {}
    for step in survey_steps:
        field_name = step["field"]
        type_str = step.get("type", "str")
        options = step.get("options", [])
        
        valid_values = []
        if isinstance(options, list) and options:
            for opt in options:
                if isinstance(opt, dict) and "value" in opt:
                    valid_values.append(str(opt["value"]))
                elif isinstance(opt, str):
                    valid_values.append(opt)
        
        valid_values.extend(["Unclear", "Skipped"])
                    
        if valid_values:
            # Create a Literal type with the valid options
            field_type = Optional[Literal[tuple(valid_values)]]
        elif type_str == "int":
            field_type = Optional[int]
        elif type_str == "list":
            field_type = Optional[List[str]]
        else:
            field_type = Optional[str]
            
        desc = step.get("description") or f"Value for {field_name}."
        if valid_values:
            desc += f" MUST strictly be one of: {valid_values}. If not a strict match or out of context, output null."
        else:
            desc += " Must be translated/written in English."
            
        fields[field_name] = (field_type, Field(None, description=desc))
        
    DynamicTargetSurveyData = create_model("DynamicTargetSurveyData", **fields)
    
    extraction_system_prompt = (
        "You are an expert data extraction assistant. Your job is to extract structured survey data from the "
        "provided conversation history between a survey assistant and a customer. "
        "CRITICAL RULE 1: DO NOT extract data or infer answers for questions that the SYSTEM has not explicitly asked yet in the transcript. "
        "If the SYSTEM has not asked a specific question, you MUST leave its corresponding field as null, even if you think you can infer the answer from the context.\n"
        "CRITICAL RULE 2: If a field was not mentioned or answered by the customer, leave it as null.\n"
        "CRITICAL RULE 3: Pay strict attention to the exact question the SYSTEM asked. If the user gives a generic answer like 'Yes, I am satisfied', "
        "you MUST ONLY extract this for the specific question the SYSTEM just asked. Do NOT hallucinate and populate other similar but unasked questions.\n"
        "For annual income, parse the spoken description into a clean integer in Indian Rupees (INR). "
        "For example, '15 lakhs' -> 1500000, '5 lakh' -> 500000, '50 thousand a month' -> 600000.\n"
        "CRITICAL RULE 4: Translate any non-English spoken answers/extracted textual values (like additional_feedback, "
        "investment_preferences, etc.) to the English language before extracting and populating the schema."
    )
    
    messages = [{"role": "system", "content": extraction_system_prompt}]
    
    for turn in history:
        role = "assistant" if turn["speaker"] == "SYSTEM" else "user"
        messages.append({"role": role, "content": turn["text_content"]})
        
    try:
        fallback_model = getattr(settings, "CHAT_MODEL", "gpt-5.6-luna")
        if survey_config:
            fallback_model = survey_config.get("chat_model") or survey_config.get("llm_model") or fallback_model
            
        logger.info(f"Running structured survey response extraction via {fallback_model}...")
        response = await async_client.beta.chat.completions.parse(
            model=fallback_model,
            messages=messages,
            response_format=DynamicTargetSurveyData
        )
        extracted = response.choices[0].message.parsed
        # Clean "null"/"none" strings from the extracted fields
        extracted_dict = extracted.model_dump()
        for k, v in extracted_dict.items():
            if isinstance(v, str) and v.strip().lower() in ["null", "none"]:
                extracted_dict[k] = None
        cleaned_extracted = DynamicTargetSurveyData(**extracted_dict)
        logger.info(f"Structured survey data extracted successfully: {cleaned_extracted.model_dump()}")
        
        if session_id:
            try:
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                from app.services.cost_service import track_turn_cost
                await track_turn_cost(
                    session_id=session_id,
                    turn_type="EXTRACTION",
                    llm_prompt_tokens=prompt_tokens,
                    llm_completion_tokens=completion_tokens,
                    llm_model=fallback_model
                )
            except Exception as ex:
                logger.error(f"Failed to track cost in extract_survey_data: {ex}")
                
        return cleaned_extracted
    except Exception as e:
        logger.error(f"Error in structured data extraction: {e}")
        raise e

async def translate_to_english(text: str, session_id: str = None) -> str:
    """
    Translates the given text into English using GPT-5.6-luna if it is not already in English.
    """
    if not text or text.strip() in ["[Silence]", "[Unintelligible speech]"]:
        return text
    # Skip OpenAI translation if the text is already in ASCII (English / transliterated Latin script)
    if text.isascii():
        return text
    try:
        fallback_model = getattr(settings, "CHAT_MODEL", "gpt-5.6-luna")
        logger.info(f"Translating customer text to English via {fallback_model}: '{text}'")
        create_kwargs = {
            "model": fallback_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional translator and transliterator. Translate the given text to English. "
                        "If the text is a person's name, transliterate it to Latin characters (e.g. 'राहुल' -> 'Rahul') "
                        "but do NOT invent or add any surname, and do not translate honorifics like 'Ji' to arbitrary names. "
                        "If it is already in English/Latin, return it exactly as is without any added prefix, explanation, or notes."
                    )
                },
                {"role": "user", "content": text}
            ],
            "max_completion_tokens": 1024
        }
        if not is_no_temperature_model(fallback_model):
            create_kwargs["temperature"] = 0.0
            
        create_kwargs["timeout"] = 5.0

        response = await async_client.chat.completions.create(**create_kwargs)
        translated = response.choices[0].message.content.strip()
        logger.info(f"Translation result: '{translated}'")
        
        if session_id:
            try:
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                from app.services.cost_service import track_turn_cost
                await track_turn_cost(
                    session_id=session_id,
                    turn_type="TRANSLATION",
                    llm_prompt_tokens=prompt_tokens,
                    llm_completion_tokens=completion_tokens,
                    llm_model=fallback_model
                )
            except Exception as ex:
                logger.error(f"Failed to track cost in translate_to_english: {ex}")
                
        return translated
    except Exception as e:
        logger.error(f"Error in translating text to English: {e}")
        return text

async def translate_extracted_data(data: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
    """
    Ensures all textual fields in the extracted survey data dictionary are in English.
    """
    # 1. Fetch survey config
    survey_id = "default"
    if session_id:
        try:
            from app.core.db import get_collection
            sessions_col = get_collection("survey_sessions")
            session = await sessions_col.find_one({"session_id": session_id})
            if session:
                survey_id = session.get("survey_id", "default")
        except Exception as ex:
            logger.error(f"Failed to get survey_id in translate_extracted_data: {ex}")
            
    from app.core.db import get_survey_config
    survey_config = await get_survey_config(survey_id)

    survey_steps = survey_config.get("survey_steps", []) if survey_config else []
    
    # 2. Iterate through survey steps to translate based on field type
    for step in survey_steps:
        field_name = step["field"]
        type_str = step.get("type", "str")
        val = data.get(field_name)
        if not val:
            continue
            
        if type_str == "str":
            if isinstance(val, str) and val.strip():
                data[field_name] = await translate_to_english(val, session_id=session_id)
        elif type_str == "list":
            if isinstance(val, list):
                translated_prefs = []
                for pref in val:
                    if isinstance(pref, str) and pref.strip():
                        translated_prefs.append(await translate_to_english(pref, session_id=session_id))
                    else:
                        translated_prefs.append(pref)
                data[field_name] = translated_prefs
                
    return data
