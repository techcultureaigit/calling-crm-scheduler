import urllib.parse
from typing import Optional, Tuple
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.core.logger import logger
import time

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

db_client = MongoDB()

def escape_mongodb_url(url: str) -> str:
    """
    Automatically escapes special characters (like @ or :) in the username
    and password parts of a MongoDB connection URI to conform to RFC 3986.
    """
    if "@" not in url:
        return url
        
    scheme = "mongodb"
    if "://" in url:
        scheme, rest = url.split("://", 1)
    else:
        rest = url
        
    query_part = ""
    if "?" in rest:
        rest, query_part = rest.split("?", 1)
        query_part = "?" + query_part
        
    path_part = ""
    if "/" in rest:
        parts = rest.rsplit("@", 1)
        if len(parts) == 2:
            creds, host_db = parts
            if "/" in host_db:
                host, db = host_db.split("/", 1)
                path_part = "/" + db
                rest = f"{creds}@{host}"
                
    parts = rest.rsplit("@", 1)
    if len(parts) != 2:
        return url
        
    creds, host = parts
    if ":" not in creds:
        username = urllib.parse.quote_plus(creds)
        return f"{scheme}://{username}@{host}{path_part}{query_part}"
        
    username, password = creds.split(":", 1)
    safe_user = urllib.parse.quote_plus(username)
    safe_pass = urllib.parse.quote_plus(password)
    return f"{scheme}://{safe_user}:{safe_pass}@{host}{path_part}{query_part}"

def connect_to_mongo():
    try:
        escaped_url = escape_mongodb_url(settings.MONGODB_URL)
        logger.info(f"Connecting to MongoDB (database: '{settings.DATABASE_NAME}')...")
        db_client.client = AsyncIOMotorClient(escaped_url)
        db_client.db = db_client.client[settings.DATABASE_NAME]
        logger.info("Successfully initialized MongoDB AsyncIOMotorClient.")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise e

def close_mongo_connection():
    if db_client.client:
        db_client.client.close()
        logger.info("Closed MongoDB connection session.")

def get_database():
    if db_client.db is None:
        connect_to_mongo()
    return db_client.db

def get_collection(name: str):
    db = get_database()
    return db[name]

async def resolve_model_and_provider_by_id(model_id_val) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Given a model_id (ObjectId, string, or {'$oid': '...'}),
    looks up the model in 'models' collection and the provider in 'providers' collection.
    Returns (model_name, provider_name, provider_type).
    """
    if not model_id_val:
        return None, None, None
        
    if isinstance(model_id_val, dict) and "$oid" in model_id_val:
        model_id_val = model_id_val["$oid"]
        
    target_id = str(model_id_val).strip()
    if not target_id or target_id == "not_set":
        return None, None, None
        
    try:
        db = get_database()
        from bson import ObjectId
        
        model_doc = None
        if ObjectId.is_valid(target_id):
            model_doc = await db["models"].find_one({"_id": ObjectId(target_id)})
        if not model_doc:
            model_doc = await db["models"].find_one({"_id": target_id})
            
        if not model_doc:
            return None, None, None
            
        model_name = (
            model_doc.get("name") 
            or model_doc.get("model") 
            or model_doc.get("modelId") 
            or model_doc.get("model_id")
        )
        provider_id = (
            model_doc.get("provider_id") 
            or model_doc.get("providerId") 
            or model_doc.get("provider") 
            or model_doc.get("provider_name")
        )
        
        if not provider_id:
            return model_name, None, None
            
        if isinstance(provider_id, dict):
            provider_id = provider_id.get("$oid") or provider_id.get("_id") or str(provider_id)
            
        prov_doc = None
        if ObjectId.is_valid(str(provider_id)):
            prov_doc = await db["providers"].find_one({"_id": ObjectId(str(provider_id))})
        if not prov_doc:
            prov_doc = await db["providers"].find_one({"_id": str(provider_id)})
            
        prov_name = (
            prov_doc.get("name") 
            or prov_doc.get("provider") 
            or prov_doc.get("slug") 
            or prov_doc.get("type")
        ) if prov_doc else None
        prov_type = prov_doc.get("type") if prov_doc else None
        
        return model_name, prov_name, prov_type
    except Exception as ex:
        logger.warning(f"Error resolving model/provider by ID ({model_id_val}): {ex}")
        return None, None, None

async def resolve_voice_by_id(voice_val) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a voice value (ObjectId string, {'$oid': '...'}, or voice ID/name),
    looks up the voice in 'voices' collection.
    Returns (actual_voice_id, voice_display_name).
    """
    if not voice_val:
        return None, None
        
    if isinstance(voice_val, dict) and "$oid" in voice_val:
        voice_val = voice_val["$oid"]
        
    target_id = str(voice_val).strip()
    if not target_id:
        return None, None
        
    try:
        db = get_database()
        from bson import ObjectId
        
        voice_doc = None
        if ObjectId.is_valid(target_id):
            voice_doc = await db["voices"].find_one({"_id": ObjectId(target_id)})
        if not voice_doc:
            voice_doc = await db["voices"].find_one({"_id": target_id})
            
        if voice_doc:
            actual_voice_id = (
                voice_doc.get("voiceId") 
                or voice_doc.get("voice_id") 
                or voice_doc.get("voice") 
                or voice_doc.get("name")
            )
            voice_name = voice_doc.get("name") or actual_voice_id
            return actual_voice_id, voice_name
    except Exception as ex:
        logger.warning(f"Error resolving voice by ID ({voice_val}): {ex}")
        
    return target_id, target_id

def _flatten_questions_to_steps(questions: list, parent_condition: str = None) -> list:
    flattened = []
    for idx, q in enumerate(questions):
        q_id = q.get("id") or f"q_{idx}"
        q_type = q.get("type") or "text"
        
        type_str = "str"
        if q_type in ("number", "int"):
            type_str = "int"
        elif q_type == "rating":
            type_str = "int"
        elif q_type in ("list", "multi"):
            type_str = "list"
            
        options = q.get("options") or []
        if q_type == "yes_no" and not options:
            options = [{"label": "Yes", "value": "Yes"}, {"label": "No", "value": "No"}]
            
        options_val_map = []
        options_formatted = []
        for opt in options:
            if isinstance(opt, str):
                options_val_map.append({"label": opt, "value": opt})
                options_formatted.append(f"'{opt}'")
            elif isinstance(opt, dict):
                lbl = str(opt.get("label") or opt.get("text") or str(opt))
                val = str(opt.get("value") or lbl)
                options_val_map.append({"label": lbl, "value": val})
                options_formatted.append(f"'{lbl}' (value: '{val}')")

        if options_formatted:
            read_aloud_prompt = ""
            if q_type in ("mcq", "list", "multi", "radio", "dropdown"):
                read_aloud_prompt = " Since this is a multiple-choice question, you MUST explicitly read the available options aloud to the user when asking the question. To maintain a natural conversational flow, incorporate the options directly into your sentence rather than listing them as bullet points or numbers (e.g. ask 'Would you like A, B, or C?' instead of '1. A, 2. B, 3. C'). "
                
            options_str = (
                f" Allowed Options: [{', '.join(options_formatted)}]."
                f"{read_aloud_prompt} "
                "CRITICAL MANDATORY RULE: If the user's answer does not match any valid option or is out of context, politely acknowledge what they said and ask them to choose strictly from the allowed options. "
                "HOWEVER, if you have already asked them this question twice and they STILL give an invalid answer, ACCEPT whatever they said, extract it exactly as they said it prefixed with 'FALLBACK: ', and MOVE ON to the next question."
            )
        else:
            options_str = ""

        if q_type == "rating":
            options_str += " Ensure you extract the rating as a single integer number."

        # Support custom instructions/hints provided by the user in the UI
        custom_instructions = q.get("instructions") or q.get("instruction") or q.get("description") or ""
        custom_instructions_str = f" EXTRA INSTRUCTIONS FOR THIS QUESTION: {custom_instructions}" if custom_instructions else ""

        step = {
            "id": q_id,
            "field": q_id,
            "question": q.get("question", ""),
            "instruction": f"Ask the customer: '{q.get('question', '')}'.{options_str}{custom_instructions_str}",
            "skip_if": parent_condition,
            "type": type_str,
            "options": options_val_map,
            "description": q.get("question", "")
        }
        flattened.append(step)
        
        conditions = q.get("conditions") or []
        for cond in conditions:
            if_answer = cond.get("ifAnswer")
            then_questions = cond.get("thenShowQuestions") or []
            if if_answer and then_questions:
                my_cond = f"str(data.get('{q_id}', '')).lower() != '{str(if_answer).lower()}'"
                combined_cond = f"({parent_condition}) or ({my_cond})" if parent_condition else my_cond
                flattened.extend(_flatten_questions_to_steps(then_questions, combined_cond))
                
    return flattened

async def normalize_survey_config(survey_config: dict) -> dict:
    if not survey_config:
        return {}
    
    normalized = dict(survey_config)
    uuid_val = (
        survey_config.get("uuid") 
        or survey_config.get("survey_id") 
        or str(survey_config.get("_id", "default"))
    )
    normalized["survey_id"] = uuid_val
    normalized["uuid"] = uuid_val
    normalized["assistant_name"] = (
        survey_config.get("assistant_name") 
        or survey_config.get("name") 
        or "Aisha"
    )
    
    persona = survey_config.get("persona") or {}
    stt = persona.get("stt") or {}
    tts = persona.get("tts") or {}
    llm = persona.get("llm") or {}
    
    # 1. STT Resolution via ObjectIDs or string fields
    stt_model_ref = stt.get("modelId") or stt.get("model_id") or stt.get("model") or survey_config.get("stt_model_id") or survey_config.get("stt_model")
    stt_m_name, stt_p_name, _ = await resolve_model_and_provider_by_id(stt_model_ref)

    normalized["stt_provider"] = (
        stt_p_name 
        or stt.get("provider") 
        or stt.get("name") 
        or survey_config.get("stt_provider") 
        or getattr(settings, "STT_PROVIDER", "elevenlabs")
    )
    normalized["stt_model"] = (
        stt_m_name 
        or stt.get("model") 
        or survey_config.get("stt_model") 
        or getattr(settings, "STT_MODEL", "scribe_v2")
    )
    
    # 2. TTS Resolution via ObjectIDs or string fields
    tts_model_ref = tts.get("modelId") or tts.get("model_id") or tts.get("model") or survey_config.get("tts_model_id") or survey_config.get("tts_model")
    tts_m_name, tts_p_name, _ = await resolve_model_and_provider_by_id(tts_model_ref)

    raw_tts_prov = (
        tts_p_name 
        or tts.get("provider") 
        or tts.get("name") 
        or survey_config.get("tts_provider") 
        or survey_config.get("voice_provider")
        or getattr(settings, "TTS_PROVIDER", "openai")
    )
    tts_prov = str(raw_tts_prov).lower().strip()
    normalized["tts_provider"] = tts_prov
    
    # Determine default voice for the provider
    if tts_prov in ("google", "google_tts"):
        def_voice = "hi-IN-Chirp3-HD-Achernar"
        def_model = "default"
    elif tts_prov in ("openai", "openai_tts"):
        def_voice = getattr(settings, "TTS_VOICE", "nova")
        def_model = getattr(settings, "TTS_MODEL", "tts-1")
    elif tts_prov == "sarvam":
        def_voice = "meera"
        def_model = "bulbul:v1"
    else:
        def_voice = getattr(settings, "ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
        def_model = getattr(settings, "ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")

    raw_voice = (
        tts.get("voice_id") 
        or tts.get("voice") 
        or survey_config.get("tts_voice_id") 
        or survey_config.get("voice_id") 
        or survey_config.get("voice")
    )
    res_voice_id, res_voice_name = await resolve_voice_by_id(raw_voice)
    normalized["tts_voice_id"] = res_voice_id if res_voice_id else def_voice
    normalized["tts_voice_name"] = res_voice_name if res_voice_name else normalized["tts_voice_id"]
    
    raw_model = (
        tts_m_name 
        or tts.get("model") 
        or survey_config.get("tts_model_id") 
        or survey_config.get("model_id") 
        or survey_config.get("model")
    )
    normalized["tts_model_id"] = raw_model if raw_model else def_model
    normalized["tts_speed"] = (
        tts.get("tts_speed")
        or tts.get("speed")
        or (persona.get("tts_speed") if isinstance(persona, dict) else None)
        or survey_config.get("tts_speed")
        or survey_config.get("speech_speed")
        or survey_config.get("speed")
    )
    
    # 3. LLM Resolution via ObjectIDs or string fields
    llm_model_ref = llm.get("modelId") or llm.get("model_id") or llm.get("model") or survey_config.get("chat_model") or survey_config.get("llm_model_id")
    llm_m_name, llm_p_name, _ = await resolve_model_and_provider_by_id(llm_model_ref)

    normalized["llm_provider"] = (
        llm_p_name 
        or llm.get("provider") 
        or llm.get("name") 
        or survey_config.get("llm_provider") 
        or survey_config.get("chat_provider") 
        or "openai"
    )
    normalized["chat_model"] = (
        llm_m_name 
        or llm.get("model") 
        or survey_config.get("chat_model") 
        or getattr(settings, "CHAT_MODEL", "gpt-5.6-luna")
    )
    
    # 4. Language & Prompts
    normalized["language"] = persona.get("language") or survey_config.get("language") or "hi"
    
    prompts = survey_config.get("prompts") or {}
    normalized["initial_greeting"] = (
        prompts.get("greeting") 
        or survey_config.get("initial_greeting") 
        or survey_config.get("greeting") 
        or "नमस्कार"
    )
    normalized["farewell_message"] = (
        prompts.get("farewell") 
        or survey_config.get("farewell_message") 
        or survey_config.get("farewell") 
        or ""
    )
    normalized["system_prompt"] = (
        prompts.get("systemPrompt") 
        or survey_config.get("system_prompt") 
        or ""
    )
    
    # 5. Survey Steps / Questions
    survey_steps = survey_config.get("survey_steps")
    if not survey_steps:
        survey_steps = []
        survey_questions_container = survey_config.get("surveyQuestions") or {}
        questions = survey_questions_container.get("questions") or []
        survey_steps = _flatten_questions_to_steps(questions)
            
            
        normalized["survey_steps"] = survey_steps
        
        # Keep other fields
        for k, v in survey_config.items():
            if k not in normalized:
                normalized[k] = v
        return normalized
    
    return survey_config

SURVEY_CONFIG_CACHE = {}

def invalidate_survey_config_cache(survey_id: str = None):
    """Clears in-memory survey config cache so newly updated fields take effect immediately."""
    if survey_id:
        SURVEY_CONFIG_CACHE.pop(str(survey_id), None)
        SURVEY_CONFIG_CACHE.clear()
    else:
        SURVEY_CONFIG_CACHE.clear()

async def ensure_db_indexes():
    """Background creation of MongoDB indexes for fast lookups."""
    try:
        db = get_database()
        await db["call_survey_mappings"].create_index([("call_sid", 1)])
        await db["call_survey_mappings"].create_index([("customer_number", 1)])
        await db["call_survey_mappings"].create_index([("created_at", -1)])
        await db["survey_sessions"].create_index([("session_id", 1)])
        await db["surveys"].create_index([("survey_id", 1)])
        logger.info("MongoDB indexes initialized successfully.")
    except Exception as idx_err:
        logger.warning(f"Failed to create MongoDB indexes: {idx_err}")

async def get_survey_config(survey_id: str) -> Optional[dict]:
    """
    Resolves survey config with 0ms in-memory TTL caching and safe BSON lookup.
    Supports ObjectId, string _id, survey_id, uuid, and default fallback.
    """
    if not survey_id:
        survey_id = "default"

    now = time.time()
    if survey_id in SURVEY_CONFIG_CACHE:
        cached_time, cached_data = SURVEY_CONFIG_CACHE[survey_id]
        if now - cached_time < 60:
            return cached_data

    surveys_col = get_collection("surveys")
    survey = None

    # 1. Try finding by ObjectId if valid 24-hex string
    from bson import ObjectId
    try:
        if ObjectId.is_valid(survey_id):
            survey = await surveys_col.find_one({"_id": ObjectId(survey_id)})
    except Exception:
        pass

    # 2. Try finding by string _id
    if not survey:
        try:
            survey = await surveys_col.find_one({"_id": str(survey_id)})
        except Exception:
            pass

    # 3. Try finding by survey_id field
    if not survey:
        survey = await surveys_col.find_one({"survey_id": str(survey_id)})

    # 4. Try finding by uuid field
    if not survey:
        survey = await surveys_col.find_one({"uuid": str(survey_id)})

    # 5. Fall back to default survey config if specific survey ID not found
    if not survey and survey_id != "default":
        survey = await surveys_col.find_one({"survey_id": "default"})

    if survey:
        normalized = await normalize_survey_config(survey)
        SURVEY_CONFIG_CACHE[survey_id] = (now, normalized)
        return normalized

    return None
    