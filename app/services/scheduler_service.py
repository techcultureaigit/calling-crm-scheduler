import asyncio
from datetime import datetime, timedelta
from app.core.db import get_collection, get_survey_config
from app.services.telecommunication_service import get_my_numbers
from app.core.logger import logger
import httpx
from app.core.config import settings


async def check_websocket_active() -> bool:
    """
    Verifies that the server is active and responding by checking HTTP /health.
    """
    import httpx

    raw_urls = getattr(settings, "CALL_WORKFLOW_URLS", ["http://127.0.0.1:8000"])
    if isinstance(raw_urls, str):
        call_workflow_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
    else:
        call_workflow_urls = raw_urls
        
    if not call_workflow_urls:
        call_workflow_urls = ["http://127.0.0.1:8000"]

    for base_url in call_workflow_urls:
        http_url = f"{base_url.rstrip('/')}/health"
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                res = await client.get(http_url)
                if res.status_code == 200:
                    return True
        except Exception:
            pass

    return False

async def process_scheduled_surveys(specific_survey_id: str = None):
    """
    Scans the 'surveys' collection for scheduled surveys that have not been processed.
    If specific_survey_id is passed, runs only that survey (as a bypass).
    Downloads the contacts, parses them, and initiates outbound calls.
    Supports resuming interrupted surveys (status='processing').
    """
    try:
        surveys_col = get_collection("surveys")
        mappings_col = get_collection("call_survey_mappings")

        # Verify WebSocket endpoint is active before processing any scheduled surveys
        if not await check_websocket_active():
            logger.warning("WebSocket server / URL is NOT active or reachable. Skipping outbound survey processing until server is active.")
            return

        if specific_survey_id:
            from bson import ObjectId
            match_conditions = [
                {"survey_id": specific_survey_id},
                {"uuid": specific_survey_id}
            ]
            try:
                match_conditions.append({"_id": ObjectId(specific_survey_id)})
            except Exception:
                match_conditions.append({"_id": specific_survey_id})
                
            query = {
                "$or": match_conditions,
                "isDeleted": {"$ne": True},
                "scheduling_status": {"$nin": ["completed"]}
            }
            logger.info(f"Scheduler bypass triggered for specific survey: {specific_survey_id}")
        else:
            # Filter surveys that are scheduled, or in processing but stale (interrupted/crashed worker lease expired)
            # We use a 5-minute lease time. The active worker renews its lease via heartbeats during the dialing loop.
            lease_expiry = datetime.utcnow() - timedelta(minutes=5)
            query = {
                "isDeleted": {"$ne": True},
                "scheduling_status": {"$nin": ["completed", "failed"]},
                "$or": [
                    {"scheduling_status": "scheduled"},
                    {"status": "scheduled"},
                    {
                        "scheduling_status": "processing",
                        "$or": [
                            {"schedule.lastScheduledAt": {"$exists": False}},
                            {"schedule.lastScheduledAt": {"$lt": lease_expiry}}
                        ]
                    }
                ]
            }
        
        
        cursor = surveys_col.find(query)
        
        async for survey in cursor:
            survey_id = str(survey["_id"])
            if not survey_id:
                continue

            current_status = survey.get("scheduling_status") or "scheduled"

            # Validate schedule data only if starting fresh (not when resuming 'processing' state)
            if not specific_survey_id and current_status != "processing":
                schedule = survey.get("schedule")
                if not schedule or not isinstance(schedule, dict):
                    logger.info(f"Skipping survey {survey_id}: 'schedule' data is missing.")
                    continue

                if not schedule.get("enabled"):
                    logger.info(f"Skipping survey {survey_id}: schedule is not enabled (enabled=False).")
                    continue

                start_at_raw = schedule.get("startAt")
                if not start_at_raw:
                    logger.info(f"Skipping survey {survey_id}: schedule.startAt is missing or null.")
                    continue

                # Parse startAt date/time
                start_at_dt = None
                if isinstance(start_at_raw, datetime):
                    start_at_dt = start_at_raw
                elif isinstance(start_at_raw, str):
                    try:
                        from dateutil import parser as dt_parser
                        start_at_dt = dt_parser.parse(start_at_raw)
                    except Exception:
                        try:
                            start_at_dt = datetime.fromisoformat(start_at_raw.replace("Z", "+00:00"))
                        except Exception:
                            logger.error(f"Failed to parse startAt '{start_at_raw}' for survey {survey_id}")
                            continue

                if start_at_dt:
                    from datetime import timezone
                    now_utc = datetime.now(timezone.utc)
                    if start_at_dt.tzinfo is None:
                        start_at_dt = start_at_dt.replace(tzinfo=timezone.utc)
                    else:
                        start_at_dt = start_at_dt.astimezone(timezone.utc)

                    if now_utc < start_at_dt:
                        logger.info(f"Skipping survey {survey_id}: startAt ({start_at_dt.isoformat()}) is in the future. Current time: {now_utc.isoformat()}")
                        continue

                # Check endAt if specified
                end_at_raw = schedule.get("endAt")
                if end_at_raw:
                    end_at_dt = None
                    if isinstance(end_at_raw, datetime):
                        end_at_dt = end_at_raw
                    elif isinstance(end_at_raw, str):
                        try:
                            from dateutil import parser as dt_parser
                            end_at_dt = dt_parser.parse(end_at_raw)
                        except Exception:
                            try:
                                end_at_dt = datetime.fromisoformat(end_at_raw.replace("Z", "+00:00"))
                            except Exception:
                                pass
                    if end_at_dt:
                        from datetime import timezone
                        now_utc = datetime.now(timezone.utc)
                        if end_at_dt.tzinfo is None:
                            end_at_dt = end_at_dt.replace(tzinfo=timezone.utc)
                        else:
                            end_at_dt = end_at_dt.astimezone(timezone.utc)

                        if now_utc > end_at_dt:
                            logger.info(f"Skipping survey {survey_id}: endAt ({end_at_dt.isoformat()}) has passed.")
                            continue

            contact_url = survey.get("clientContact", {}).get("contactFileUrl") or survey.get("contactListUrl")
            if not contact_url:
                logger.warning(f"Survey {survey_id} is scheduled but missing contact file URL.")
                continue

            caller_id = None



            logger.info(f"Processing schedule for survey {survey_id} (status: {current_status}) with url: {contact_url}")

            # Rule 2: Set scheduling_status to 'processing' when starting atomically
            lease_expiry = datetime.utcnow() - timedelta(minutes=5)
            update_query = {"_id": survey["_id"]}
            if not specific_survey_id:
                # Standard concurrency lease check
                update_query["$or"] = [
                    {"scheduling_status": "scheduled"},
                    {"status": "scheduled"},
                    {
                        "scheduling_status": "processing",
                        "$or": [
                            {"schedule.lastScheduledAt": {"$exists": False}},
                            {"schedule.lastScheduledAt": {"$lt": lease_expiry}}
                        ]
                    }
                ]
            
            from pymongo import ReturnDocument
            updated_doc = await surveys_col.find_one_and_update(
                update_query,
                {"$set": {
                    "scheduling_status": "processing",
                    "schedule.status": "processing",
                    "schedule.lastScheduledAt": datetime.utcnow()
                }},
                return_document=ReturnDocument.AFTER
            )
            
            if not updated_doc:
                logger.info(f"Survey {survey_id} is already active or lease has not expired. Skipping.")
                continue
            
            try:
                from app.services.telecommunication_service import downsample_pcm_24k_to_8k, pcm_8k_to_mulaw
                from app.services import ai_service
                import base64, httpx as _httpx
                from app.core.db import invalidate_survey_config_cache

                # Invalidate cache BEFORE loading config to ensure we get the latest data
                invalidate_survey_config_cache(survey_id)
                survey_config = await get_survey_config(survey_id)
                tts_provider = survey_config.get("tts_provider") if survey_config else None
                voice_id = survey_config.get("tts_voice_id") if survey_config else None
                model_id = survey_config.get("tts_model_id") if survey_config else None
                survey_lang = survey_config.get("language") if survey_config else "hi"
                tts_speed = survey_config.get("tts_speed") if survey_config else None

                voice_name = survey_config.get("tts_voice_name") if survey_config else None
                voice_disp = f"{voice_id} ({voice_name})" if voice_name and voice_name != voice_id else voice_id

                logger.info(
                    f"\n============================================================\n"
                    f" SURVEY CONFIGURATION LOADED FOR PRE-GENERATION ({survey_id})\n"
                    f"============================================================\n"
                    f" • STT Provider : {survey_config.get('stt_provider')} (Model: {survey_config.get('stt_model')})\n"
                    f" • TTS Provider : {tts_provider} (Voice: {voice_disp}, Model: {model_id})\n"
                    f" • LLM Provider : {survey_config.get('llm_provider')} (Model: {survey_config.get('chat_model')})\n"
                    f" • Language     : {survey_lang}\n"
                    f"============================================================"
                )

                # 1. Pre-generate / load Initial Greeting
                existing_greeting_mulaw = survey.get("greetingAudioMulaw")

                if not existing_greeting_mulaw:
                    greeting_text = survey_config.get("initial_greeting", "") if survey_config else ""
                    if not greeting_text:
                        prompts = survey.get("prompts") or {}
                        greeting_text = prompts.get("greeting", "नमस्कार")

                    logger.info(f"Pre-generating greeting audio for survey {survey_id}: '{greeting_text[:50]}...'")
                    greeting_pcm = await ai_service.synthesize_speech_pcm(greeting_text, provider=tts_provider, voice=voice_id, model_id=model_id, language=survey_lang, speed=tts_speed)
                    greeting_pcm_8k = downsample_pcm_24k_to_8k(greeting_pcm)
                    
                    greeting_mulaw = pcm_8k_to_mulaw(greeting_pcm_8k)
                    greeting_mulaw_b64 = base64.b64encode(greeting_mulaw).decode("utf-8")

                    await surveys_col.update_one(
                        {"_id": survey["_id"]},
                        {"$set": {
                            "greetingAudioMulaw": greeting_mulaw_b64
                        }}
                    )
                    logger.info(f"Cached greeting audio for survey {survey_id}")

                # 2. Pre-generate Farewell Message
                farewell_text = survey_config.get("farewell_message", "") if survey_config else ""
                if not farewell_text:
                    prompts = survey.get("prompts") or {}
                    farewell_text = prompts.get("farewell", "")

                if farewell_text:
                    logger.info(f"Pre-generating farewell audio for survey {survey_id}")
                    farewell_pcm = await ai_service.synthesize_speech_pcm(farewell_text, provider=tts_provider, voice=voice_id, model_id=model_id, language=survey_lang, speed=tts_speed)
                    from app.services.telecommunication_service import downsample_pcm_24k_to_8k, pcm_8k_to_mulaw, trim_pcm_silence
                    farewell_pcm_8k = trim_pcm_silence(downsample_pcm_24k_to_8k(farewell_pcm))
                    
                    farewell_mulaw = pcm_8k_to_mulaw(farewell_pcm_8k)
                    farewell_mulaw_b64 = base64.b64encode(farewell_mulaw).decode("utf-8")
                    
                    await surveys_col.update_one(
                        {"_id": survey["_id"]},
                        {"$set": {
                            "farewellAudioMulaw": farewell_mulaw_b64
                        }}
                    )
                    logger.info(f"Cached farewell audio for survey {survey_id}")

                # 3. Pre-generate all Survey Questions
                survey_steps = survey_config.get("survey_steps", []) if survey_config else []
                questions_mulaw_dict = {}

                for idx, step in enumerate(survey_steps, 1):
                    q_text = step.get("question") or step.get("description") or ""
                    q_id = step.get("id") or f"q_{idx}"
                    if q_text:
                        logger.info(f"Pre-generating audio for Q{idx} ({q_id}): '{q_text[:50]}...'")
                        q_pcm = await ai_service.synthesize_speech_pcm(q_text, provider=tts_provider, voice=voice_id, model_id=model_id, language=survey_lang, speed=tts_speed)
                        from app.services.telecommunication_service import downsample_pcm_24k_to_8k, pcm_8k_to_mulaw, trim_pcm_silence
                        q_pcm_8k = trim_pcm_silence(downsample_pcm_24k_to_8k(q_pcm))
                        
                        q_mulaw = pcm_8k_to_mulaw(q_pcm_8k)
                        q_mulaw_b64 = base64.b64encode(q_mulaw).decode("utf-8")
                        questions_mulaw_dict[q_id] = q_mulaw_b64

                if questions_mulaw_dict:
                    await surveys_col.update_one(
                        {"_id": survey["_id"]},
                        {"$set": {"questionsAudioMulaw": questions_mulaw_dict}}
                    )
                    logger.info(f"Pre-cached audio for {len(questions_mulaw_dict)} questions in survey {survey_id}")

            except Exception as audio_pre_err:
                logger.error(f"Failed to pre-generate survey audio assets for survey {survey_id}: {audio_pre_err}. Will synthesize live.")
            
            try:
                from app.services.cloudinary_service import stream_contact_file
                filename = contact_url.split("/")[-1]
                
                # Rule 3: Resume support - fetch numbers already called for this survey to avoid duplicates
                existing_mappings = await mappings_col.find({"survey_id": survey_id}).to_list(length=50000)
                already_called_numbers = set()
                for m in existing_mappings:
                    c_num = m.get("customer_number")
                    if c_num:
                        already_called_numbers.add(str(c_num))
                        cleaned_c_num = "".join(filter(str.isdigit, str(c_num)))
                        if cleaned_c_num:
                            already_called_numbers.add(cleaned_c_num)
                
                # 3. Fetch My Numbers list from Smartflo to enforce dynamic call concurrency limit
                my_numbers_list = await get_my_numbers()
                my_numbers_count = len(my_numbers_list)
                max_concurrency = max(1, my_numbers_count) if my_numbers_count > 0 else 10

                print(my_numbers_list)
                
                agent_numbers = []
                for n in my_numbers_list:
                    if isinstance(n, dict):
                        num_val = n.get("number") or n.get("alias") or n.get("did")
                        if num_val:
                            # Strip '+' prefix if it came from 'did'
                            num_val = str(num_val).replace("+", "")
                            agent_numbers.append(num_val)
                    elif isinstance(n, str):
                        agent_numbers.append(n)
                        
                if not agent_numbers:
                    raise RuntimeError("No Smartflo agent numbers available. Check your Smartflo authentication token.")
                    
                logger.info(f"Smartflo My Numbers available: {my_numbers_count}. Enforcing maximum concurrent call limit = {max_concurrency}")

                sessions_col = get_collection("survey_sessions")

                # 4. Trigger calls iteratively
                logger.info(f"Scheduling calls iteratively (already called: {len(already_called_numbers)})...")
                
                active_tasks = set()
                
                raw_urls = getattr(settings, "CALL_WORKFLOW_URLS", ["http://127.0.0.1:8000"])
                if isinstance(raw_urls, str):
                    call_workflow_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
                else:
                    call_workflow_urls = raw_urls
                    
                if not call_workflow_urls:
                    call_workflow_urls = ["http://127.0.0.1:8000"]
                
                async def trigger_call(num_to_dial, agent_idx):
                    try:
                        current_agent = agent_numbers[agent_idx % len(agent_numbers)]
                        current_url = call_workflow_urls[agent_idx % len(call_workflow_urls)]
                        logger.info(f"Initiating call directly from Scheduler for survey {survey_id}: Agent={current_agent}, Destination={num_to_dial}, Workflow Server={current_url} (Tasks In-Flight: {len(active_tasks)})")
                        
                        from app.services.telecommunication_service import initiate_call
                        result = await initiate_call(
                            agent_number=current_agent,
                            destination_number=num_to_dial,
                            caller_id=current_agent,
                            is_async=1,
                            survey_id=survey_id,
                            create_mapping=True,
                            custom_identifier={
                                "survey_id": survey_id,
                                "customer_number": num_to_dial,
                                "server_url": current_url
                            }
                        )
                        logger.info(f"Scheduler direct call initiation completed for {num_to_dial}: {result}")
                    except Exception as call_err:
                        logger.error(f"Failed to trigger call to {num_to_dial} for survey {survey_id}: {call_err}")

                total_processed = 0

                async for num in stream_contact_file(contact_url, filename):
                    cleaned_num = "".join(filter(str.isdigit, str(num)))
                    if str(num) in already_called_numbers or (cleaned_num and cleaned_num in already_called_numbers):
                        logger.info(f"Skipping already called contact {num} for survey {survey_id}")
                        continue

                    # Concurrency check
                    while True:
                        active_calls_count = await sessions_col.count_documents({
                            "status": "IN_PROGRESS",
                            "created_at": {"$gte": datetime.utcnow() - timedelta(minutes=2)}
                        })
                        
                        # Clean up done tasks
                        done_tasks = {t for t in active_tasks if t.done()}
                        active_tasks.difference_update(done_tasks)
                        
                        current_in_flight = active_calls_count + len(active_tasks)
                        if current_in_flight < max_concurrency:
                            break
                            
                        logger.info(f"Active/Pending calls ({current_in_flight}) reached concurrency limit ({max_concurrency}). Pausing 2s for available slot...")
                        await asyncio.sleep(2.0)

                    task = asyncio.create_task(trigger_call(num, total_processed))
                    active_tasks.add(task)
                    total_processed += 1
                    
                    done_tasks = {t for t in active_tasks if t.done()}
                    active_tasks.difference_update(done_tasks)

                    await surveys_col.update_one(
                        {"_id": survey["_id"]},
                        {"$set": {"schedule.lastScheduledAt": datetime.utcnow()}}
                    )
                    await asyncio.sleep(0.5)
                
                if total_processed == 0 and len(already_called_numbers) == 0:
                    logger.warning(f"No numbers found in contact file for survey {survey_id}")
                    await surveys_col.update_one(
                        {"_id": survey["_id"]},
                        {"$set": {"scheduling_status": "completed", "schedule.status": "completed", "error": "No numbers found"}}
                    )
                    continue

                # Wait for remaining tasks
                if active_tasks:
                    logger.info(f"Waiting for {len(active_tasks)} pending trigger tasks to complete...")
                    await asyncio.gather(*active_tasks, return_exceptions=True)
                
                # Rule 2: Update status to completed after whole survey is completed
                await surveys_col.update_one(
                    {"_id": survey["_id"]},
                    {"$set": {
                        "scheduling_status": "completed",
                        "schedule.status": "completed",
                        "processed_at": datetime.utcnow(),
                        "total_contacts_called": total_processed + len(already_called_numbers)
                    }}
                )
                logger.info(f"Successfully finished call scheduling for survey {survey_id}")
                
            except Exception as proc_err:
                logger.error(f"Error processing schedule for survey {survey_id}: {proc_err}")
                await surveys_col.update_one(
                    {"_id": survey["_id"]},
                    {"$set": {
                        "scheduling_status": "failed",
                        "schedule.status": "failed",
                        "error": str(proc_err)
                    }}
                )
    except Exception as e:
        logger.error(f"Error in process_scheduled_surveys: {e}")

async def scheduler_loop():
    """
    Background loop that runs every 30 seconds to scan and process scheduled calls.
    """
    logger.info("Outbound call scheduler service started.")
    while True:
        try:
            logger.info("Scanning for scheduled surveys...")
            await process_scheduled_surveys()
        except Exception as e:
            logger.error(f"Error in scheduler_loop iteration: {e}")
        await asyncio.sleep(30.0)

_scheduler_task = None

def start_scheduler():
    global _scheduler_task
    if _scheduler_task is None:
        _scheduler_task = asyncio.create_task(scheduler_loop())
        logger.info("Scheduler task created in background.")
