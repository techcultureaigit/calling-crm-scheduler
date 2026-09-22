import os
import random
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import asyncio
from app.core.logger import logger
from app.core.db import connect_to_mongo, close_mongo_connection, ensure_db_indexes
from app.services.scheduler_service import scheduler_loop

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB and start scheduler in the background
    logger.info("Initializing DB and starting background Scheduler...")
    connect_to_mongo()
    await ensure_db_indexes()
    
    scheduler_task = asyncio.create_task(scheduler_loop())
    
    yield
    
    # Shutdown: Cleanup
    logger.info("Shutting down Scheduler and DB...")
    scheduler_task.cancel()
    close_mongo_connection()

app = FastAPI(title="Calling CRM Scheduler API", lifespan=lifespan)

@app.get('/')
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.api_route("/api/v1/voice/dynamic-endpoint", methods=["GET"])
async def dynamic_voice_endpoint(request: Request):
    """
    Dynamic endpoint for voice streaming that returns the WebSocket URL.
    Handles both GET and POST requests from the Voice Bot platform.
    """
    print('hey')
    if request.method == "POST":
        data = await request.json()
    else:
        data = dict(request.query_params)
        
    logger.info(f"Dynamic endpoint received request: {request.method} data: {data}")
        
    from app.services.telecommunication_service import extract_call_sid
    call_id = extract_call_sid(data)
    ref_id = data.get("refId") or data.get("ref_id") or data.get("referenceId") or data.get("reference_id") or ""
    
    if not call_id:
        call_id = ref_id # fallback if callId is not sent
        
    # If Smartflo gives us both the real call_id and the ref_id, link them in the DB!
    if call_id and ref_id and call_id != ref_id and not str(call_id).startswith("ref_"):
        from app.core.db import get_collection
        mappings_col = get_collection("call_survey_mappings")
        await mappings_col.update_one(
            {"ref_id": str(ref_id)},
            {"$set": {"call_sid": str(call_id)}}
        )
        logger.info(f"Dynamic endpoint linked real call_id {call_id} to ref_id {ref_id} in DB.")

        try:
            mapping = await mappings_col.find_one({"ref_id": str(ref_id)})
            if mapping and mapping.get("customer_number"):
                c_num = mapping["customer_number"]
                sessions_col = get_collection("survey_sessions")
                results_col = get_collection("survey_results")
                await sessions_col.update_many({"customer_number": c_num, "status": "IN_PROGRESS"}, {"$set": {"call_sid": str(call_id)}})
                await results_col.update_many({"customer_number": c_num}, {"$set": {"call_sid": str(call_id)}})
        except Exception as sync_err:
            logger.error(f"Error syncing call_sid from dynamic-endpoint: {sync_err}")
    
    ws_urls_env = os.getenv("WEBSOCKET_URL") or os.getenv("WEBSOCket_URL") or "wss://voice-pilot-calling-server-2.techculture.ai"
    ws_urls = [url.strip().rstrip('/') for url in ws_urls_env.split(",") if url.strip()]
    base_ws_url = random.choice(ws_urls) if ws_urls else "wss://voice-pilot-calling-server-2.techculture.ai"
    
    # We construct the URL to point to our streaming endpoint with the call_id
    wss_url = f"{base_ws_url}/api/v1/survey/ws/{call_id}"
    
    logger.info(f"Dynamic endpoint resolved callId: {call_id}. Returning wss_url: {wss_url}")
    return {
        "success": True,
        "wss_url": wss_url
    }
