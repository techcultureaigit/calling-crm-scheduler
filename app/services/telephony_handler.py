import os
import uuid
import asyncio
import base64
import json
import time
from typing import Tuple, Optional
from datetime import datetime, timedelta
import audioop
from fastapi import WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.db import get_collection, get_survey_config
from app.core.logger import logger
from app.services import ai_service, survey_rules

# Global dictionary to track session state: "LISTENING", "PROCESSING", "SPEAKING", or "INTERRUPTED"
session_states = {}

# Global dictionary for session-level concurrency locks
session_locks = {}

# In-memory fast cache for survey mu-law greeting audio (survey_id -> bytes)
AUDIO_MULAW_CACHE = {}

import re

def sanitize_tts_text(text: str) -> str:
    """
    Strips stray Unicode characters from non-target scripts (e.g. Kannada, Tamil, Telugu)
    that the LLM occasionally outputs. Keeps only:
    - Latin (A-Z, a-z, digits, punctuation)
    - Devanagari (Hindi) U+0900-U+097F
    - Common punctuation, spaces, and symbols
    """
    # Remove system commands
    text = text.replace("[END_CALL]", "")
    
    # Keep: ASCII printable, Devanagari block, Devanagari Extended, common punctuation
    cleaned = re.sub(r'[^\u0000-\u007F\u0900-\u097F\u0980-\u09FF\u2000-\u206F\u2018-\u201F।॥]+', '', text)
    return cleaned.strip()



# --- Ambient Track: Pre-loaded PCM to mix under speech ---
import random

AMBIENT_TRACKS = {}
COMFORT_TONE_PCM = b""

def _load_ambient_wavs():
    """Load all ambient background WAVs from the ambience directory for mixing under TTS."""
    import os
    import wave as _wave
    global AMBIENT_TRACKS
    ambience_dir = os.path.join(os.path.dirname(__file__), "..", "static", "audio", "ambience")
    ambience_dir = os.path.abspath(ambience_dir)
    try:
        if os.path.exists(ambience_dir):
            for filename in os.listdir(ambience_dir):
                if filename.endswith(".wav"):
                    wav_path = os.path.join(ambience_dir, filename)
                    name_without_ext = os.path.splitext(filename)[0]
                    with _wave.open(wav_path, 'rb') as wf:
                        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 8000:
                            logger.warning(f"Skipping {filename}: Must be mono, 16-bit, 8kHz")
                            continue
                        AMBIENT_TRACKS[name_without_ext] = wf.readframes(wf.getnframes())
            logger.info(f"Loaded {len(AMBIENT_TRACKS)} ambient track WAVs from {ambience_dir}")
        
        ct_path = os.path.join(os.path.dirname(__file__), "..", "static", "audio", "comfort_tone.wav")
        ct_path = os.path.abspath(ct_path)
        if os.path.exists(ct_path):
            with _wave.open(ct_path, 'rb') as wf:
                if wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == 8000:
                    global COMFORT_TONE_PCM
                    COMFORT_TONE_PCM = wf.readframes(wf.getnframes())
                    logger.info(f"Loaded comfort tone: {len(COMFORT_TONE_PCM)} bytes")
                    
    except Exception as e:
        logger.error(f"Failed to load ambient track WAVs: {e}")

_load_ambient_wavs()

class ContinuousAudioMixer:
    def __init__(self, websocket: WebSocket, stream_sid: str, session_id: str, survey_config: dict):
        self.websocket = websocket
        self.stream_sid = stream_sid
        self.session_id = session_id
        
        self.speech_pcm_buffer = bytearray()
        
        persona = survey_config.get("persona", {}) if survey_config else {}
        ambience_name = persona.get("noise_type") or (survey_config.get("noise_type") if survey_config else None) or "crowded_room"
        self.ambient_pcm = AMBIENT_TRACKS.get(ambience_name) if ambience_name else None
        
        try:
            raw_vol = persona.get("volume", survey_config.get("volume", 0.7)) if survey_config else 0.7
            self.volume = float(raw_vol)
        except:
            self.volume = 0.7
            
        self.cursor = 0
        if self.ambient_pcm and len(self.ambient_pcm) > 320:
            self.cursor = random.randint(0, (len(self.ambient_pcm) - 320) // 2) * 2
            logger.info(f"ContinuousAudioMixer initialized with ambience '{ambience_name}', volume {self.volume}")
        else:
            logger.info(f"ContinuousAudioMixer initialized without ambience (name='{ambience_name}', found={bool(self.ambient_pcm)})")

        self.running = False
        self.timing_chunks = 0
        self.media_sequence = 0

    def put_speech_pcm(self, pcm_data: bytes):
        if session_states.get(self.session_id) != "INTERRUPTED":
            self.speech_pcm_buffer.extend(pcm_data)

    def clear_speech(self):
        self.speech_pcm_buffer.clear()
        
    async def wait_for_speech_completion(self):
        while len(self.speech_pcm_buffer) > 0 and self.running and session_states.get(self.session_id) != "INTERRUPTED":
            if 0 < len(self.speech_pcm_buffer) < 320:
                self.speech_pcm_buffer.extend(b"\x00" * (320 - len(self.speech_pcm_buffer)))
            await asyncio.sleep(0.05)

    async def start(self):
        self.running = True
        self.task = asyncio.create_task(self._mixer_loop())

    async def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def _mixer_loop(self):
        import time
        start_time = time.perf_counter()
        while self.running:
            try:
                media_msg = self.generate_media_message(chunk_size_pcm=320)
                await self.websocket.send_json(media_msg)
                
                target_elapsed = self.timing_chunks * 0.020
                actual_elapsed = time.perf_counter() - start_time
                delay = target_elapsed - actual_elapsed
                
                if delay < -0.25:
                    # We fell behind significantly (e.g. event loop blocked >250ms).
                    # Do NOT blast massive amounts of packets to catch up, as this overflows the PSTN jitter buffer.
                    # Reset timing anchors to NOW.
                    start_time = time.perf_counter()
                    self.timing_chunks = 0
                    delay = 0
                
                self.timing_chunks += 1
                
                if delay > 0.002:
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(0)
                    
            except asyncio.CancelledError:
                break
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                if "Cannot call" in str(e) or "close message" in str(e).lower():
                    break
                logger.error(f"Runtime error in mixer loop: {e}")
                break
            except Exception as e:
                # Log cleanly and continue
                if "websocket.send" not in str(e) and "close message" not in str(e).lower():
                    logger.error(f"Error in mixer loop: {e}")
                else:
                    break
                await asyncio.sleep(0.020)

    def generate_media_message(self, chunk_size_pcm: int = 320):
        if session_states.get(self.session_id) == "INTERRUPTED":
            self.clear_speech()

        speech_chunk = None
        if len(self.speech_pcm_buffer) >= chunk_size_pcm:
            speech_chunk = bytes(self.speech_pcm_buffer[:chunk_size_pcm])
            del self.speech_pcm_buffer[:chunk_size_pcm]

        if self.ambient_pcm:
            take = min(chunk_size_pcm, len(self.ambient_pcm) - self.cursor)
            amb_chunk = self.ambient_pcm[self.cursor:self.cursor+take]
            self.cursor += take
            if self.cursor >= len(self.ambient_pcm):
                self.cursor = 0
            if len(amb_chunk) < chunk_size_pcm:
                amb_chunk += self.ambient_pcm[:chunk_size_pcm - len(amb_chunk)]
                self.cursor = chunk_size_pcm - len(amb_chunk)
                
            import audioop
            base_vol = self.volume * 10.0
            current_vol = base_vol * 0.4 if speech_chunk else base_vol
            amb_chunk = audioop.mul(amb_chunk, 2, current_vol)
            
            if speech_chunk:
                final_chunk = audioop.add(speech_chunk, amb_chunk, 2)
            else:
                final_chunk = amb_chunk
        else:
            if speech_chunk:
                final_chunk = speech_chunk
            else:
                final_chunk = b"\x00" * chunk_size_pcm

        from app.services.telecommunication_service import pcm_8k_to_mulaw
        import base64
        mu_chunk = pcm_8k_to_mulaw(final_chunk)
        payload_b64 = base64.b64encode(mu_chunk).decode("ascii")
        
        media_msg = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {
                "payload": payload_b64,
                "chunk": str(self.media_sequence + 1),
                "track": "outbound",
                "streamSid": self.stream_sid
            }
        }
        self.media_sequence += 1
        return media_msg



async def get_greeting_mulaw(survey_config: dict, survey_id: str, greeting_text: str, voice_id: str = None, model_id: str = None, session_id: str = None) -> bytes:
    """
    Retrieves G.711 mu-law greeting audio with 0ms latency.
    Order of preference:
    1. In-memory cache (AUDIO_MULAW_CACHE)
    2. Pre-cached base64 mu-law in survey_config (greetingAudioMulaw)
    3. Cloudinary audio URL (greetingAudioUrl)
    4. Live ElevenLabs TTS synthesis
    """
    if survey_id and survey_id in AUDIO_MULAW_CACHE:
        logger.info(f"Using in-memory cached mu-law greeting audio for survey_id '{survey_id}' (0ms delay)")
        return AUDIO_MULAW_CACHE[survey_id]

    greeting_audio_mulaw_b64 = survey_config.get("greetingAudioMulaw") if survey_config else None
    greeting_audio_url = survey_config.get("greetingAudioUrl") if survey_config else None
    from app.services.telecommunication_service import downsample_pcm_24k_to_8k, pcm_8k_to_mulaw

    greeting_mulaw = None
    if greeting_audio_mulaw_b64:
        try:
            logger.info(f"Using pre-cached mu-law greeting audio from survey config")
            greeting_mulaw = base64.b64decode(greeting_audio_mulaw_b64)
        except Exception as b64_err:
            logger.warning(f"Failed to decode greetingAudioMulaw: {b64_err}. Falling back to Cloudinary/live.")
            greeting_mulaw = None

    if not greeting_mulaw and greeting_audio_url:
        try:
            logger.info(f"Downloading cached greeting audio from Cloudinary: {greeting_audio_url}")
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=5.0) as dl_client:
                dl_resp = await dl_client.get(greeting_audio_url)
                if dl_resp.status_code == 200:
                    wav_bytes = dl_resp.content
                    greeting_pcm_24k = wav_bytes[44:] if len(wav_bytes) > 44 else wav_bytes
                    loop = asyncio.get_running_loop()
                    greeting_pcm_8k = await loop.run_in_executor(None, downsample_pcm_24k_to_8k, greeting_pcm_24k)
                    greeting_mulaw = await loop.run_in_executor(None, pcm_8k_to_mulaw, greeting_pcm_8k)
                    logger.info(f"Cached greeting audio loaded successfully ({len(greeting_mulaw)} bytes mu-law)")
                else:
                    raise RuntimeError(f"Download status {dl_resp.status_code}")
        except Exception as cache_err:
            logger.warning(f"Failed to use cached greeting audio: {cache_err}. Falling back to live TTS.")
            greeting_mulaw = None

    if not greeting_mulaw:
        logger.info(f"Synthesizing live greeting audio via TTS for text: '{greeting_text[:30]}...'")
        tts_provider = survey_config.get("tts_provider") if survey_config else None
        greeting_pcm_24k = await ai_service.synthesize_speech_pcm(
            greeting_text, provider=tts_provider, voice=voice_id, model_id=model_id, session_id=session_id, survey_config=survey_config
        )
        loop = asyncio.get_running_loop()
        greeting_pcm_8k = await loop.run_in_executor(None, downsample_pcm_24k_to_8k, greeting_pcm_24k)
        greeting_mulaw = await loop.run_in_executor(None, pcm_8k_to_mulaw, greeting_pcm_8k)

    if survey_id and greeting_mulaw:
        AUDIO_MULAW_CACHE[survey_id] = greeting_mulaw

    return greeting_mulaw

async def stream_mulaw_to_smartflo(websocket: WebSocket, stream_sid: str, mulaw_bytes: bytes, session_id: str = None) -> bool:
    """
    Streams raw G.711 mu-law audio bytes back to Smartflo in standard 20ms chunks (160 bytes).
    Optimized for Smartflo's 160-byte RTP frame size with 3-chunk initial burst and 19ms pacing
    to prevent audio choppiness, breakiness, or underflow.
    Supports barge-in: if session state changes to INTERRUPTED, stops streaming immediately.
    """
    if not mulaw_bytes:
        return True

    import sys
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass

    # Standard telephony frame size: 20ms of G.711 mu-law at 8kHz (160 bytes)
    chunk_size = 160
    frame_duration = 0.019  # 19ms target pacing per 20ms chunk to keep buffer fed
    
    num_chunks = (len(mulaw_bytes) + chunk_size - 1) // chunk_size
    logger.info(f"Streaming {len(mulaw_bytes)} bytes of mu-law audio to Smartflo ({num_chunks} chunks)...")
    
    # Send initial burst of 5 chunks (100ms of audio) without delay to fill telephony jitter buffer
    BURST_CHUNKS = 5
    start_time = time.perf_counter()
    
    for i in range(num_chunks):
        if session_id and session_states.get(session_id) == "INTERRUPTED":
            logger.info(f"Barge-in detected at chunk {i}/{num_chunks}. Stopping AI audio playback.")
            return False
        
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(mulaw_bytes))
        chunk_data = mulaw_bytes[start_idx:end_idx]
        
        if len(chunk_data) < chunk_size:
            chunk_data = chunk_data + b'\xff' * (chunk_size - len(chunk_data))  # \xff is silence in mu-law
            
        payload_b64 = base64.b64encode(chunk_data).decode("utf-8")
        
        message = {
            "event": "media",
            "streamSid": stream_sid,
            "stream_sid": stream_sid,
            "media": {
                "payload": payload_b64,
                "chunk": str(i + 1),
                "streamSid": stream_sid,
                "track": "outbound"
            }
        }
        
        try:
            await websocket.send_json(message)
        except (WebSocketDisconnect, RuntimeError) as ws_err:
            logger.info(f"WebSocket disconnected during audio streaming to streamSid {stream_sid}.")
            return False
        except Exception as send_err:
            logger.warning(f"Failed to send audio chunk to streamSid {stream_sid}: {send_err}")
            return False
        
        # Pacing: Send initial burst instantly, then align with real-time clock
        if i >= BURST_CHUNKS:
            target_elapsed = (i - BURST_CHUNKS + 1) * frame_duration
            actual_elapsed = time.perf_counter() - start_time
            sleep_needed = target_elapsed - actual_elapsed
            if sleep_needed > 0:
                await asyncio.sleep(sleep_needed)
        else:
            await asyncio.sleep(0.001)
            
    return True

async def handle_customer_audio(
    websocket: WebSocket,
    stream_sid: str,
    session_id: str,
    pcm_bytes: bytes,
    survey_config: dict = None,
    session_history: list = None,
    extracted_data: dict = None,
    customer_number: str = None,
    call_sid: str = None,
    mixer = None
):
    """
    Transcribes customer voice PCM bytes, generates streamed AI sentences,
    synthesizes speech in parallel, and streams it back.
    Runs data extraction in the background (fire-and-forget).
    """
    lock = session_locks.setdefault(session_id, asyncio.Lock())
    await lock.acquire()
    try:
        global COMFORT_TONE_PCM
        if COMFORT_TONE_PCM and mixer:
            mixer.put_speech_pcm(COMFORT_TONE_PCM)
        t_start = time.perf_counter()
        from app.services.telecommunication_service import pcm_8k_to_wav, downsample_pcm_24k_to_8k, pcm_8k_to_mulaw
        
        # 1. Convert 8kHz PCM to a valid WAV file for transcription
        user_transcript = ""
        if len(pcm_bytes) == 0:
            user_transcript = "[Silence] The user did not respond. Gracefully ask them if they are still there or repeat the question."
            logger.info("[STT] PCM bytes empty. Injecting silence reprompt.")
        else:
            loop = asyncio.get_running_loop()
            wav_audio = await loop.run_in_executor(None, pcm_8k_to_wav, pcm_bytes)
            
            # 2. Transcribe audio - no DB reads during gather!
            stt_provider = survey_config.get("stt_provider") if survey_config else None
            stt_model = survey_config.get("stt_model") if survey_config else None
            survey_lang = survey_config.get("language") if survey_config else None
            
            logger.info(f"[STT-START] Sending {len(wav_audio)} bytes of audio to STT ({stt_provider}:{stt_model})...")
            t_stt = time.perf_counter()
            user_transcript = await ai_service.transcribe_audio(
                wav_audio, "audio/wav", provider=stt_provider, model=stt_model, language=survey_lang
            )
            t_stt_done = time.perf_counter()
            logger.info(f"[STT-END] Transcription complete. Text: '{user_transcript}' (Time: {(t_stt_done - t_stt)*1000:.0f}ms)")
        
        survey_id = survey_config.get("survey_id", "default") if survey_config else "default"
        
        # 3. Fallback compatibility for history and extracted data if None
        if session_history is None or extracted_data is None:
            results_col = get_collection("survey_results")
            sessions_col = get_collection("survey_sessions")
            prev_result, session_doc = await asyncio.gather(
                results_col.find_one({"session_id": session_id}),
                sessions_col.find_one({"session_id": session_id})
            )
            if session_history is None:
                session_history = session_doc.get("history", []) if session_doc else []
            if extracted_data is None:
                extracted_data = prev_result["extracted_data"] if prev_result else {}
            if not customer_number and session_doc:
                customer_number = session_doc.get("customer_number")
            if not call_sid and session_doc:
                call_sid = session_doc.get("call_sid")
                
        clean_transcript = user_transcript.strip()
        if (
            not clean_transcript 
            or clean_transcript == "[Silence]" 
            or (clean_transcript.startswith("[") and clean_transcript.endswith("]"))
        ):
            logger.info(f"Customer transcript '{user_transcript}' contains only non-speech events. No response triggered.")
            session_states[session_id] = "LISTENING"
            return
            
        # 4. Build history (append current user turn to context history)
        history_formatted = [
            {"speaker": t["speaker"], "text_content": t["text_content"]}
            for t in session_history
        ]
        history_formatted.append({"speaker": "CUSTOMER", "text_content": user_transcript})
        
        # Check if customer transcript has explicit hangup/disconnect intent
        disconnect_keywords = [
            "disconnect", "cut the call", "cut call", "phone kaat", "phone kaato", 
            "kaat do", "stop", "end call", "talk later", "baat nahi karni", 
            "nahin baat", "band karo"
        ]
        customer_lower = clean_transcript.lower()
        customer_wants_disconnect = any(kw in customer_lower for kw in disconnect_keywords)
        
        # --- PASS 1: STRUCTURED INTENT EXTRACTION (BACKGROUND) ---
        survey_steps = survey_config.get("survey_steps", []) if survey_config else []
        current_step_dict = survey_rules.get_current_step_dict(extracted_data, survey_steps=survey_steps)
        current_field = current_step_dict.get("field") if current_step_dict else None
        
        # Fire-and-forget Pass 1 so it doesn't block Time-To-First-Audio (TTFA).
        # This saves 1-1.5 seconds of latency per turn!
        async def _bg_pass1_extraction(user_text, step_dict, field_name):
            try:
                t_extract_start = time.perf_counter()
                intent_result = await ai_service.extract_user_intent_structured(
                    user_transcript=user_text,
                    current_step=step_dict,
                    survey_config=survey_config
                )
                t_extract = time.perf_counter() - t_extract_start
                
                state_val = intent_result.get("state", "unclear")
                extracted_val = intent_result.get("value", "null")
                
                retry_key = f"__retry_{field_name}"
                retry_count = extracted_data.get(retry_key, 0)
                
                if state_val == "valid" and extracted_val != "null":
                    logger.info(f"Pass 1 Extraction (BG) [{t_extract:.2f}s]: {field_name} = {extracted_val}")
                    extracted_data[field_name] = extracted_val
                    extracted_data[retry_key] = 0
                else:
                    retry_count += 1
                    extracted_data[retry_key] = retry_count
                    
                    if retry_count >= 2:
                        logger.info(f"Pass 1 Extraction (BG): {field_name} failed 2 retries. Forcing forward.")
                        extracted_data[field_name] = user_text.strip() if user_text and user_text.strip() else "Skipped (Unclear)"
                    else:
                        logger.info(f"Pass 1 Extraction (BG) [{t_extract:.2f}s]: {field_name} = Unclear (Retry {retry_count}/2)")
                        extracted_data[field_name] = "Unclear"
            except Exception as e:
                logger.error(f"Error in background Pass 1 extraction: {e}")
                
        if current_step_dict and not customer_wants_disconnect:
            retry_key = f"__retry_{current_field}"
            retry_count = extracted_data.get(retry_key, 0)
            
            # We must await it SYNCHRONOUSLY so the state updates BEFORE Pass 2 runs.
            # This guarantees the AI is physically prevented from asking the question a 2nd time unnecessarily.
            await _bg_pass1_extraction(clean_transcript, current_step_dict, current_field)
        
        # Note: is_complete is checked instantly before Pass 1 finishes, so it relies on the PREVIOUS turn's state.
        # But should_hangup is checked AGAIN at the end of the turn (line ~830), where Pass 1 WILL be finished,
        # ensuring the call still hangs up gracefully!
        is_complete = survey_rules.is_survey_complete(extracted_data, survey_steps=survey_steps)
        current_step_post = survey_rules.get_current_step_dict(extracted_data, survey_steps=survey_steps)
        
        lang = survey_config.get("language", "hi") if survey_config else "hi"
        prompts_obj = survey_config.get("prompts") or {} if survey_config else {}
        custom_farewell = (survey_config.get("farewell_message") or prompts_obj.get("farewell") or "").strip() if survey_config else ""
        
        voice_id = survey_config.get("tts_voice_id") or settings.ELEVENLABS_VOICE_ID if survey_config else settings.ELEVENLABS_VOICE_ID
        model_id = survey_config.get("tts_model_id") or settings.ELEVENLABS_MODEL_ID if survey_config else settings.ELEVENLABS_MODEL_ID
        tts_provider = survey_config.get("tts_provider") if survey_config else None
        survey_lang = survey_config.get("language") if survey_config else None
        tts_speed = survey_config.get("tts_speed") if survey_config else None
        
        ai_text = None
        should_hangup = False
        
        if is_complete or customer_wants_disconnect or (current_step_post and current_step_post.get("field") == "__farewell_acked"):
            should_hangup = True
            hangup_reason = f"is_complete={is_complete}, customer_wants_disconnect={customer_wants_disconnect}, farewell_acked={current_step_post.get('field') if current_step_post else None}"
            logger.info(f"[HANGUP-TRACE] Session {session_id} - EARLY hangup triggered. Reason: {hangup_reason}. extracted_data keys: {list(extracted_data.keys())}")
            if custom_farewell:
                ai_text = custom_farewell
            elif is_complete or (current_step_post and current_step_post.get("field") == "__farewell_acked"):
                if lang == "en":
                    ai_text = "Thank you for your responses and time. Goodbye!"
                else:
                    ai_text = "आपकी प्रतिक्रिया के लिए धन्यवाद। आपका समय देने के लिए शुक्रिया, अलविदा!"
            else:
                if lang == "en":
                    ai_text = "Okay, thank you. Goodbye!"
                else:
                    ai_text = "ठीक है, धन्यवाद। अलविदा!"
                    
        # 5. Overlapped Streaming Completion + TTS Pipeline
        full_ai_response_text = ""
        
        if ai_text:
            # Direct non-streaming synthesis of farewell to skip LLM entirely
            logger.info(f"Direct playing of farewell message: '{ai_text}'")
            pcm_24k = await ai_service.synthesize_speech_pcm(
                ai_text, 
                provider=tts_provider,
                voice=voice_id,
                model_id=model_id,
                language=survey_lang,
                speed=tts_speed
            )
            loop = asyncio.get_running_loop()
            pcm_8k = await loop.run_in_executor(None, downsample_pcm_24k_to_8k, pcm_24k)
            
            # Streaming cost tracking & DB save task
            async def _save_direct_farewell():
                try:
                    sessions_col = get_collection("survey_sessions")
                    results_col = get_collection("survey_results")
                    farewell_turns = [
                        {"speaker": "CUSTOMER", "text_content": user_transcript, "timestamp": datetime.utcnow()},
                        {"speaker": "SYSTEM", "text_content": ai_text, "timestamp": datetime.utcnow()}
                    ]
                    
                    save_payload = {
                        "session_id": session_id,
                        "survey_id": survey_id,
                        "extracted_data": survey_rules.clean_internal_fields(dict(extracted_data)),
                        "extracted_at": datetime.utcnow()
                    }
                    if customer_number:
                        save_payload["customer_number"] = customer_number
                    if call_sid:
                        save_payload["call_sid"] = call_sid
                        
                    await sessions_col.update_one(
                        {"session_id": session_id},
                        {"$push": {"history": {"$each": farewell_turns}}}
                    )
                    
                    # Store transcriptions and the final extracted_data in survey_results
                    await results_col.update_one(
                        {"session_id": session_id},
                        {
                            "$set": save_payload,
                            "$push": {"transcriptions": {"$each": farewell_turns}}
                        },
                        upsert=True
                    )
                    
                    # Also trigger translation for the final data
                    try:
                        translated_dict = await ai_service.translate_extracted_data(dict(extracted_data), session_id=session_id)
                        await results_col.update_one(
                            {"session_id": session_id},
                            {"$set": {"extracted_data": survey_rules.clean_internal_fields(translated_dict)}}
                        )
                    except Exception as trans_ex:
                        logger.error(f"Error in direct farewell translation: {trans_ex}")
                        
                except Exception as ex:
                    logger.error(f"Error in direct farewell save: {ex}")
            save_task = asyncio.create_task(_save_direct_farewell())
            
            session_states[session_id] = "SPEAKING"
            mixer.put_speech_pcm(pcm_8k)
            await mixer.wait_for_speech_completion()
            fully_played = session_states.get(session_id) != "INTERRUPTED"
            await save_task
            
        else:
            state_context = survey_rules.get_survey_prompt_context(extracted_data, survey_steps=survey_steps)
            
            # Fetch previous filler from session states or default memory if available
            last_filler = session_states.get(f"{session_id}_last_filler")
            
            t_llm = time.perf_counter()
            response_stream, target_model, target_provider = await ai_service.generate_response_stream(
                history=history_formatted,
                state_context=state_context,
                survey_id=survey_id,
                survey_config=survey_config,
                last_filler_message=last_filler
            )
            
            # Clean sentence stream segmenter: splits cleanly on natural phrase boundaries to preserve TTS prosody
            async def sentence_stream(stream):
                buffer = ""
                sentence_endings = {'.', '!', '?', '।', '\n'}
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    content = delta.content
                    if content:
                        for char in content:
                            buffer += char
                            if char in sentence_endings:
                                sentence = buffer.strip()
                                if sentence:
                                    yield sanitize_tts_text(sentence)
                                buffer = ""
                sentence = buffer.strip()
                if sentence:
                    yield sanitize_tts_text(sentence)

            session_states[session_id] = "SPEAKING"
            fully_played = True
            sentence_idx = 0
            
            queue = asyncio.Queue()
            save_task = None
            
            # Fire-and-forget: Save historical logs and extracted parameters in the background
            async def _bg_save(system_text: str):
                try:
                    # Update database collections in background

                    results_col = get_collection("survey_results")
                    sessions_col = get_collection("survey_sessions")
                    
                    new_turns = [
                        {"speaker": "CUSTOMER", "text_content": user_transcript, "timestamp": datetime.utcnow()},
                        {"speaker": "SYSTEM", "text_content": system_text, "timestamp": datetime.utcnow()}
                    ]
                    
                    save_payload = {
                        "session_id": session_id,
                        "survey_id": survey_id,
                        "extracted_data": survey_rules.clean_internal_fields(dict(extracted_data)),
                        "extracted_at": datetime.utcnow()
                    }
                    if customer_number:
                        save_payload["customer_number"] = customer_number
                    if call_sid:
                        save_payload["call_sid"] = call_sid
                        
                    # Save extracted data and push transcription turns atomically
                    await results_col.update_one(
                        {"session_id": session_id},
                        {
                            "$set": save_payload,
                            "$push": {"transcriptions": {"$each": new_turns}}
                        },
                        upsert=True
                    )
                    
                    await sessions_col.update_one(
                        {"session_id": session_id},
                        {"$push": {"history": {"$each": new_turns}}}
                    )
                except Exception as ex:
                    logger.error(f"Error in background extraction/save: {ex}")
                    
                # Also translate extracted data in the background
                try:
                    translated_dict = await ai_service.translate_extracted_data(dict(extracted_data), session_id=session_id)
                    
                    # Update in-memory extracted_data so it doesn't get re-translated every turn
                    for k, v in translated_dict.items():
                        extracted_data[k] = v
                        
                    await get_collection("survey_results").update_one(
                        {"session_id": session_id},
                        {"$set": {"extracted_data": survey_rules.clean_internal_fields(translated_dict)}}
                    )
                except Exception as ex:
                    logger.error(f"Error in background translation: {ex}")

            # Telephony cost tracking background task
            async def _bg_track_cost(system_text: str):
                try:
                    from app.services.cost_service import track_turn_cost
                    prompt_toks = len(state_context.split()) + len(user_transcript.split()) + 100
                    comp_toks = len(system_text.split()) + 10
                    await track_turn_cost(
                        session_id=session_id,
                        turn_type="TELEPHONY_TURN",
                        stt_bytes=wav_audio,
                        tts_text=system_text,
                        llm_prompt_tokens=prompt_toks,
                        llm_completion_tokens=comp_toks,
                        llm_model=target_model
                    )
                except Exception as ex:
                    logger.error(f"Failed to track telephony cost: {ex}")
            
            # Producer: Consumes LLM stream, creates background synthesis tasks, and puts them into the queue
            async def producer_task():
                nonlocal should_hangup
                try:
                    async for sentence in sentence_stream(response_stream):
                        # Stop producing if consumer was interrupted by barge-in
                        if session_states.get(session_id) == "INTERRUPTED":
                            logger.info("[PRODUCER] Barge-in detected. Stopping LLM stream consumption.")
                            break
                            
                        clean_sent = sentence.strip()
                        if not clean_sent:
                            continue
                            
                        if "[END_CALL]" in clean_sent:
                            should_hangup = True
                            logger.info(f"[HANGUP-TRACE] Session {session_id} - [END_CALL] tag found in LLM output. Sentence: '{clean_sent[:80]}'")
                            clean_sent = clean_sent.replace("[END_CALL]", "").strip()
                            if not clean_sent:
                                continue
                        
                        # Wrapper to handle full synthesis pipeline for each chunk in the background
                        async def tts_task_wrapper(text):
                            pcm_24k = await ai_service.synthesize_speech_pcm(
                                text,
                                provider=tts_provider,
                                voice=voice_id,
                                model_id=model_id,
                                language=survey_lang,
                                speed=tts_speed,
                                survey_config=survey_config
                            )
                            loop = asyncio.get_running_loop()
                            pcm_8k = await loop.run_in_executor(None, downsample_pcm_24k_to_8k, pcm_24k)
                            return pcm_8k
                            
                        # Spawn background synthesis task immediately
                        task = asyncio.create_task(tts_task_wrapper(clean_sent))
                        await queue.put((clean_sent, task))
                        
                    await queue.put(None)  # Sentinel
                except asyncio.CancelledError:
                    logger.info("[PRODUCER] Cancelled due to barge-in.")
                    await queue.put(None)  # Ensure consumer exits
                except Exception as prod_err:
                    logger.error(f"Error in LLM stream producer: {prod_err}")
                    await queue.put(None)

            prod_task = asyncio.create_task(producer_task())
            
            # Consumer: plays sentences from queue
            while True:
                item = await queue.get()
                if item is None:
                    break
                    
                sentence_text, task = item
                
                # Skip TTS synthesis entirely if already interrupted (avoid wasting API calls)
                if session_states.get(session_id) == "INTERRUPTED":
                    logger.info("Barge-in interrupt detected before playback. Skipping remaining sentences.")
                    task.cancel()
                    fully_played = False
                    break
                    
                t_synth_wait = time.perf_counter()
                try:
                    # Race the TTS synthesis against an interrupt poller to abort instantly on barge-in
                    async def check_interrupt():
                        while session_states.get(session_id) != "INTERRUPTED":
                            await asyncio.sleep(0.1)
                    
                    interrupt_task = asyncio.create_task(check_interrupt())
                    done, pending = await asyncio.wait(
                        [task, interrupt_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    if task not in done:
                        # Barge-in occurred during synthesis!
                        logger.info(f"Barge-in interrupt detected DURING synthesis of sentence {sentence_idx+1}. Aborting.")
                        task.cancel()
                        fully_played = False
                        break
                    
                    # Synthesis finished first
                    interrupt_task.cancel()
                    pcm_bytes_sent = task.result()
                    
                    synth_wait_ms = (time.perf_counter() - t_synth_wait) * 1000
                    logger.info(f"Synthesized sentence {sentence_idx+1} in background (awaited {synth_wait_ms:.0f}ms): '{sentence_text[:40]}...' ({len(pcm_bytes_sent)} bytes PCM)")
                except Exception as synth_err:
                    logger.error(f"TTS synthesis failed for sentence: '{sentence_text}': {synth_err}")
                    continue
                    
                if sentence_idx == 0:
                    logger.info(f"[PERF] Time-to-first-audio: {(time.perf_counter() - t_start)*1000:.0f}ms")
                    mixer.clear_speech()
                    try:
                        await websocket.send_json({"event": "clear", "streamSid": stream_sid})
                        logger.info("Sent 'clear' event to flush provider buffer")
                    except Exception as e:
                        logger.warning(f"Failed to send clear event: {e}")
                    
                mixer.put_speech_pcm(pcm_bytes_sent)
                
                # Since mixer is fully asynchronous, we need to manually yield to let the loop run
                # wait_for_speech_completion is needed only at the END of the response, or we can just 
                # wait for it to finish playing this chunk before proceeding to the next one to allow barge-in
                while len(mixer.speech_pcm_buffer) > 16000 and session_states.get(session_id) != "INTERRUPTED":
                    # Pause feeding the queue if it gets too large (e.g. 1s of audio) to preserve barge-in latency
                    await asyncio.sleep(0.1)

                if session_states.get(session_id) == "INTERRUPTED":
                    # Include the sentence that was being played (user heard at least part of it)
                    full_ai_response_text += sentence_text + " "
                    fully_played = False
                    break
                    
                full_ai_response_text += sentence_text + " "
                sentence_idx += 1
            
            # Cancel producer if still running (e.g. LLM still streaming after barge-in)
            if not prod_task.done():
                prod_task.cancel()
                try:
                    await prod_task
                except asyncio.CancelledError:
                    pass
                
        # Determine actual spoken text (partial on barge-in, full otherwise)
        # Note: For the direct farewell path (ai_text is set), saving is handled by _save_direct_farewell above
        if not ai_text:
            actual_spoken_text = full_ai_response_text.strip()
            
            # Save the spoken text as the last filler for context in the next turn
            session_states[f"{session_id}_last_filler"] = actual_spoken_text
            
            # Update in-memory session history with what was ACTUALLY spoken
            session_history.append({"speaker": "CUSTOMER", "text_content": user_transcript, "timestamp": datetime.utcnow()})
            if actual_spoken_text:
                session_history.append({"speaker": "SYSTEM", "text_content": actual_spoken_text, "timestamp": datetime.utcnow()})
            
            # Save to database with actual spoken text (not full LLM output)
            if actual_spoken_text or user_transcript:
                save_task = asyncio.create_task(_bg_save(actual_spoken_text))
                asyncio.create_task(_bg_track_cost(actual_spoken_text))
                if not fully_played:
                    logger.info(f"[BARGE-IN] Saved partial AI response ({len(actual_spoken_text)} chars) instead of full LLM output.")
                    # Inject system note to LLM history so it knows it was interrupted
                    session_history.append({
                        "speaker": "SYSTEM",
                        "text_content": "[System Note: The user interrupted your previous response mid-sentence.]",
                        "timestamp": datetime.utcnow()
                    })
                
        # Wait for all generated audio to finish playing
        if full_ai_response_text and session_states.get(session_id) != "INTERRUPTED":
            logger.info(f"[HANGUP-TRACE] Session {session_id} - Waiting for speech completion. should_hangup={should_hangup}")
            await mixer.wait_for_speech_completion()
            logger.info(f"[HANGUP-TRACE] Session {session_id} - Speech completion done. State now: {session_states.get(session_id)}")
        
        # save_task runs in the background to avoid blocking VAD and TTFA
        # (Removed await save_task to prevent SPEAKING state from lingering during translations)
            
        # 8. Determine final hangup check
        if not should_hangup:
            recheck_complete = survey_rules.is_survey_complete(extracted_data, survey_steps=survey_steps)
            if recheck_complete:
                logger.info(f"[HANGUP-TRACE] Session {session_id} - POST-PLAYBACK is_survey_complete=True. extracted_data: {dict(extracted_data)}")
                should_hangup = True
            
            if not should_hangup and custom_farewell and full_ai_response_text:
                import re
                clean_ai = re.sub(r'[^\w\s]', '', full_ai_response_text.strip().lower())
                clean_fw = re.sub(r'[^\w\s]', '', custom_farewell.strip().lower())
                if len(clean_fw) > 5 and clean_fw in clean_ai:
                    logger.info(f"[HANGUP-TRACE] Session {session_id} - FAREWELL TEXT MATCH detected. LLM said: '{full_ai_response_text[:100]}'. Forcing hangup.")
                    should_hangup = True
            elif should_hangup and custom_farewell:
                # Survey became complete this turn! Check if farewell was already spoken
                import re
                clean_ai = re.sub(r'[^\w\s]', '', full_ai_response_text.strip().lower())
                clean_fw = re.sub(r'[^\w\s]', '', custom_farewell.strip().lower())
                if len(clean_fw) > 5 and clean_fw not in clean_ai:
                    logger.info("Survey completed but farewell not spoken. Synthesizing farewell manually before hangup.")
                    try:
                        session_states[session_id] = "SPEAKING"
                        farewell_mulaw = AUDIO_MULAW_CACHE.get(f"{survey_id}:farewell")
                        
                        if farewell_mulaw:
                            import audioop
                            loop = asyncio.get_running_loop()
                            pcm_8k = await loop.run_in_executor(None, audioop.ulaw2lin, farewell_mulaw, 2)
                            logger.info("Injected pre-cached zero-latency farewell audio.")
                        else:
                            logger.info("Pre-cached farewell not found, synthesizing live...")
                            from app.services.telecommunication_service import downsample_pcm_24k_to_8k
                            pcm_24k = await ai_service.synthesize_speech_pcm(
                                custom_farewell,
                                provider=tts_provider,
                                voice=voice_id,
                                model_id=model_id,
                                language=survey_lang,
                                speed=tts_speed,
                                survey_config=survey_config
                            )
                            loop = asyncio.get_running_loop()
                            pcm_8k = await loop.run_in_executor(None, downsample_pcm_24k_to_8k, pcm_24k)
                            
                        if mixer:
                            mixer.put_speech_pcm(pcm_8k)
                            await mixer.wait_for_speech_completion()
                    except Exception as ex:
                        logger.error(f"Error synthesizing final farewell manually: {ex}")
        if should_hangup:
            logger.info(f"[HANGUP-TRACE] Session {session_id} - *** EXECUTING HANGUP ***. call_sid={call_sid}, stream_sid={stream_sid}")
            sessions_col = get_collection("survey_sessions")
            await sessions_col.update_one(
                {"session_id": session_id},
                {"$set": {"status": "COMPLETED"}}
            )
            logger.info(f"Session {session_id} - Survey completed. Closing connection.")
            await asyncio.sleep(3.0)
            
            try:
                stop_msg = {"event": "stop", "streamSid": stream_sid}
                await websocket.send_json(stop_msg)
                logger.info(f"[HANGUP-TRACE] Session {session_id} - Sent 'stop' event to Smartflo.")
            except Exception as stop_err:
                logger.info(f"[HANGUP-TRACE] Session {session_id} - Failed to send stop event: {stop_err}")
                
            if call_sid:
                from app.services.telecommunication_service import hangup_call
                logger.info(f"[HANGUP-TRACE] Session {session_id} - Calling hangup_call API for call_sid={call_sid}")
                await hangup_call(call_sid)
                
            try:
                await websocket.close()
            except Exception:
                pass
        elif not fully_played:
            logger.info("AI playback was interrupted. Switching to LISTENING.")
            session_states[session_id] = "LISTENING"
        else:
            await asyncio.sleep(0.1)
            session_states[session_id] = "LISTENING"
            
    except Exception as e:
        logger.error(f"[HANGUP-TRACE] Session {session_id} - UNHANDLED EXCEPTION in handle_customer_audio: {e}", exc_info=True)
        session_states[session_id] = "LISTENING"
    finally:
        if lock.locked():
            lock.release()


async def resolve_survey_id_for_call(start_data: dict, initial_message: dict, query_survey_id: str = None) -> Tuple[str, Optional[str]]:
    """
    Dynamically resolves survey_id and customer_number for an incoming Smartflo call stream.
    Checks query params, custom_identifier, call_sid, ref_id, and DB call_survey_mappings.
    """
    if query_survey_id and query_survey_id != "default":
        return query_survey_id, None

    start_obj = start_data if isinstance(start_data, dict) else {}
    init_obj = initial_message if isinstance(initial_message, dict) else {}

    # 1. Check customParameters / custom_identifier in start payload
    custom_params = (
        start_obj.get("customParameters") 
        or start_obj.get("custom_identifier") 
        or init_obj.get("custom_identifier") 
        or init_obj.get("customParameters")
    )
    if isinstance(custom_params, dict):
        s_id = custom_params.get("survey_id")
        c_num = custom_params.get("customer_number")
        if s_id:
            logger.info(f"Resolved survey_id '{s_id}' and customer '{c_num}' from customParameters/custom_identifier")
            return str(s_id), str(c_num) if c_num else None
    elif isinstance(custom_params, str) and custom_params.startswith("{"):
        try:
            parsed_ci = json.loads(custom_params)
            if isinstance(parsed_ci, dict) and parsed_ci.get("survey_id"):
                s_id = parsed_ci.get("survey_id")
                c_num = parsed_ci.get("customer_number")
                logger.info(f"Resolved survey_id '{s_id}' and customer '{c_num}' from parsed customParameters")
                return str(s_id), str(c_num) if c_num else None
        except Exception:
            pass

    # 2. Check call_survey_mappings DB
    call_sid = start_obj.get("callSid") or init_obj.get("callSid") or init_obj.get("streamSid")
    ref_id = start_obj.get("ref_id") or init_obj.get("ref_id")
    cust_num = start_obj.get("customerNumber") or start_obj.get("customer_number") or start_obj.get("from") or start_obj.get("to")

    try:
        from app.core.db import get_collection
        mappings_col = get_collection("call_survey_mappings")
        mapping = None

        if call_sid:
            mapping = await mappings_col.find_one({"call_sid": str(call_sid)})
        if not mapping and ref_id:
            mapping = await mappings_col.find_one({"ref_id": str(ref_id)})

        if not mapping and cust_num:
            clean_c = "".join(filter(str.isdigit, str(cust_num)))
            if clean_c:
                mapping = await mappings_col.find_one({"customer_number": clean_c}, sort=[("created_at", -1)])

        if not mapping:
            # Fallback to most recent call mapping created in last 15 minutes
            cutoff = datetime.utcnow() - timedelta(minutes=15)
            mapping = await mappings_col.find_one(
                {"created_at": {"$gte": cutoff}},
                sort=[("created_at", -1)]
            )

        if mapping and mapping.get("survey_id"):
            s_id = str(mapping["survey_id"])
            c_num = mapping.get("customer_number")
            logger.info(f"Resolved survey_id '{s_id}' via call_survey_mappings DB (callSid={call_sid}, customer={c_num})")
            return s_id, c_num
    except Exception as ex:
        logger.error(f"Error resolving survey_id from DB: {ex}")

    return "default", None

class StreamState:
    def __init__(self):
        self.stream_sid = None
        self.call_sid = None
        self.session_id = None
        self.survey_config = None
        self.customer_number = None
        self.session_history = []
        self.extracted_data = {}
        self.pcm_buffer = bytearray()
        self.silence_frames = 0
        self.is_speaking = False
        self.mixer = None
        self.barge_in_frames = 0
        self.audio_task = None
        # Packet-count based silence timeout (immune to event loop congestion)
        # Each packet = 20ms of real audio, so 250 packets = 5 seconds
        self.listening_packet_count = 0
        self.reprompt_count = 0
        self.previous_state = "LISTENING"
        self.max_duration_task = None

async def smartflo_websocket_handler(websocket: WebSocket, initial_message: dict, expected_call_id: str = None):
    logger.info(f"Starting Smartflo handler inside `/api/v1/survey/ws` (expected_call_id: {expected_call_id})...")
    
    if expected_call_id:
        try:
            sessions_col = get_collection("survey_sessions")
            session = await sessions_col.find_one({
                "$or": [{"session_id": expected_call_id}, {"call_sid": expected_call_id}]
            })
            if session and session.get("status") == "COMPLETED":
                logger.warning(f"Connection rejected: Session/Call {expected_call_id} is already completed.")
                try:
                    await websocket.send_json({"type": "error", "message": "Call already completed"})
                    await websocket.close(code=4000)
                except Exception:
                    pass
                return
        except Exception as e:
            logger.error(f"Error checking expected_call_id status: {e}")
    
    # Map of stream_sid -> StreamState
    streams = {}

    async def finalize_stream(stream_sid: str):
        stream = streams.get(stream_sid)
        if not stream: return
        logger.info(f"[HANGUP-TRACE] finalize_stream called for stream_sid={stream_sid}, session_id={stream.session_id}, customer={stream.customer_number}")
        if stream.mixer:
            await stream.mixer.stop()
        if stream.max_duration_task:
            stream.max_duration_task.cancel()
        
        session_id = stream.session_id
        if session_id:
            try:
                sessions_col = get_collection("survey_sessions")
                await sessions_col.update_one(
                    {"session_id": session_id, "status": {"$ne": "COMPLETED"}},
                    {"$set": {"status": "COMPLETED", "ended_at": datetime.utcnow()}}
                )
                session = await sessions_col.find_one({"session_id": session_id})
                if session:
                    total_cost = session.get("total_cost", 0.0)
                    total_cost_inr = total_cost * getattr(settings, "USD_TO_INR_RATE", 83.0)
                    logger.info(f"Session {session_id} - Call ended. Total accumulated cost: ${total_cost:.6f} (~Rs. {total_cost_inr:.4f})")
            except Exception as ex:
                logger.error(f"Failed to finalize session status on WebSocket close: {ex}")
            
            try:
                survey_id_val = None
                try:
                    sess_doc = await get_collection("survey_sessions").find_one({"session_id": session_id})
                    survey_id_val = sess_doc.get("survey_id") if sess_doc else None
                except Exception:
                    pass
                
                if survey_id_val:
                    # Clean AUDIO_MULAW_CACHE entries for this survey
                    mulaw_keys_to_remove = [k for k in AUDIO_MULAW_CACHE if k == survey_id_val or k.startswith(f"{survey_id_val}:")]
                    for k in mulaw_keys_to_remove:
                        del AUDIO_MULAW_CACHE[k]
                    
                    # Clean TTS_PCM_CACHE only if it grows too large
                    from app.services.ai_service import TTS_PCM_CACHE
                    if len(TTS_PCM_CACHE) > 1000:
                        TTS_PCM_CACHE.clear()
                        logger.info(f"Cleared TTS_PCM_CACHE because size exceeded 1000 entries")
                    
                    if mulaw_keys_to_remove:
                        logger.info(f"Cleaned up {len(mulaw_keys_to_remove)} AUDIO_MULAW_CACHE entries for survey {survey_id_val}")
            except Exception as cache_err:
                logger.warning(f"Failed to clean up audio caches: {cache_err}")
                
        # Remove from active streams
        del streams[stream_sid]

    async def handle_start_event(data_msg: dict):
        start_data = data_msg.get("start", {}) if isinstance(data_msg.get("start"), dict) else {}
        stream_sid = data_msg.get("streamSid") or start_data.get("streamSid")
        call_sid = start_data.get("callSid") or data_msg.get("callSid")
        
        if not stream_sid:
            logger.warning("Start event received with no streamSid.")
            return
            
        if stream_sid in streams:
            logger.info(f"Stream {stream_sid} already started.")
            return

        stream = StreamState()
        stream.stream_sid = stream_sid
        stream.call_sid = call_sid
        stream.session_id = str(uuid.uuid4())
        streams[stream_sid] = stream

        logger.info(f"Smartflo Stream started. StreamSid: {stream.stream_sid}, CallSid: {stream.call_sid}, SessionId: {stream.session_id}")

        survey_id, customer_number = await resolve_survey_id_for_call(start_data, data_msg, websocket.query_params.get("survey_id"))
        stream.customer_number = customer_number
        
        # Concurrency safety: Prevent duplicate AI sessions if Smartflo AMD/Voicemail triggers two streams
        if stream.customer_number:
            sessions_col = get_collection("survey_sessions")
            cutoff = datetime.utcnow() - timedelta(minutes=30)
            existing_session = await sessions_col.find_one({
                "customer_number": stream.customer_number,
                "status": "IN_PROGRESS",
                "created_at": {"$gte": cutoff}
            })
            
            if existing_session:
                logger.warning(f"Rejecting duplicate WebSocket connection for customer {stream.customer_number}. Active session {existing_session.get('session_id')} already exists!")
                del streams[stream_sid]
                try:
                    await websocket.send_json({"type": "error", "message": "Duplicate active session detected for this customer"})
                    await websocket.close(code=4000)
                except Exception:
                    pass
                return

        stream.survey_config = await get_survey_config(survey_id)
        survey_config = stream.survey_config

        v_id = survey_config.get("tts_voice_id")
        v_name = survey_config.get("tts_voice_name")
        v_disp = f"{v_id} ({v_name})" if v_name and v_name != v_id else v_id

        logger.info(
            f"\n============================================================\n"
            f" CALL SESSION STARTED ({stream.session_id}) | SURVEY ({survey_id})\n"
            f"============================================================\n"
            f" • STT Provider : {survey_config.get('stt_provider')} (Model: {survey_config.get('stt_model')})\n"
            f" • TTS Provider : {survey_config.get('tts_provider')} (Voice: {v_disp}, Model: {survey_config.get('tts_model_id')})\n"
            f" • LLM Provider : {survey_config.get('llm_provider')} (Model: {survey_config.get('chat_model')})\n"
            f" • Language     : {survey_config.get('language')}\n"
            f"============================================================"
        )

        greeting_text = survey_config.get("initial_greeting", "hello how are you?") if survey_config else "hello how are you?"
        voice_id = survey_config.get("tts_voice_id") if survey_config else None
        model_id = survey_config.get("tts_model_id") if survey_config else None

        stream.session_history = [
            {
                "speaker": "SYSTEM",
                "text_content": greeting_text,
                "timestamp": datetime.utcnow()
            }
        ]
        
        # Initialize session in MongoDB
        sessions_col = get_collection("survey_sessions")
        session_doc = {
            "session_id": stream.session_id,
            "survey_id": survey_id,
            "customer_number": stream.customer_number,
            "call_sid": stream.call_sid,
            "status": "IN_PROGRESS",
            "current_question_index": 0,
            "total_cost": 0.0,
            "cost_breakdown": [],
            "history": stream.session_history.copy(),
            "created_at": datetime.utcnow()
        }
        await sessions_col.insert_one(session_doc)
        
        # Initialize survey_results with call_sid and transcriptions
        results_col = get_collection("survey_results")
        initial_result_doc = {
            "session_id": stream.session_id,
            "survey_id": survey_id,
            "extracted_data": {},
            "extracted_at": datetime.utcnow(),
            "transcriptions": stream.session_history.copy()
        }
        if stream.customer_number:
            initial_result_doc["customer_number"] = stream.customer_number
        if stream.call_sid:
            initial_result_doc["call_sid"] = stream.call_sid
        await results_col.update_one(
            {"session_id": stream.session_id},
            {"$set": initial_result_doc},
            upsert=True
        )
        
        # Play greeting instantly using fast cache / pre-cached mu-law
        logger.info(f"Smartflo: Playing greeting text: '{greeting_text}' using voice: {voice_id}")
        greeting_mulaw = await get_greeting_mulaw(survey_config, survey_id, greeting_text, voice_id, model_id, stream.session_id)
        
        try:
            from app.services.cost_service import track_turn_cost
            await track_turn_cost(
                session_id=stream.session_id,
                turn_type="TELEPHONY_GREETING",
                tts_text=greeting_text
            )
        except Exception as e:
            logger.error(f"Failed to track telephony greeting cost: {e}")
            
        # Set session state to SPEAKING immediately to discard any early network noise / RTP packets
        session_states[stream.session_id] = "SPEAKING"

        # Initialize and start ContinuousAudioMixer
        stream.mixer = ContinuousAudioMixer(websocket, stream.stream_sid, stream.session_id, survey_config)
        await stream.mixer.start()

        persona = survey_config.get("persona", {}) if survey_config else {}
        max_duration_minutes = persona.get("maxCallDurationMinutes", 5)
        max_duration_seconds = max_duration_minutes * 60

        async def enforce_max_duration(s_sid: str, duration_sec: int):
            try:
                await asyncio.sleep(duration_sec)
                s = streams.get(s_sid)
                if s and session_states.get(s.session_id) != "COMPLETED":
                    logger.warning(f"[MAX-DURATION] Session {s.session_id} exceeded max call duration of {duration_sec}s. Hanging up forcefully.")
                    session_states[s.session_id] = "COMPLETED"
                    try:
                        if s.mixer:
                            s.mixer.clear_speech()
                        await websocket.send_json({"event": "stop", "streamSid": s.stream_sid})
                        logger.info(f"[HANGUP-TRACE] Session {s.session_id} - Sent 'stop' event due to max duration.")
                    except Exception:
                        pass
                    if s.call_sid:
                        from app.services.telecommunication_service import hangup_call
                        logger.info(f"[HANGUP-TRACE] Session {s.session_id} - Calling hangup_call API on max duration for call_sid={s.call_sid}")
                        asyncio.create_task(hangup_call(s.call_sid))
            except asyncio.CancelledError:
                pass

        stream.max_duration_task = asyncio.create_task(enforce_max_duration(stream.stream_sid, max_duration_seconds))

        # Play greeting in background so WebSocket receive loop starts immediately and drains/discards early echo
        async def play_greeting_bg(s, g_mulaw):
            try:
                await asyncio.sleep(0.2)  # Wait 2.0s for PSTN audio path to fully stabilize
                from app.services.telecommunication_service import mulaw_to_pcm_8k
                greeting_pcm = mulaw_to_pcm_8k(g_mulaw) if g_mulaw else b""
                if greeting_pcm and s.mixer:
                    s.mixer.put_speech_pcm(greeting_pcm)
                
                if s.mixer:
                    await s.mixer.wait_for_speech_completion()
                await asyncio.sleep(0.2)
                session_states[s.session_id] = "LISTENING"
                logger.info(f"Initial greeting playback completed. Session {s.session_id} state switched to LISTENING.")
            except Exception as play_err:
                logger.error(f"Error in background greeting playback: {play_err}")
                session_states[s.session_id] = "LISTENING"

        asyncio.create_task(play_greeting_bg(stream, greeting_mulaw))
    
    try:
        # Process initial_message if it contains start event
        event_type = initial_message.get("event")
        if event_type == "connected":
            logger.info("Smartflo handshake completed (connected event received).")
        elif event_type == "start":
            await handle_start_event(initial_message)
            
        while True:
            try:
                message = await websocket.receive_text()
            except RuntimeError as e:
                # Catch RuntimeError (e.g. "WebSocket is not connected") which happens if
                # another task closed the connection, and exit gracefully.
                logger.info(f"[HANGUP-TRACE] WebSocket RuntimeError (loop exit): {e}")
                break
            except WebSocketDisconnect as e:
                logger.info(f"[HANGUP-TRACE] WebSocket disconnected (loop exit): code={e.code if hasattr(e, 'code') else 'N/A'}, reason={e.reason if hasattr(e, 'reason') else e}")
                break
            except Exception as e:
                err_str = str(e)
                if "1005" in err_str or "1000" in err_str or "close" in err_str.lower():
                    logger.info(f"[HANGUP-TRACE] WebSocket closed gracefully (loop exit): {e}")
                else:
                    logger.error(f"[HANGUP-TRACE] WebSocket unexpected error (loop exit): {e}")
                break
                
            data = json.loads(message)
            event_type = data.get("event")
            
            if event_type == "connected":
                logger.info("Smartflo handshake completed (connected event received).")
            elif event_type == "start":
                await handle_start_event(data)
                
            elif event_type == "media":
                stream_sid = data.get("streamSid")
                if not stream_sid or stream_sid not in streams:
                    continue
                
                stream = streams[stream_sid]
                media_data = data.get("media", {})
                payload_b64 = media_data.get("payload")
                
                if payload_b64 and stream.session_id:
                    current_state = session_states.get(stream.session_id, "LISTENING")
                    
                    if current_state == "LISTENING" and stream.previous_state != "LISTENING":
                        stream.listening_packet_count = 0
                    stream.previous_state = current_state
                    
                    # During PROCESSING, discard audio to prevent concurrency issues
                    if current_state == "PROCESSING":
                        stream.pcm_buffer.clear()
                        stream.silence_frames = 0
                        stream.is_speaking = False
                        stream.barge_in_frames = 0
                        continue
                    
                    mulaw_bytes = base64.b64decode(payload_b64)
                    from app.services.telecommunication_service import mulaw_to_pcm_8k
                    pcm_chunk = mulaw_to_pcm_8k(mulaw_bytes)
                    
                    import audioop
                    rms = audioop.rms(pcm_chunk, 2)
                    
                    # --- BARGE-IN DETECTION: Monitor user speech during AI playback ---
                    if current_state == "SPEAKING":
                        barge_in_enabled = stream.survey_config.get("persona", {}).get("callBargeInEnabled", False) if stream.survey_config else False
                        if not barge_in_enabled:
                            # If barge-in is disabled, still buffer audio so we don't clip words spoken right at the end
                            stream.pcm_buffer.extend(pcm_chunk)
                            if len(stream.pcm_buffer) > 8000:
                                stream.pcm_buffer = stream.pcm_buffer[-8000:]
                            continue
                        
                        # We need a sustained RMS over 400 for ~200ms (10 frames) to consider it a barge-in
                        BARGE_IN_RMS_THRESHOLD = 400
                        BARGE_IN_MIN_FRAMES = 10
                        
                        if rms > BARGE_IN_RMS_THRESHOLD:
                            # Increase faster if very loud
                            stream.barge_in_frames += 2 if rms > 1200 else 1
                        else:
                            # Gradual decay to handle brief gaps in speech
                            stream.barge_in_frames = max(0, stream.barge_in_frames - 1)
                        
                        if stream.barge_in_frames >= BARGE_IN_MIN_FRAMES:
                            logger.info(f"[BARGE-IN] User speech detected during AI playback (RMS: {rms}, sustained frames: {stream.barge_in_frames}). Interrupting.")
                            session_states[stream.session_id] = "INTERRUPTED"
                            
                            # Immediately stop all queued audio playback
                            if stream.mixer:
                                stream.mixer.clear_speech()
                            try:
                                await websocket.send_json({"event": "clear", "streamSid": stream.stream_sid})
                            except Exception:
                                pass
                            
                            stream.pcm_buffer.extend(pcm_chunk)
                            stream.is_speaking = True
                            stream.silence_frames = 0
                            stream.barge_in_frames = 0
                        else:
                            stream.pcm_buffer.extend(pcm_chunk)
                            if len(stream.pcm_buffer) > 8000:
                                stream.pcm_buffer = stream.pcm_buffer[-8000:]
                        continue
                    
                    if current_state == "INTERRUPTED":
                        stream.barge_in_frames = 0
                    
                    # --- NORMAL VAD (LISTENING and INTERRUPTED states) ---
                    stream.pcm_buffer.extend(pcm_chunk)
                    
                    packet_count = getattr(websocket, "_packet_count", 0) + 1
                    websocket._packet_count = packet_count
                    if packet_count % 100 == 0:
                        logger.info(f"[AUDIO-PACKET] Received 100 audio chunks from Smartflo (Current buffer: {len(stream.pcm_buffer)} bytes)")
                    
                    if rms < 600:
                        if stream.is_speaking:
                            stream.silence_frames += 1
                    else:
                        if not stream.is_speaking:
                            logger.info(f"[AUDIO-IN] Speech detected (RMS: {rms}). Starting audio capture.")
                        stream.silence_frames = 0
                        stream.is_speaking = True
                        stream.reprompt_count = 0
                        
                    if not stream.is_speaking:
                        # Count packets instead of wall clock time to be immune to event loop congestion
                        # Each packet = 20ms, so 250 packets = 5 seconds of real audio silence
                        if current_state == "LISTENING":
                            stream.listening_packet_count += 1
                        
                        SILENCE_TIMEOUT_PACKETS = 150  # 150 packets * 20ms = 3 seconds
                        if current_state == "LISTENING" and stream.listening_packet_count >= SILENCE_TIMEOUT_PACKETS and stream.reprompt_count == 0:
                            logger.info(f"[SILENCE-TIMEOUT] User silent for {stream.listening_packet_count} packets (~{stream.listening_packet_count * 0.02:.1f}s). Triggering reprompt.")
                            stream.reprompt_count += 1
                            session_states[stream.session_id] = "PROCESSING"
                            stream.silence_frames = 0
                            stream.pcm_buffer.clear()
                            
                            stream.audio_task = asyncio.create_task(
                                handle_customer_audio(
                                    websocket=websocket,
                                    stream_sid=stream.stream_sid,
                                    session_id=stream.session_id,
                                    pcm_bytes=b"",
                                    survey_config=stream.survey_config,
                                    session_history=stream.session_history,
                                    extracted_data=stream.extracted_data,
                                    customer_number=stream.customer_number,
                                    call_sid=stream.call_sid,
                                    mixer=stream.mixer
                                )
                            )
                            continue
                        elif current_state == "LISTENING" and stream.listening_packet_count >= SILENCE_TIMEOUT_PACKETS * 2 and stream.reprompt_count > 0:
                            logger.info(f"[SILENCE-TIMEOUT] User silent for {stream.listening_packet_count} packets after reprompt. Hanging up.")
                            session_states[stream.session_id] = "PROCESSING"
                            
                            try:
                                if stream.mixer:
                                    stream.mixer.clear_speech()
                                await websocket.send_json({"event": "stop", "streamSid": stream.stream_sid})
                                logger.info(f"[HANGUP-TRACE] Session {stream.session_id} - Sent 'stop' event to Smartflo on silence timeout.")
                            except Exception as stop_err:
                                logger.info(f"[HANGUP-TRACE] Session {stream.session_id} - Failed to send stop event: {stop_err}")
                                
                            if stream.call_sid:
                                from app.services.telecommunication_service import hangup_call
                                logger.info(f"[HANGUP-TRACE] Session {stream.session_id} - Calling hangup_call API on silence timeout for call_sid={stream.call_sid}")
                                asyncio.create_task(hangup_call(stream.call_sid))
                            continue
                            
                        if len(stream.pcm_buffer) > 8000:
                            stream.pcm_buffer = stream.pcm_buffer[-8000:]
                        
                    if stream.is_speaking and stream.silence_frames >= 25:
                        user_pcm = bytes(stream.pcm_buffer)
                        logger.info(f"[AUDIO-IN] Silence detected after speech. Capture complete. Utterance size: {len(user_pcm)} bytes (~{len(user_pcm)/16000:.2f}s)")
                        
                        if stream.audio_task and not stream.audio_task.done():
                            logger.info("[BARGE-IN] Waiting for previous handle_customer_audio to finish...")
                            try:
                                await asyncio.wait_for(stream.audio_task, timeout=3.0)
                            except (asyncio.TimeoutError, Exception) as wait_err:
                                logger.warning(f"Previous audio task did not finish in time: {wait_err}")
                        
                        session_states[stream.session_id] = "PROCESSING"
                        stream.is_speaking = False
                        stream.silence_frames = 0
                        stream.pcm_buffer.clear()
                        
                        stream.audio_task = asyncio.create_task(
                            handle_customer_audio(
                                websocket=websocket,
                                stream_sid=stream.stream_sid,
                                session_id=stream.session_id,
                                pcm_bytes=user_pcm,
                                survey_config=stream.survey_config,
                                session_history=stream.session_history,
                                extracted_data=stream.extracted_data,
                                customer_number=stream.customer_number,
                                call_sid=stream.call_sid,
                                mixer=stream.mixer
                            )
                        )
                        
            elif event_type == "stop":
                stream_sid = data.get("streamSid")
                logger.info(f"Smartflo Stream stopped. StreamSid: {stream_sid}")
                if stream_sid:
                    await finalize_stream(stream_sid)
                
    except WebSocketDisconnect:
        logger.info("[HANGUP-TRACE] Outer handler: WebSocket disconnected gracefully.")
    except Exception as e:
        logger.error(f"[HANGUP-TRACE] Outer handler: UNHANDLED EXCEPTION in WebSocket handler: {e}", exc_info=True)
    finally:
        for stream_sid in list(streams.keys()):
            await finalize_stream(stream_sid)
