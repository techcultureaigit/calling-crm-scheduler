from datetime import timedelta
import httpx
from datetime import datetime
import asyncio
import struct
from app.core.config import settings
from app.core.logger import logger

def downsample_pcm_24k_to_8k(pcm_bytes: bytes) -> bytes:
    """
    Downsamples 24kHz 16-bit PCM to 8kHz 16-bit PCM with proper anti-aliasing low-pass filtering.
    Prevents robotic aliasing distortion, breakiness, and audio crackling.
    """
    if not pcm_bytes:
        return b""
    try:
        import audioop
        res, _ = audioop.ratecv(pcm_bytes, 2, 1, 24000, 8000, None)
        return res
    except Exception:
        num_samples = len(pcm_bytes) // 2
        samples = struct.unpack(f"<{num_samples}h", pcm_bytes)
        if num_samples < 3:
            return pcm_bytes
        filtered = [samples[0]]
        for i in range(1, num_samples - 1):
            filtered.append(int(0.25 * samples[i-1] + 0.5 * samples[i] + 0.25 * samples[i+1]))
        filtered.append(samples[-1])
        downsampled = filtered[::3]
        return struct.pack(f"<{len(downsampled)}h", *downsampled)

def pcm_8k_to_mulaw(pcm_bytes: bytes) -> bytes:
    """
    Converts 8kHz 16-bit PCM to G.711 mu-law.
    """
    import audioop
    return audioop.lin2ulaw(pcm_bytes, 2)

def mulaw_to_pcm_8k(mulaw_bytes: bytes) -> bytes:
    """
    Converts G.711 mu-law bytes back to 16-bit PCM.
    """
    import audioop
    return audioop.ulaw2lin(mulaw_bytes, 2)

def trim_pcm_silence(pcm_bytes: bytes, threshold_rms: int = 1000, chunk_size: int = 320) -> bytes:
    """
    Trims silence from the beginning and end of PCM 8kHz 16-bit audio.
    """
    if len(pcm_bytes) < chunk_size * 2:
        return pcm_bytes
    import audioop
    start_idx = 0
    end_idx = len(pcm_bytes)
    
    # Find start
    for i in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[i:i+chunk_size]
        if len(chunk) < 2:
            break
        rms = audioop.rms(chunk, 2)
        if rms > threshold_rms:
            start_idx = max(0, i - chunk_size * 5)
            break
            
    # Find end
    for i in range(len(pcm_bytes) - chunk_size, -1, -chunk_size):
        chunk = pcm_bytes[i:i+chunk_size]
        if len(chunk) < 2:
            continue
        rms = audioop.rms(chunk, 2)
        if rms > threshold_rms:
            end_idx = min(len(pcm_bytes), i + chunk_size * 5)
            break
            
    if start_idx >= end_idx:
        return pcm_bytes
    return pcm_bytes[start_idx:end_idx]

def pcm_8k_to_wav(pcm_bytes: bytes) -> bytes:
    """
    Wraps raw 8kHz 16-bit PCM mono bytes in a standard 8kHz WAV header.
    Preserves clean telephony audio quality for Whisper STT without ratecv distortion.
    """
    sample_rate = 8000
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_bytes)
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,  # PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header + pcm_bytes


import time
from typing import Optional

class SmartfloClient:
    def __init__(self):
        self._access_token: Optional[str] = None
        self._expires_at: float = 0
        self._lock = None

    def _get_lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def login(self) -> str:
        """
        Authenticates with Smartflo API to generate an authentication token.
        POST /v1/auth/login
        Keeps generated access_token in memory instead of saving to .env.
        """
        email = getattr(settings, "SMARTFLO_EMAIL", None)
        password = getattr(settings, "SMARTFLO_PASSWORD", None)
        base_url = settings.SMARTFLO_BASE_URL.rstrip('/')

        if not email or email == "not_set" or not password or password == "not_set":
            fallback_token = getattr(settings, "SMARTFLO_API_TOKEN", "") or ""
            if fallback_token and fallback_token != "not_set":
                self._access_token = fallback_token
                self._expires_at = time.time() + 86400 * 30
                return self._access_token
            logger.warning("SMARTFLO_EMAIL or SMARTFLO_PASSWORD is not configured in settings.")
            return ""

        url = f"{base_url}/v1/auth/login"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        payload = {
            "email": email,
            "password": password
        }

        logger.info(f"Authenticating with Smartflo API ({url}) for user '{email}'...")
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success") is True and data.get("access_token"):
                        token = data["access_token"]
                        expires_in = data.get("expires_in", 3600)
                        self._access_token = token
                        # Cache in memory with 5-minute safety buffer before expiration (~55 minutes)
                        self._expires_at = time.time() + max(300, expires_in - 300)
                        logger.info(f"Smartflo login successful! Access token cached in memory for {expires_in} seconds.")
                        return token
                    else:
                        logger.error(f"Smartflo auth login failed: {data.get('message', response.text)}")
                elif response.status_code == 429:
                    res_json = {}
                    try: res_json = response.json()
                    except Exception: pass
                    logger.warning(f"Smartflo auth login rate limited (429): {res_json.get('message', response.text)}. Retry after: {res_json.get('retry_after')}")
                    # If we already have a cached token in memory, reuse it during rate-limit cooldown
                    if self._access_token:
                        return self._access_token
                else:
                    logger.error(f"Smartflo auth login API returned status {response.status_code}: {response.text}")
            except Exception as e:
                logger.error(f"Exception during Smartflo login request: {e}")

        # Fall back to SMARTFLO_API_TOKEN only if it is explicitly configured and not "not_set"
        fallback_token = getattr(settings, "SMARTFLO_API_TOKEN", "") or ""
        if fallback_token and fallback_token != "not_set":
            return fallback_token

        return self._access_token or ""

    async def get_access_token(self, force_refresh: bool = False) -> str:
        """
        Returns a valid in-memory Smartflo API token. Automatically logs in if token is missing or expired.
        Uses async lock to prevent parallel concurrent login requests.
        """
        now = time.time()
        if not force_refresh and self._access_token and now < self._expires_at:
            return self._access_token

        async with self._get_lock():
            now = time.time()
            if not force_refresh and self._access_token and now < self._expires_at:
                return self._access_token
            return await self.login()

    async def get_my_numbers(self) -> list:
        """
        Fetches the list of registered My Numbers from Smartflo API.
        GET /v1/my_number
        """
        token = await self.get_access_token()
        url = f"{settings.SMARTFLO_BASE_URL.rstrip('/')}/v1/my_number"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {token}"
        }
        logger.info(f"Fetching list of My Numbers from Smartflo API ({url})...")
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url, headers=headers)
                if response.status_code == 401:
                    logger.warning("Smartflo get_my_numbers returned 401 Unauthorized. Force refreshing token in memory...")
                    token = await self.get_access_token(force_refresh=True)
                    headers["Authorization"] = f"Bearer {token}"
                    response = await client.get(url, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    numbers_list = data if isinstance(data, list) else []
                    logger.info(f"Successfully fetched {len(numbers_list)} My Numbers from Smartflo API.")
                    return numbers_list
                else:
                    logger.error(f"Smartflo get_my_numbers API returned status {response.status_code}: {response.text}")
                    return []
            except Exception as e:
                logger.error(f"Failed to fetch My Numbers from Smartflo API: {e}")
                return []

    async def initiate_call(
        self, 
        destination_number: str = None, 
        agent_number: str = None, 
        caller_id: str = None, 
        api_key: str = None,
        is_async: int = 1,
        customer_ring_timeout: int = 30,
        call_timeout: int = None,
        custom_identifier = None,
        survey_id: str = "default",
        create_mapping: bool = True,
        **kwargs
    ) -> dict:
        """
        Initiates a customer-first Click-to-Call Support request via Smartflo API.
        Endpoint: POST /v1/click_to_call_support
        """
        # Verify WebSocket server readiness before calling customer
        from app.services.scheduler_service import check_websocket_active
        if not await check_websocket_active():
            logger.error("Aborting call initiation: Local WebSocket server/URL is NOT active or reachable.")
            raise RuntimeError("WebSocket server is NOT active or listening. Outbound call aborted.")

        token = await self.get_access_token()
        url = f"{settings.SMARTFLO_BASE_URL.rstrip('/')}/v1/click_to_call_support"
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        # Resolve API Key required by Click-to-Call Support API
        resolved_api_key = (
            api_key 
            or getattr(settings, "SMARTFLO_CLICK_TO_CALL_API_KEY", None)
            or getattr(settings, "SMARTFLO_API_KEY", None)
            or getattr(settings, "SMARTFLO_API_TOKEN", "not_set")
        )
        
        # Resolve valid provisioned Smartflo DID for caller_id
        my_numbers = await self.get_my_numbers()
        valid_dids = []
        if my_numbers and isinstance(my_numbers, list):
            for m in my_numbers:
                if isinstance(m, dict):
                    alias = m.get("alias")
                    did = m.get("did", "").replace("+", "")
                    if alias: valid_dids.append(alias)
                    if did and did not in valid_dids: valid_dids.append(did)
                    
        if not hasattr(self.__class__, '_rr_index'):
            self.__class__._rr_index = 0
            
        fallback_did = None
        if valid_dids:
            fallback_did = valid_dids[self.__class__._rr_index % len(valid_dids)]
            self.__class__._rr_index = (self.__class__._rr_index + 1) % len(valid_dids)
        
        # Clean caller_id and agent_number to match owned DIDs
        target_caller_id = caller_id if (caller_id and (not valid_dids or caller_id in valid_dids)) else fallback_did
        target_dest = destination_number or agent_number

        # Clean customer number (ensure 10 to 12 digits)
        clean_cust = "".join(filter(str.isdigit, str(target_dest)))
        if len(clean_cust) == 12 and clean_cust.startswith("91"):
            clean_cust = clean_cust[2:]

        # Build custom_identifier object for correlation
        cust_ident = custom_identifier if isinstance(custom_identifier, dict) else {}
        if survey_id and "survey_id" not in cust_ident:
            cust_ident["survey_id"] = str(survey_id)
        cust_ident["customer_number"] = clean_cust

        payload = {
            "customer_number": clean_cust,
            "api_key": resolved_api_key,
            "caller_id": target_caller_id,
            "async": 1,
            "customer_ring_timeout": customer_ring_timeout or 30
        }
        if call_timeout is not None:
            payload["call_timeout"] = call_timeout
        if cust_ident:
            payload["custom_identifier"] = cust_ident

        logger.info(f"Initiating Smartflo Click-to-Call Support to customer {clean_cust} (caller_id: {target_caller_id}, survey_id: {survey_id})...")

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 401:
                    logger.warning("Smartflo initiate_call returned 401 Unauthorized. Force refreshing token in memory...")
                    token = await self.get_access_token(force_refresh=True)
                    headers["Authorization"] = f"Bearer {token}"
                    response = await client.post(url, headers=headers, json=payload)
            except httpx.RequestError as e:
                logger.error(f"HTTP Request error while initiating Smartflo call: {e}")
                raise RuntimeError(f"Failed to connect to Smartflo Click-to-Call API: {e}")

            if response.status_code not in (200, 201):
                logger.error(f"Smartflo Click-to-Call API returned status {response.status_code}: {response.text}")
                try:
                    error_data = response.json()
                    message = error_data.get("message", "Unknown error")
                except Exception:
                    message = response.text
                raise RuntimeError(f"Smartflo call initiation failed: {message}")

            try:
                data = response.json()
            except ValueError:
                logger.error(f"Invalid JSON response from Smartflo Click-to-Call: {response.text}")
                raise RuntimeError("Smartflo Click-to-Call returned invalid JSON format.")

            logger.info(f"Smartflo Click-to-Call API response: {data}")
            
            ref_id = None
            call_id = None
            if isinstance(data, dict):
                ref_id = data.get("ref_id")
                call_id = data.get("call_id") or data.get("data", {}).get("call_id") or data.get("id") or ref_id
            
            # Determine target customer number for polling
            target_number = payload.get("customer_number") or destination_number
            
            def clean_number(num):
                if not num:
                    return ""
                return "".join(filter(str.isdigit, str(num)))
                
            cleaned_target = clean_number(target_number)

            # Instantly record mapping in DB so WebSocket start event resolves survey_id without delay
            primary_ref = ref_id or (f"ref_{call_id}" if call_id else f"call_{cleaned_target}_{int(datetime.utcnow().timestamp())}")
            if survey_id and create_mapping:
                try:
                    from app.core.db import get_collection
                    mappings_col = get_collection("call_survey_mappings")
                    mapping_doc = {
                        "survey_id": survey_id,
                        "customer_number": cleaned_target,
                        "ref_id": primary_ref,
                        "call_sid": str(call_id) if call_id else primary_ref,
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                    await mappings_col.update_one(
                        {"ref_id": primary_ref},
                        {"$set": mapping_doc},
                        upsert=True
                    )
                    logger.info(f"Instantly registered call mapping: customer={cleaned_target}, ref_id={primary_ref}, call_id={call_id} -> survey_id={survey_id}")
                except Exception as map_err:
                    logger.error(f"Failed to instantly record call mapping: {map_err}")
            
            is_connected = False
            max_attempts = 15
            poll_delay = 3.0
            
            logger.info(f"Checking connection status of call to {target_number} (cleaned: {cleaned_target})...")
            
            for attempt in range(1, max_attempts + 1):
                logger.info(f"Polling call status (attempt {attempt}/{max_attempts})...")
                await asyncio.sleep(poll_delay)
                
                try:
                    # 1. Check live calls
                    live_url = f"{settings.SMARTFLO_BASE_URL.rstrip('/')}/v1/live_calls"
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        live_resp = await client.get(live_url, headers=headers)
                        if live_resp.status_code == 200:
                            live_data = live_resp.json()
                            logger.info(f"Live calls response: {live_data}")
                            
                            calls = []
                            if isinstance(live_data, list):
                                calls = live_data
                            elif isinstance(live_data, dict):
                                calls = live_data.get("results", []) or live_data.get("data", []) or []
                                
                            for call in calls:
                                dest = clean_number(call.get("destination", ""))
                                source = clean_number(call.get("source", ""))
                                callerid = clean_number(call.get("callerid", ""))
                                
                                # Use exact matching for 10-digit numbers to avoid mismatch when 9 calls fire simultaneously
                                is_match = False
                                if cleaned_target:
                                    t_10 = cleaned_target[-10:] if len(cleaned_target) >= 10 else cleaned_target
                                    d_10 = dest[-10:] if len(dest) >= 10 else dest
                                    s_10 = source[-10:] if len(source) >= 10 else source
                                    if (d_10 and t_10 == d_10) or (s_10 and t_10 == s_10):
                                        is_match = True
                                        
                                if is_match:
                                    state = str(call.get("state", call.get("status", ""))).lower()
                                    logger.info(f"Found active call matching target. State: {state}")
                                    if not call_id:
                                        call_id = call.get("id") or call.get("call_id") or call.get("uniqueid")
                                    
                                    # Register call_sid to survey_id mapping by updating existing call document
                                    resolved_sid = str(call.get("call_id") or call.get("uniqueid") or "")
                                    if resolved_sid and survey_id and create_mapping:
                                        try:
                                            from app.core.db import get_collection
                                            mappings_col = get_collection("call_survey_mappings")
                                            target_filter = None
                                            if primary_ref:
                                                target_filter = {"ref_id": primary_ref}
                                            elif call_id:
                                                target_filter = {"call_sid": str(call_id)}
                                            elif cleaned_target:
                                                cutoff = datetime.utcnow() - timedelta(minutes=15)
                                                target_filter = {"customer_number": cleaned_target, "created_at": {"$gte": cutoff}}

                                            if target_filter:
                                                existing = await mappings_col.find_one(target_filter, sort=[("created_at", -1)])
                                                if existing:
                                                    await mappings_col.update_one(
                                                        {"_id": existing["_id"]},
                                                        {"$set": {"call_sid": resolved_sid, "updated_at": datetime.utcnow()}}
                                                    )
                                                    logger.info(f"Linked call_sid {resolved_sid} to existing call mapping {existing['_id']}")
                                                else:
                                                    await mappings_col.update_one(
                                                        {"call_sid": resolved_sid},
                                                        {"$set": {"survey_id": survey_id, "customer_number": cleaned_target, "ref_id": primary_ref, "created_at": datetime.utcnow()}},
                                                        upsert=True
                                                    )
                                                    logger.info(f"Registered new call_sid mapping: {resolved_sid} -> {survey_id}")
                                        except Exception as db_err:
                                            logger.error(f"Failed to write call mapping: {db_err}")
                                            
                                    if state in ("answered", "connected", "active", "up"):
                                        is_connected = True
                                        break
                            if is_connected:
                                break
                except Exception as e:
                    logger.error(f"Error checking live calls: {e}")
                    
                try:
                    # 2. Check call records as a fallback
                    now = datetime.utcnow()
                    from_date = (now - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S")
                    to_date = (now + timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S")
                    
                    records_url = f"{settings.SMARTFLO_BASE_URL.rstrip('/')}/v1/call/records"
                    params = {
                        "from_date": from_date,
                        "to_date": to_date,
                        "callerid": target_number,
                        "call_type": "c"
                    }
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        rec_resp = await client.get(records_url, headers=headers, params=params)
                        if rec_resp.status_code == 200:
                            rec_data = rec_resp.json()
                            logger.info(f"Call records response: {rec_data}")
                            results = rec_data.get("results", []) if isinstance(rec_data, dict) else []
                            for rec in results:
                                dest = clean_number(rec.get("destination", ""))
                                caller = clean_number(rec.get("callerid", ""))
                                status = str(rec.get("status", "")).lower()
                                if status == "answered" and cleaned_target and (
                                    (dest and (cleaned_target in dest or dest in cleaned_target)) or 
                                    (caller and (cleaned_target in caller or caller in cleaned_target))
                                ):
                                    logger.info("Found answered call in call records!")
                                    if not call_id:
                                        call_id = rec.get("id") or rec.get("call_id") or rec.get("uniqueid")
                                    
                                    # Register call_sid to survey_id mapping by updating existing call document
                                    resolved_sid = str(rec.get("call_id") or rec.get("uniqueid") or "")
                                    if resolved_sid and survey_id and create_mapping:
                                        try:
                                            from app.core.db import get_collection
                                            mappings_col = get_collection("call_survey_mappings")
                                            target_filter = None
                                            if primary_ref:
                                                target_filter = {"ref_id": primary_ref}
                                            elif call_id:
                                                target_filter = {"call_sid": str(call_id)}
                                            elif cleaned_target:
                                                cutoff = datetime.utcnow() - timedelta(minutes=15)
                                                target_filter = {"customer_number": cleaned_target, "created_at": {"$gte": cutoff}}

                                            if target_filter:
                                                existing = await mappings_col.find_one(target_filter, sort=[("created_at", -1)])
                                                if existing:
                                                    await mappings_col.update_one(
                                                        {"_id": existing["_id"]},
                                                        {"$set": {"call_sid": resolved_sid, "updated_at": datetime.utcnow()}}
                                                    )
                                                    logger.info(f"Linked call_sid {resolved_sid} to existing call mapping {existing['_id']} from records")
                                                else:
                                                    await mappings_col.update_one(
                                                        {"call_sid": resolved_sid},
                                                        {"$set": {"survey_id": survey_id, "customer_number": cleaned_target, "ref_id": primary_ref, "created_at": datetime.utcnow()}},
                                                        upsert=True
                                                    )
                                                    logger.info(f"Registered new call_sid mapping from records: {resolved_sid} -> {survey_id}")
                                        except Exception as db_err:
                                            logger.error(f"Failed to write call mapping: {db_err}")
                                            
                                    is_connected = True
                                    break
                            if is_connected:
                                break
                except Exception as e:
                    logger.error(f"Error checking call records: {e}")
            
            if is_connected:
                logger.info("Call is successfully connected on Smartflo.")
            else:
                logger.warning("Call was not detected as connected within the timeout period.")
            
            if isinstance(data, dict) and call_id:
                data["call_id"] = call_id
            return data

# Single instance of the client to reuse cache across requests
smartflo_client = SmartfloClient()

async def get_access_token() -> str:
    """Convenience helper to retrieve the Smartflo access token."""
    return await smartflo_client.get_access_token()

async def get_my_numbers() -> list:
    """Convenience helper to fetch list of My Numbers from Smartflo."""
    return await smartflo_client.get_my_numbers()

async def initiate_call(
    agent_number: str, 
    destination_number: str, 
    caller_id: str, 
    is_async: int = 1,
    call_timeout: int = None,
    custom_identifier: str = None,
    survey_id: str = "default",
    create_mapping: bool = True
) -> dict:
    """Convenience helper to initiate a Smartflo call."""
    return await smartflo_client.initiate_call(
        agent_number=agent_number,
        destination_number=destination_number,
        caller_id=caller_id,
        is_async=is_async,
        call_timeout=call_timeout,
        custom_identifier=custom_identifier,
        survey_id=survey_id,
        create_mapping=create_mapping
    )


async def hangup_call(call_id: str) -> bool:
    """
    Hangs up an active call using Smartflo's hangup API.
    """
    token = await smartflo_client.get_access_token()
    url = f"{settings.SMARTFLO_BASE_URL.rstrip('/')}/v1/call/hangup"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    payload = {
        "call_id": call_id
    }
    
    logger.info(f"Attempting to hang up Smartflo call: {call_id}...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 401:
                logger.warning("Smartflo hangup_call returned 401. Force refreshing token in memory...")
                token = await smartflo_client.get_access_token(force_refresh=True)
                headers["Authorization"] = f"Bearer {token}"
                response = await client.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                logger.info(f"Successfully hung up call {call_id}")
                return True
            else:
                if response.status_code == 422:
                    logger.info(f"Failed to hang up call {call_id} (likely already disconnected): {response.status_code} - {response.text}")
                else:
                    logger.error(f"Failed to hang up call {call_id}: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error hanging up call {call_id}: {e}")
            return False
