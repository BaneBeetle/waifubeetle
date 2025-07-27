import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import sounddevice as sd
from faster_whisper import WhisperModel

from process.asr_func.asr_push_to_talk import record_and_transcribe
from process.llm_funcs.llm_scr import llm_response
from process.tts_func.sovits_ping import sovits_gen, play_audio

import pyvts
from pyvts import VTSRequest
from websockets.exceptions import ConnectionClosedError


BASE_DIR  = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

async def safe_request(vts, msg):
    """
    Send a request to VTS, reconnect+reauth and retry once on websocket errors.
    """
    try:
        return await vts.request(msg)
    except (ConnectionClosedError):
        print("VTS socket closed—reconnecting…")
        try:
            await vts.close()
        except:
            pass
        await vts.connect()
        # re‑authenticate if needed
        if vts.get_authentic_status() != 2:
            await vts.request_authenticate_token(force=True)
            await vts.request_authenticate()
            await vts.write_token()
        return await vts.request(msg)

async def wait_for_write_complete(path: Path,
                                  poll_interval: float = 0.05,
                                  stable_time:   float = 0.1,
                                  max_wait:      float = 5.0):
    """Return once `path` exists and its size hasn’t changed for `stable_time`."""
    deadline     = time.time() + max_wait
    prev_size    = -1
    stable_start = None

    while time.time() < deadline:
        if path.exists():
            curr = path.stat().st_size
            if curr != prev_size:
                prev_size    = curr
                stable_start = time.time()
            elif time.time() - stable_start >= stable_time:
                return
        await asyncio.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for {path} to finish writing")


def compute_envelope(path, frame_ms=30):
    """Yield normalized RMS values in [0,1] for each frame."""
    data, sr = sf.read(path)
    frame_len = int(sr * (frame_ms / 1000))
    for i in range(0, len(data), frame_len):
        chunk = data[i:i+frame_len]
        if chunk.size == 0: 
            break
        rms = np.sqrt(np.mean(chunk**2))
        yield min(rms * 10, 1.0)  # simple gain ×10, clamp to 1.0

async def lip_sync(vts, vts_req, wav_path: Path):
    """
    Send lip‑sync envelope values to the custom 'LipLevel' input parameter.
    """
    for level in compute_envelope(wav_path):
        msg = vts_req.requestSetParameterValue(
            "LipLevel",        # custom input param
            level,
            weight=1.0,
            face_found=False,
            mode="set"
        )
        await safe_request(vts, msg)
        await asyncio.sleep(0.01)



async def start_chat():
    vts = pyvts.vts()
    print("Connecting to VTS")
    await vts.connect()
    if not vts.get_connection_status():
        raise RuntimeError("Could not connect to VTube Studio API")

    if vts.get_authentic_status() != 2:
        await vts.request_authenticate_token(force=True)
        await vts.write_token()
        success = await vts.request_authenticate()
        if not success:
            raise RuntimeError("VTS authentication failed")
        
    vts_req = pyvts.VTSRequest(developer="genteki", plugin_name="pyvts")

    await safe_request(vts, vts_req.requestCustomParameter(
        parameter="LipLevel",
        min=0.0,
        max=1.0,
        default_value=0.0,
        info="Normalized TTS lip‑open envelope"
    ))

    print(' \n ========= Starting Chat... ================ \n')
    whisper_model = WhisperModel("base.en", device="cpu", compute_type="float32")

    while True:

        conversation_recording = output_wav_path = Path("audio") / "conversation.wav"
        conversation_recording.parent.mkdir(parents=True, exist_ok=True)
        user_spoken_text = record_and_transcribe(whisper_model, conversation_recording)

        ### pass to LLM and get a LLM output.

        llm_output = llm_response(user_spoken_text)
        tts_read_text = llm_output
        print(f"LLM Generated: {tts_read_text}")

        # ## file organization 
        # 1. Generate a unique filename
        uid = uuid.uuid4().hex
        filename = f"output_{uid}.wav"
        output_wav_path = Path("audio") / filename
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        sovits_gen(llm_output, output_wav_path)

        # Wait for file to be written
        await wait_for_write_complete(output_wav_path)

        audio_task   = asyncio.to_thread(play_audio, str(output_wav_path))
        lipsync_task = asyncio.create_task(lip_sync(vts, vts_req, str(output_wav_path)))
        await asyncio.gather(audio_task, lipsync_task)

        # play_audio(output_wav_path)
        # time.sleep(duration)
        # clean up audio files
        [fp.unlink() for fp in Path("audio").glob("*.wav") if fp.is_file()]
        # # Example

        # print("waiting for audio to finish...")
        # time.sleep(duration)
if __name__ == "__main__":
    asyncio.run(start_chat())