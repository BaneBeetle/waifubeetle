# Original Github Repo: https://github.com/rayenfeng/riko_project
# Had to make multiple changes in api.py (seperate folder) to make this work

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from faster_whisper import WhisperModel
from process.asr_func.asr_push_to_talk import record_and_transcribe
from process.llm_funcs.llm_scr import llm_response
from process.tts_func.sovits_ping import sovits_gen, play_audio
from pathlib import Path
import time
### transcribe audio 
import uuid
import soundfile as sf


def get_wav_duration(path):
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate


print(' \n ========= Starting Chat... ================ \n')
whisper_model = WhisperModel("base.en", device="cpu", compute_type="float32")

while True:

    conversation_recording = output_wav_path = Path("audio") / "conversation.wav"
    conversation_recording.parent.mkdir(parents=True, exist_ok=True)

    try:
        user_spoken_text = record_and_transcribe(whisper_model, conversation_recording)
        print("🔍 ASR →", repr(user_spoken_text), flush=True) 
    
    except:
        print("‼ ASR crashed with:", repr(e), flush=True)

        import traceback; traceback.print_exc()
        break   # or continue, depending on how you want to recover

    ### pass to LLM and get a LLM output.

    llm_output = llm_response(user_spoken_text)
    tts_read_text = llm_output
    print(tts_read_text)
### file organization 

    # 1. Generate a unique filename
    uid = uuid.uuid4().hex
    filename = f"output_{uid}.wav"
    output_wav_path = Path("audio") / filename
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    # generate audio and save it to client/audio 
    gen_aud_path = sovits_gen(tts_read_text,output_wav_path)


    play_audio(output_wav_path)
    # clean up audio files
    [fp.unlink() for fp in Path("audio").glob("*.wav") if fp.is_file()]
    # # Example
    # duration = get_wav_duration(output_wav_path)

    # print("waiting for audio to finish...")
    # time.sleep(duration)