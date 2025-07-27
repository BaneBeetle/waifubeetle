import requests
### MUST START SERVERS FIRST USING START ALL SERVER SCRIPT
import time
import soundfile as sf 
import sounddevice as sd
import yaml

# Load YAML config
with open('character_config.yaml', 'r') as f:
    char_config = yaml.safe_load(f)


def play_audio(path):
    data, samplerate = sf.read(path)
    sd.play(data, samplerate)
    sd.wait()  # Wait until playback is finished

def sovits_gen(in_text, out_wav="output.wav"):
    API_URL = "http://127.0.0.1:9880/"

    payload = {
        "refer_wav_path":   char_config['sovits_ping_config']['ref_audio_path'],
        "prompt_text":      char_config['sovits_ping_config']['prompt_text'],
        "prompt_language":  char_config['sovits_ping_config']['prompt_lang'],
        "text":             in_text,
        "text_language":    char_config['sovits_ping_config']['text_lang'],
    }

    resp = requests.post(API_URL, json=payload)
    resp.raise_for_status()
    with open(out_wav, "wb") as f:
        f.write(resp.content)
    return out_wav



if __name__ == "__main__":

    start_time = time.time()
    output_wav_pth1 = "output.wav"
    path_to_aud = sovits_gen("if you hear this, that means it is set up correctly", output_wav_pth1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(path_to_aud)


