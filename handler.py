import runpod
import requests
import os
import tempfile
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
import base64

def apply_voice_change(input_path, output_path, preset="deep"):
    y, sr = librosa.load(input_path, sr=None)

    presets = {
        "deep": {"pitch_shift": -4, "speed_factor": 0.95},
        "deeper": {"pitch_shift": -7, "speed_factor": 0.92},
        "high": {"pitch_shift": 4, "speed_factor": 1.05},
        "higher": {"pitch_shift": 7, "speed_factor": 1.08},
        "neutral": {"pitch_shift": -2, "speed_factor": 1.0},
        "robot": {"pitch_shift": 0, "speed_factor": 1.0},
    }

    params = presets.get(preset, presets["deep"])

    y_shifted = librosa.effects.pitch_shift(
        y=y,
        sr=sr,
        n_steps=params["pitch_shift"]
    )

    if params["speed_factor"] != 1.0:
        y_shifted = librosa.effects.time_stretch(y_shifted, rate=params["speed_factor"])

    if preset == "robot":
        t = np.arange(len(y_shifted)) / sr
        modulator = np.sin(2 * np.pi * 30 * t) * 0.3 + 0.7
        y_shifted = y_shifted * modulator
        y_shifted = librosa.effects.pitch_shift(y=y_shifted, sr=sr, n_steps=-3)

    y_shifted = y_shifted / np.max(np.abs(y_shifted)) * 0.95
    sf.write(output_path, y_shifted, sr)
    return output_path

def handler(event):
    try:
        input_data = event["input"]
        audio_url = input_data["audio_url"]
        preset = input_data.get("voice_preset", "deep")

        response = requests.get(audio_url, timeout=60)
        if response.status_code != 200:
            return {"error": f"Failed to download audio: HTTP {response.status_code}"}

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            input_path = f.name
            f.write(response.content)

        if not audio_url.lower().endswith('.wav'):
            try:
                audio = AudioSegment.from_file(input_path)
                wav_path = input_path.replace('.wav', '_converted.wav')
                audio.export(wav_path, format="wav")
                os.unlink(input_path)
                input_path = wav_path
            except Exception as e:
                return {"error": f"Audio conversion failed: {str(e)}"}

        output_path = input_path.replace('.wav', '_output.wav')
        apply_voice_change(input_path, output_path, preset)

        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')

        output_size = os.path.getsize(output_path)

        os.unlink(input_path)
        os.unlink(output_path)

        return {
            "audio_base64": audio_base64,
            "preset_used": preset,
            "output_format": "wav",
            "output_size_bytes": output_size,
            "status": "success"
        }

    except Exception as e:
        return {"error": str(e), "status": "failed"}

runpod.serverless.start({"handler": handler})
