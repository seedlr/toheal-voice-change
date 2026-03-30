"""
RunPod Serverless Handler for RVC v2 Voice Conversion
Downloads trained RVC models from HuggingFace on demand, caches them,
and runs voice conversion using the RVC WebUI inference pipeline.
"""

import runpod
import torch
import os
import sys
import base64
import tempfile
import subprocess
import traceback

RVC_DIR = "/workspace/RVC"
MODELS_DIR = "/workspace/rvc_models"
HUBERT_PATH = os.path.join(RVC_DIR, "assets", "hubert", "hubert_base.pt")
RMVPE_PATH = os.path.join(RVC_DIR, "assets", "rmvpe.pt")

os.environ["rmvpe_root"] = os.path.join(RVC_DIR, "assets")
os.environ["weight_root"] = os.path.join(RVC_DIR, "assets", "weights")

sys.path.insert(0, RVC_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = device == "cuda"

vc_instance = None
current_model = None


def get_vc():
    global vc_instance
    if vc_instance is not None:
        return vc_instance

    from configs.config import Config
    from infer.modules.vc.modules import VC

    config = Config()
    vc_instance = VC(config)
    print(f"[Init] VC pipeline initialized on {device}")
    return vc_instance


def download_model(model_url, index_url, model_name):
    model_dir = os.path.join(MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{model_name}.pth")
    if not os.path.exists(model_path):
        print(f"[Download] Model: {model_url}")
        result = subprocess.run(
            ["curl", "-sL", model_url, "-o", model_path],
            timeout=120, capture_output=True
        )
        if result.returncode != 0 or not os.path.exists(model_path):
            raise RuntimeError(f"Failed to download model from {model_url}")
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"[Download] Model saved: {size_mb:.0f} MB")

    index_path = None
    if index_url:
        index_name = index_url.split("/")[-1]
        index_path = os.path.join(model_dir, index_name)
        if not os.path.exists(index_path):
            print(f"[Download] Index: {index_url}")
            subprocess.run(
                ["curl", "-sL", index_url, "-o", index_path],
                timeout=120, capture_output=True
            )
            if os.path.exists(index_path):
                size_mb = os.path.getsize(index_path) / (1024 * 1024)
                print(f"[Download] Index saved: {size_mb:.0f} MB")

    return model_path, index_path


def load_model(model_name, model_url, index_url):
    global current_model

    model_path, index_path = download_model(model_url, index_url, model_name)

    if current_model == model_name:
        return model_path, index_path

    vc = get_vc()
    vc.get_vc(model_path)
    current_model = model_name
    print(f"[Model] Loaded: {model_name}")

    return model_path, index_path


def handler(job):
    try:
        job_input = job["input"]

        action = job_input.get("action")
        if action == "health":
            get_vc()
            return {
                "status": "healthy",
                "gpu": torch.cuda.is_available(),
                "device": device,
                "current_model": current_model,
            }

        audio_base64 = job_input.get("audio_base64")
        if not audio_base64:
            return {"error": "No audio_base64 provided"}

        model_name = job_input.get("model_name")
        model_url = job_input.get("model_url")
        if not model_name or not model_url:
            return {"error": "model_name and model_url required"}

        index_url = job_input.get("index_url")
        pitch = int(job_input.get("pitch", 0))
        f0_method = job_input.get("f0_method", "rmvpe")
        index_rate = float(job_input.get("index_rate", 0.75))
        rms_mix_rate = float(job_input.get("rms_mix_rate", 0.25))
        protect = float(job_input.get("protect", 0.33))

        model_path, index_path = load_model(model_name, model_url, index_url)

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "input.wav")

            audio_bytes = base64.b64decode(audio_base64)
            with open(input_path, "wb") as f:
                f.write(audio_bytes)

            norm_path = os.path.join(tmp_dir, "normalized.wav")
            norm_result = subprocess.run([
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1",
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                norm_path
            ], capture_output=True, timeout=30)

            if norm_result.returncode == 0 and os.path.exists(norm_path):
                input_path = norm_path

            vc = get_vc()
            info, audio_opt = vc.vc_single(
                sid=0,
                input_audio_path=input_path,
                f0_up_key=pitch,
                f0_file=None,
                f0_method=f0_method,
                file_index=index_path or "",
                file_index2="",
                index_rate=index_rate,
                filter_radius=3,
                resample_sr=0,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
            )

            if audio_opt is None:
                return {"error": f"Conversion returned no audio. Info: {info}"}

            tgt_sr, audio_data = audio_opt

            import soundfile as sf
            output_path = os.path.join(tmp_dir, "output.wav")
            sf.write(output_path, audio_data, tgt_sr)

            with open(output_path, "rb") as f:
                result_base64 = base64.b64encode(f.read()).decode("utf-8")

            return {
                "audio_base64": result_base64,
                "model_name": model_name,
                "sample_rate": tgt_sr,
                "format": "wav",
            }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


print(f"[Init] RVC Handler starting — device={device}, half={is_half}")
print(f"[Init] HuBERT: {'OK' if os.path.exists(HUBERT_PATH) else 'MISSING'}")
print(f"[Init] RMVPE: {'OK' if os.path.exists(RMVPE_PATH) else 'MISSING'}")
print(f"[Init] RVC repo: {'OK' if os.path.exists(RVC_DIR) else 'MISSING'}")

get_vc()
print("[Init] Ready for requests")

runpod.serverless.start({"handler": handler})
