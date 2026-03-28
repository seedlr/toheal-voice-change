import runpod
import requests
import os
import tempfile
import base64
import subprocess
import glob

MODELS_DIR = "/app/rvc_models"
os.makedirs(MODELS_DIR, exist_ok=True)

rvc_instance = None

def get_rvc():
    global rvc_instance
    if rvc_instance is None:
        from rvc_python.infer import RVCInference
        rvc_instance = RVCInference(device="cuda:0")
    return rvc_instance


def download_model(model_name):
    safe_name = model_name.replace("/", "__")
    model_dir = os.path.join(MODELS_DIR, safe_name)

    pth_files = glob.glob(os.path.join(model_dir, "**/*.pth"), recursive=True)
    if pth_files:
        index_files = glob.glob(os.path.join(model_dir, "**/*.index"), recursive=True)
        return pth_files[0], index_files[0] if index_files else None

    os.makedirs(model_dir, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN", "")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            token=hf_token if hf_token else None,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download model '{model_name}': {e}")

    pth_files = glob.glob(os.path.join(model_dir, "**/*.pth"), recursive=True)
    if not pth_files:
        raise RuntimeError(f"No .pth file found in model '{model_name}'. Ensure the HuggingFace repo contains an RVC .pth model file.")

    index_files = glob.glob(os.path.join(model_dir, "**/*.index"), recursive=True)
    return pth_files[0], index_files[0] if index_files else None


def handler(event):
    try:
        input_data = event["input"]
        audio_url = input_data["audio_url"]
        model_name = input_data.get("model_name", "")
        pitch = input_data.get("pitch", 0)
        f0_method = input_data.get("f0_method", "rmvpe")
        index_rate = input_data.get("index_rate", 0.75)
        filter_radius = input_data.get("filter_radius", 3)
        rms_mix_rate = input_data.get("rms_mix_rate", 0.25)
        protect = input_data.get("protect", 0.33)

        if not model_name:
            return {"error": "model_name is required (HuggingFace repo like 'username/model-name')", "status": "failed"}

        print(f"[RVC] Downloading audio from {audio_url[:80]}...")
        response = requests.get(audio_url, timeout=60)
        if response.status_code != 200:
            return {"error": f"Failed to download audio: HTTP {response.status_code}"}

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_raw = f.name
            f.write(response.content)

        input_path = temp_raw.replace(".wav", "_input.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_raw, "-ar", "16000", "-ac", "1", input_path],
            capture_output=True, timeout=30
        )
        if os.path.exists(temp_raw) and temp_raw != input_path:
            os.unlink(temp_raw)

        if not os.path.exists(input_path):
            return {"error": "Failed to convert audio to WAV format", "status": "failed"}

        print(f"[RVC] Downloading model '{model_name}'...")
        model_path, index_path = download_model(model_name)
        print(f"[RVC] Model: {model_path}")
        print(f"[RVC] Index: {index_path or 'none'}")

        rvc = get_rvc()
        rvc.load_model(model_path, index_path=index_path)

        output_path = input_path.replace("_input.wav", "_output.wav")
        print(f"[RVC] Running inference (f0={f0_method}, pitch={pitch})...")
        rvc.infer_file(
            input_path,
            output_path,
            f0_method=f0_method,
            f0_up_key=pitch,
            index_rate=index_rate,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
        )

        if not os.path.exists(output_path):
            return {"error": "RVC inference produced no output", "status": "failed"}

        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        output_size = os.path.getsize(output_path)

        os.unlink(input_path)
        os.unlink(output_path)

        print(f"[RVC] Done! Output: {output_size} bytes")
        return {
            "audio_base64": audio_base64,
            "model_used": model_name,
            "output_format": "wav",
            "output_size_bytes": output_size,
            "status": "success",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}


runpod.serverless.start({"handler": handler})
