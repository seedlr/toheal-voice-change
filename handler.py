import runpod
import requests
import os
import tempfile
import base64
import subprocess
import glob
import zipfile
import shutil

MODELS_DIR = "/app/rvc_models"
os.makedirs(MODELS_DIR, exist_ok=True)

rvc_instance = None

def get_rvc():
    global rvc_instance
    if rvc_instance is None:
        from rvc_python.infer import RVCInference
        rvc_instance = RVCInference(device="cuda:0")
    return rvc_instance


def find_model_files(model_dir):
    pth_files = glob.glob(os.path.join(model_dir, "**/*.pth"), recursive=True)
    if not pth_files:
        return None, None
    index_files = glob.glob(os.path.join(model_dir, "**/*.index"), recursive=True)
    return pth_files[0], index_files[0] if index_files else None


def download_model_from_url(model_url):
    safe_name = model_url.split("/")[-1].replace(".zip", "").replace(" ", "_")
    model_dir = os.path.join(MODELS_DIR, safe_name)

    pth, idx = find_model_files(model_dir)
    if pth:
        return pth, idx

    os.makedirs(model_dir, exist_ok=True)
    print(f"[RVC] Downloading model from URL: {model_url[:100]}...")

    hf_token = os.environ.get("HF_TOKEN", "")
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    resp = requests.get(model_url, headers=headers, timeout=120, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download model: HTTP {resp.status_code}")

    if model_url.endswith(".zip"):
        zip_path = os.path.join(model_dir, "model.zip")
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(model_dir)
        os.unlink(zip_path)
    elif model_url.endswith(".pth"):
        pth_path = os.path.join(model_dir, safe_name + ".pth")
        with open(pth_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        file_path = os.path.join(model_dir, safe_name)
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    pth, idx = find_model_files(model_dir)
    if not pth:
        raise RuntimeError(f"No .pth file found after downloading from {model_url}")
    return pth, idx


def download_model_from_hf(model_name):
    safe_name = model_name.replace("/", "__")
    model_dir = os.path.join(MODELS_DIR, safe_name)

    pth, idx = find_model_files(model_dir)
    if pth:
        return pth, idx

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

    pth, idx = find_model_files(model_dir)
    if not pth:
        raise RuntimeError(f"No .pth file found in model '{model_name}'.")
    return pth, idx


def download_model(model_name, model_url=None):
    if model_url:
        return download_model_from_url(model_url)
    if model_name.startswith("http"):
        return download_model_from_url(model_name)
    return download_model_from_hf(model_name)


def handler(event):
    try:
        input_data = event["input"]
        audio_url = input_data.get("audio_url", "")
        audio_base64 = input_data.get("audio_base64", "")
        model_name = input_data.get("model_name", "")
        model_url = input_data.get("model_url", "")
        pitch = input_data.get("pitch", 0)
        f0_method = input_data.get("f0_method", "rmvpe")
        index_rate = input_data.get("index_rate", 0.75)
        filter_radius = input_data.get("filter_radius", 3)
        rms_mix_rate = input_data.get("rms_mix_rate", 0.25)
        protect = input_data.get("protect", 0.33)

        if not model_name and not model_url:
            return {"error": "model_name or model_url is required", "status": "failed"}

        if audio_base64:
            print(f"[RVC] Decoding audio from base64 ({len(audio_base64)} chars)...")
            audio_bytes = base64.b64decode(audio_base64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_raw = f.name
                f.write(audio_bytes)
        elif audio_url:
            print(f"[RVC] Downloading audio from {audio_url[:80]}...")
            response = requests.get(audio_url, timeout=60)
            if response.status_code != 200:
                return {"error": f"Failed to download audio: HTTP {response.status_code}"}
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_raw = f.name
                f.write(response.content)
        else:
            return {"error": "audio_url or audio_base64 is required", "status": "failed"}

        input_path = temp_raw.replace(".wav", "_input.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_raw, "-ar", "16000", "-ac", "1", input_path],
            capture_output=True, timeout=30
        )
        if os.path.exists(temp_raw) and temp_raw != input_path:
            os.unlink(temp_raw)

        if not os.path.exists(input_path):
            return {"error": "Failed to convert audio to WAV format", "status": "failed"}

        identifier = model_url or model_name
        print(f"[RVC] Downloading model '{identifier[:80]}'...")
        model_path, index_path = download_model(model_name, model_url)
        print(f"[RVC] Model: {model_path}")
        print(f"[RVC] Index: {index_path or 'none'}")

        rvc = get_rvc()
        rvc.load_model(model_path, index_path=index_path)

        rvc.f0method = f0_method
        rvc.f0up_key = pitch
        rvc.index_rate = index_rate
        rvc.filter_radius = filter_radius
        rvc.rms_mix_rate = rms_mix_rate
        rvc.protect = protect

        output_path = input_path.replace("_input.wav", "_output.wav")
        print(f"[RVC] Running inference (f0={f0_method}, pitch={pitch})...")
        rvc.infer_file(input_path, output_path)

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
            "model_used": model_name or model_url,
            "output_format": "wav",
            "output_size_bytes": output_size,
            "status": "success",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}


runpod.serverless.start({"handler": handler})
