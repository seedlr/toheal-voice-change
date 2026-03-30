"""
RunPod Serverless Handler for RVC v2 Voice Conversion
Downloads trained RVC models from HuggingFace on demand, caches them,
and runs voice conversion using the RVC WebUI inference pipeline.

Handles both proper inference models AND raw training checkpoints (G_*.pth).
Raw checkpoints are auto-converted to inference format on first load.
"""

import os
import sys
import argparse
from collections import OrderedDict

RVC_DIR = "/workspace/RVC"
MODELS_DIR = "/workspace/rvc_models"
HUBERT_PATH = os.path.join(RVC_DIR, "assets", "hubert", "hubert_base.pt")
RMVPE_PATH = os.path.join(RVC_DIR, "assets", "rmvpe.pt")

os.chdir(RVC_DIR)

os.environ["rmvpe_root"] = os.path.join(RVC_DIR, "assets")
os.environ["weight_root"] = MODELS_DIR
os.environ["index_root"] = MODELS_DIR
os.environ["outside_index_root"] = MODELS_DIR

_original_parse_args = argparse.ArgumentParser.parse_args
def _safe_parse_args(self, args=None, namespace=None):
    try:
        return _original_parse_args(self, args=[], namespace=namespace)
    except Exception:
        return argparse.Namespace(
            port=7865, pycmd=sys.executable or "python",
            colab=False, noparallel=True, noautoopen=True, dml=False
        )
argparse.ArgumentParser.parse_args = _safe_parse_args

sys.path.insert(0, RVC_DIR)

import runpod
import torch
import base64
import tempfile
import subprocess
import traceback
import shutil

configs_inuse = os.path.join(RVC_DIR, "configs", "inuse")
for sub in ["v1", "v2"]:
    os.makedirs(os.path.join(configs_inuse, sub), exist_ok=True)
    src_dir = os.path.join(RVC_DIR, "configs", sub)
    if os.path.isdir(src_dir):
        for fn in os.listdir(src_dir):
            if fn.endswith(".json"):
                src = os.path.join(src_dir, fn)
                dst = os.path.join(configs_inuse, sub, fn)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

os.makedirs(MODELS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = device == "cuda"

V2_48K_CONFIG = [
    1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
    [3, 7, 11],
    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    [12, 10, 2, 2],
    512,
    [24, 20, 4, 4],
    109, 256, 48000,
]

vc_instance = None
current_model = None


def convert_raw_checkpoint(raw_path):
    """Convert a raw training checkpoint (G_*.pth style) to proper RVC inference format."""
    print(f"[Convert] Converting raw checkpoint: {raw_path}")
    ckpt = torch.load(raw_path, map_location="cpu")

    if "config" in ckpt and "weight" in ckpt:
        print(f"[Convert] Already in inference format, skipping conversion")
        return

    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    opt = OrderedDict()
    opt["weight"] = {}
    for key in state_dict.keys():
        if "enc_q" in key:
            continue
        opt["weight"][key] = state_dict[key].half()

    opt["config"] = V2_48K_CONFIG
    opt["info"] = "converted_from_raw_checkpoint"
    opt["sr"] = "48k"
    opt["f0"] = 1
    opt["version"] = "v2"

    converted_path = raw_path + ".converted"
    torch.save(opt, converted_path)
    os.replace(converted_path, raw_path)

    size_mb = os.path.getsize(raw_path) / (1024 * 1024)
    print(f"[Convert] Saved inference model: {size_mb:.1f} MB (was raw checkpoint)")


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
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    if not os.path.exists(model_path):
        print(f"[Download] Model: {model_url}")
        result = subprocess.run(
            ["curl", "-sL", model_url, "-o", model_path],
            timeout=300, capture_output=True
        )
        if result.returncode != 0 or not os.path.exists(model_path):
            raise RuntimeError(f"Failed to download model from {model_url}")
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"[Download] Model saved: {size_mb:.0f} MB")

    needs_conversion = False
    try:
        check = torch.load(model_path, map_location="cpu")
        if "config" not in check or "weight" not in check:
            needs_conversion = True
            print(f"[Download] Model is raw checkpoint (keys: {list(check.keys())[:5]}), converting...")
        else:
            print(f"[Download] Model is proper inference format")
        del check
    except Exception as e:
        print(f"[Download] Could not inspect model: {e}")

    if needs_conversion:
        convert_raw_checkpoint(model_path)

    index_path = None
    if index_url:
        index_name = index_url.split("/")[-1]
        candidate = os.path.join(MODELS_DIR, index_name)
        if not os.path.exists(candidate):
            print(f"[Download] Index: {index_url}")
            subprocess.run(
                ["curl", "-sL", index_url, "-o", candidate],
                timeout=120, capture_output=True
            )
        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            index_path = candidate
            size_mb = os.path.getsize(candidate) / (1024 * 1024)
            print(f"[Download] Index ready: {size_mb:.1f} MB")
        else:
            print(f"[Download] Index download failed or empty, proceeding without index")

    return model_path, index_path


def load_model(model_name, model_url, index_url):
    global current_model

    model_path, index_path = download_model(model_url, index_url, model_name)

    if current_model == model_name:
        return model_path, index_path

    vc = get_vc()
    pth_filename = f"{model_name}.pth"
    vc.get_vc(pth_filename)
    current_model = model_name
    print(f"[Model] Loaded: {model_name} (file={pth_filename})")

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
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
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
print(f"[Init] CWD: {os.getcwd()}")
print(f"[Init] weight_root: {os.environ.get('weight_root')}")
print(f"[Init] HuBERT: {'OK' if os.path.exists(HUBERT_PATH) else 'MISSING'}")
print(f"[Init] RMVPE: {'OK' if os.path.exists(RMVPE_PATH) else 'MISSING'}")
print(f"[Init] RVC repo: {'OK' if os.path.isdir(RVC_DIR) else 'MISSING'}")
print(f"[Init] configs/inuse: {os.listdir(configs_inuse) if os.path.isdir(configs_inuse) else 'MISSING'}")

print("[Init] VC will be lazy-initialized on first request")
print("[Init] Ready for requests")

runpod.serverless.start({"handler": handler})
