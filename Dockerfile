FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    soundfile \
    librosa \
    numpy \
    scipy \
    faiss-gpu \
    praat-parselmouth \
    pyworld \
    torchcrepe \
    huggingface_hub

RUN git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git /workspace/RVC && \
    cd /workspace/RVC && \
    pip install --no-cache-dir -r requirements.txt || true

RUN mkdir -p /workspace/RVC/assets/hubert /workspace/RVC/assets/rmvpe /workspace/rvc_models

RUN curl -sL https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt \
    -o /workspace/RVC/assets/hubert/hubert_base.pt

RUN curl -sL https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt \
    -o /workspace/RVC/assets/rmvpe.pt

RUN ls -la /workspace/RVC/assets/hubert/ && ls -la /workspace/RVC/assets/rmvpe.pt

ENV HF_BASE=https://huggingface.co/heladawi9/toheal-voices-ar/resolve/main

RUN for voice in calm_male_ar gentle_male_ar serene_male_ar shadow_male_ar breeze_female_ar dusk_female_ar echo_female_ar mist_female_ar; do \
      echo "[Bake] Downloading ${voice}.pth ..." && \
      curl -sL "${HF_BASE}/${voice}/${voice}.pth" -o "/workspace/rvc_models/${voice}.pth" && \
      echo "[Bake] Downloading ${voice} index ..." && \
      idx_file=$(curl -sL "https://huggingface.co/api/models/heladawi9/toheal-voices-ar/tree/main/${voice}" | python3 -c "import sys,json; files=json.load(sys.stdin); print(next((f['path'] for f in files if f['path'].endswith('.index')),  ''))" 2>/dev/null) && \
      if [ -n "$idx_file" ]; then \
        idx_name=$(basename "$idx_file") && \
        curl -sL "${HF_BASE}/${idx_file}" -o "/workspace/rvc_models/${idx_name}" && \
        echo "[Bake] Index saved: ${idx_name}"; \
      else \
        echo "[Bake] No index found for ${voice}, skipping"; \
      fi; \
    done

RUN echo "[Bake] Models baked into image:" && ls -lh /workspace/rvc_models/

ARG HANDLER_VERSION=8
COPY handler.py /workspace/handler.py

CMD ["python", "-u", "/workspace/handler.py"]
