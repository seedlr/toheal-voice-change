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

ARG HANDLER_VERSION=7
COPY handler.py /workspace/handler.py

CMD ["python", "-u", "/workspace/handler.py"]
