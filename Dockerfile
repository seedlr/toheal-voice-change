FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    requests \
    rvc-python \
    huggingface_hub

RUN mkdir -p /app/rvc_models

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
