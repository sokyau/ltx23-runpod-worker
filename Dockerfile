FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    TORCH_COMPILE_DISABLE=1 \
    TORCHDYNAMO_DISABLE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements.txt

COPY spaces.py /app/spaces.py
COPY handler.py /app/handler.py
COPY space_src /app/space_src

RUN git clone https://github.com/Lightricks/LTX-2.git /app/space_src/LTX-2 && \
    git -C /app/space_src/LTX-2 checkout ae855f8538843825f9015a419cf4ba5edaf5eec2 && \
    python -m pip install --force-reinstall --no-deps \
      -e /app/space_src/LTX-2/packages/ltx-core \
      -e /app/space_src/LTX-2/packages/ltx-pipelines

RUN mkdir -p /workspace/outputs /workspace/.cache/huggingface

CMD ["python", "-u", "/app/handler.py"]
