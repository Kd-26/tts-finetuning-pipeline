# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ffmpeg git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt pyproject.toml README.md ./
RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.6.0 torchaudio==2.6.0 \
       --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt

COPY src ./src
RUN python -m pip install -e . --no-deps

COPY configs ./configs
COPY data ./data
COPY scripts ./scripts
COPY tests ./tests

ENTRYPOINT ["python", "-m"]
CMD ["tts_finetune.validate", "--config", "configs/smoke.yaml"]
