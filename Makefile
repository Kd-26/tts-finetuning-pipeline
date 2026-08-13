.PHONY: help install install-dev demo validate train smoke infer test lint docker-build docker-validate docker-smoke

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
TORCH_INSTALL = $(PIP) install torch==2.6.0 torchaudio==2.6.0
else
TORCH_INSTALL = $(PIP) install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
endif

help:
	@echo "make install         Install CPU PyTorch and runtime dependencies"
	@echo "make demo            Regenerate tiny plumbing-only demo data"
	@echo "make validate        Validate the demo manifest and audio"
	@echo "make train           Run the production training config"
	@echo "make smoke           Run one training step (downloads pretrained weights)"
	@echo "make infer           Synthesize outputs/example.wav from a trained model"
	@echo "make test            Run unit tests"
	@echo "make docker-validate Build and validate in Docker"

install:
	$(TORCH_INSTALL)
	$(PIP) install -r requirements.txt
	$(PIP) install -e . --no-deps

install-dev: install
	$(PIP) install -r requirements-dev.txt

demo:
	$(PYTHON) scripts/make_demo_data.py --output-dir data/demo

validate:
	$(PYTHON) -m tts_finetune.validate --config configs/smoke.yaml

train:
	$(PYTHON) -m tts_finetune.train --config configs/base.yaml

smoke:
	$(PYTHON) -m tts_finetune.train --config configs/smoke.yaml

infer:
	$(PYTHON) -m tts_finetune.infer --model artifacts/speecht5-final --text "This is a test of the fine tuned voice." --reference-audio data/demo/wavs/demo_001.wav --output outputs/example.wav

test:
	pytest

lint:
	ruff check src tests scripts

docker-build:
	docker build -t tts-finetune:cpu .

docker-validate:
	docker compose run --rm validate

docker-smoke:
	docker compose run --rm smoke
