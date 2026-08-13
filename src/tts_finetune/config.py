from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    checkpoint: str
    vocoder: str
    speaker_encoder: str


@dataclass(frozen=True)
class DataConfig:
    manifest: Path
    sampling_rate: int
    min_duration_seconds: float
    max_duration_seconds: float
    max_text_tokens: int
    cache_dir: Path | None


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    final_model_dir: Path
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    gradient_checkpointing: bool
    fp16: bool | str
    num_workers: int
    resume_from_checkpoint: str | None


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    seed: int
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_config(config_path: str | Path) -> ProjectConfig:
    """Load YAML and resolve project paths relative to the repository root."""
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent.parent

    required = {"seed", "model", "data", "training"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")

    model = ModelConfig(**raw["model"])
    data_raw = dict(raw["data"])
    data_raw["manifest"] = _resolve(root, data_raw["manifest"])
    data_raw["cache_dir"] = _resolve(root, data_raw.get("cache_dir"))
    data = DataConfig(**data_raw)

    training_raw = dict(raw["training"])
    training_raw["output_dir"] = _resolve(root, training_raw["output_dir"])
    training_raw["final_model_dir"] = _resolve(root, training_raw["final_model_dir"])
    training = TrainingConfig(**training_raw)
    return ProjectConfig(
        root=root,
        seed=int(raw["seed"]),
        model=model,
        data=data,
        training=training,
    )
