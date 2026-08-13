from __future__ import annotations

import argparse
import json
import platform
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
    set_seed,
)

from .collator import TTSDataCollatorWithPadding
from .config import ProjectConfig, load_config
from .preprocess import prepare_datasets
from .validate import validate


def _use_fp16(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value and torch.cuda.is_available()
    if str(value).lower() == "auto":
        return torch.cuda.is_available()
    if str(value).lower() in {"true", "1", "yes"}:
        return torch.cuda.is_available()
    return False


def _write_run_metadata(config: ProjectConfig, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (destination / "run-metadata.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def train(config: ProjectConfig) -> Path:
    errors = validate(config)
    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))

    set_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}", flush=True)
    if device.type == "cpu":
        print("WARNING: CPU training is suitable only for the one-step smoke test.", flush=True)

    processor = SpeechT5Processor.from_pretrained(config.model.checkpoint)
    model = SpeechT5ForTextToSpeech.from_pretrained(config.model.checkpoint)
    model.config.use_cache = False
    datasets = prepare_datasets(config, processor, device)
    collator = TTSDataCollatorWithPadding(
        processor=processor,
        reduction_factor=model.config.reduction_factor,
    )

    output_dir = config.training.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_metadata(config, output_dir)
    fp16 = _use_fp16(config.training.fp16)

    arguments = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
        logging_steps=config.training.logging_steps,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        gradient_checkpointing=config.training.gradient_checkpointing,
        fp16=fp16,
        dataloader_num_workers=config.training.num_workers,
        label_names=["labels"],
        report_to=[],
        remove_unused_columns=False,
        load_best_model_at_end=False,
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)

    final_dir = config.training.final_model_dir
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    _write_run_metadata(config, final_dir)
    print(f"Saved final model and processor to {final_dir}", flush=True)
    return final_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune SpeechT5 from a local manifest")
    parser.add_argument("--config", default="configs/base.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
