from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from speechbrain.inference.classifiers import EncoderClassifier

from .audio import load_audio, speaker_embedding
from .config import ProjectConfig
from .manifest import ManifestRow, read_manifest, split_rows


def _load_speaker_model(model_name: str, root: Path, device: torch.device) -> EncoderClassifier:
    savedir = root / ".cache" / "speechbrain" / model_name.replace("/", "--")
    return EncoderClassifier.from_hparams(
        source=model_name,
        run_opts={"device": str(device)},
        savedir=str(savedir),
    )


def _prepare_row(
    row: ManifestRow,
    processor: Any,
    speaker_model: EncoderClassifier,
    sampling_rate: int,
    max_text_tokens: int,
) -> dict[str, Any] | None:
    waveform = load_audio(row.audio_path, sampling_rate)
    encoded = processor(
        text=row.text,
        audio_target=waveform,
        sampling_rate=sampling_rate,
        return_attention_mask=False,
    )
    input_ids = encoded["input_ids"]
    if len(input_ids) >= max_text_tokens:
        return None
    return {
        "input_ids": input_ids,
        "labels": encoded["labels"][0],
        "speaker_embeddings": speaker_embedding(speaker_model, waveform),
    }


def prepare_datasets(config: ProjectConfig, processor: Any, device: torch.device) -> DatasetDict:
    rows = read_manifest(config.data.manifest)
    train_rows, eval_rows = split_rows(rows)
    cache_dir = _versioned_cache_dir(config, rows)
    if cache_dir and (cache_dir / "dataset_dict.json").is_file():
        print(f"Loading preprocessing cache: {cache_dir}", flush=True)
        return load_from_disk(str(cache_dir))

    speaker_model = _load_speaker_model(config.model.speaker_encoder, config.root, device)

    def prepare(items: list[ManifestRow], label: str) -> Dataset:
        records: list[dict[str, Any]] = []
        for index, row in enumerate(items, start=1):
            record = _prepare_row(
                row,
                processor,
                speaker_model,
                config.data.sampling_rate,
                config.data.max_text_tokens,
            )
            if record is not None:
                records.append(record)
            if index % 100 == 0 or index == len(items):
                print(f"Preprocessed {index}/{len(items)} {label} examples", flush=True)
        if not records:
            raise ValueError(f"No {label} examples remain after preprocessing")
        return Dataset.from_list(records)

    dataset = DatasetDict(train=prepare(train_rows, "train"), validation=prepare(eval_rows, "eval"))
    if cache_dir:
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(cache_dir))
        (cache_dir / "pipeline-metadata.json").write_text(
            json.dumps(
                {
                    "manifest": str(config.data.manifest),
                    "checkpoint": config.model.checkpoint,
                }
            ),
            encoding="utf-8",
        )
    return dataset


def _versioned_cache_dir(
    config: ProjectConfig, rows: list[ManifestRow]
) -> Path | None:
    """Derive a cache key from configuration, manifest bytes, and audio file metadata."""
    if config.data.cache_dir is None:
        return None
    digest = hashlib.sha256()
    digest.update(config.data.manifest.read_bytes())
    digest.update(config.model.checkpoint.encode())
    digest.update(config.model.speaker_encoder.encode())
    digest.update(str(config.data.sampling_rate).encode())
    digest.update(str(config.data.max_text_tokens).encode())
    for row in rows:
        try:
            stat = row.audio_path.stat()
        except FileNotFoundError:
            continue
        digest.update(str(row.audio_path).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return config.data.cache_dir / digest.hexdigest()[:16]
