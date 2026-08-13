from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = {"audio_path", "text", "speaker_id", "split"}
ALLOWED_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class ManifestRow:
    audio_path: Path
    text: str
    speaker_id: str
    split: str


def read_manifest(path: str | Path) -> list[ManifestRow]:
    manifest = Path(path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}. Copy data/manifest.example.csv to "
            "data/manifest.csv and update its paths."
        )

    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        rows: list[ManifestRow] = []
        for line_number, item in enumerate(reader, start=2):
            text = (item.get("text") or "").strip()
            speaker_id = (item.get("speaker_id") or "").strip()
            split = (item.get("split") or "").strip().lower()
            raw_audio_path = (item.get("audio_path") or "").strip()
            if not all((text, speaker_id, split, raw_audio_path)):
                raise ValueError(f"Blank required value on manifest line {line_number}")
            if split not in ALLOWED_SPLITS:
                raise ValueError(
                    f"Invalid split {split!r} on line {line_number}; "
                    f"expected {sorted(ALLOWED_SPLITS)}"
                )
            audio_path = Path(raw_audio_path)
            if not audio_path.is_absolute():
                audio_path = manifest.parent / audio_path
            rows.append(
                ManifestRow(
                    audio_path=audio_path.resolve(),
                    text=text,
                    speaker_id=speaker_id,
                    split=split,
                )
            )
    if not rows:
        raise ValueError(f"Manifest has no examples: {manifest}")
    return rows


def split_rows(rows: list[ManifestRow]) -> tuple[list[ManifestRow], list[ManifestRow]]:
    train = [row for row in rows if row.split == "train"]
    evaluation = [row for row in rows if row.split in {"validation", "test"}]
    if not train:
        raise ValueError("Manifest must contain at least one train row")
    if not evaluation:
        raise ValueError("Manifest must contain at least one validation or test row")
    return train, evaluation
