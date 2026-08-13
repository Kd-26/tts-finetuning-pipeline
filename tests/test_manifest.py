import csv
from pathlib import Path

import pytest

from tts_finetune.manifest import read_manifest, split_rows


def write_manifest(path: Path, rows: list[list[str]], header: list[str] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header or ["audio_path", "text", "speaker_id", "split"])
        writer.writerows(rows)


def test_relative_paths_and_splits(tmp_path: Path) -> None:
    manifest = tmp_path / "metadata.csv"
    write_manifest(
        manifest,
        [
            ["wavs/a.wav", "hello", "speaker", "train"],
            ["wavs/b.wav", "world", "speaker", "validation"],
        ],
    )
    rows = read_manifest(manifest)
    train, evaluation = split_rows(rows)
    assert rows[0].audio_path == (tmp_path / "wavs/a.wav").resolve()
    assert len(train) == 1
    assert len(evaluation) == 1


def test_missing_column_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "metadata.csv"
    write_manifest(manifest, [], header=["audio_path", "text", "split"])
    with pytest.raises(ValueError, match="speaker_id"):
        read_manifest(manifest)


def test_unknown_split_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "metadata.csv"
    write_manifest(manifest, [["a.wav", "hello", "speaker", "dev"]])
    with pytest.raises(ValueError, match="Invalid split"):
        read_manifest(manifest)
