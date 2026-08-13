#!/usr/bin/env python3
"""Download LJSpeech 1.1 and convert its metadata to this project's manifest."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    members: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"Unsafe archive member: {member.name}")
        if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
            raise ValueError(f"Unsupported archive member: {member.name}")
        members.append(member)
    archive.extractall(destination, members=members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--limit", type=int, default=None, help="Optional number of examples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required to convert LJSpeech audio to 16 kHz mono WAV")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / "LJSpeech-1.1.tar.bz2"
    dataset_dir = args.output_dir / "LJSpeech-1.1"

    if not dataset_dir.is_dir():
        if not archive_path.is_file():
            print(f"Downloading {URL} to {archive_path}")
            urllib.request.urlretrieve(URL, archive_path)
        with tarfile.open(archive_path, "r:bz2") as archive:
            safe_extract(archive, args.output_dir)

    records: list[tuple[str, str]] = []
    with (dataset_dir / "metadata.csv").open(encoding="utf-8") as handle:
        for line in handle:
            clip_id, _, normalized_text = line.rstrip("\n").split("|", maxsplit=2)
            records.append((clip_id, normalized_text))
    random.Random(args.seed).shuffle(records)
    if args.limit:
        records = records[: args.limit]

    converted_dir = args.output_dir / "LJSpeech-1.1-16k" / "wavs"
    converted_dir.mkdir(parents=True, exist_ok=True)
    for index, (clip_id, _) in enumerate(records, start=1):
        source = dataset_dir / "wavs" / f"{clip_id}.wav"
        target = converted_dir / f"{clip_id}.wav"
        if not target.is_file():
            subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(target),
                ],
                check=True,
            )
        if index % 500 == 0 or index == len(records):
            print(f"Converted {index}/{len(records)} clips", flush=True)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["audio_path", "text", "speaker_id", "split"])
        validation_count = max(1, round(len(records) * 0.05))
        for index, (clip_id, text) in enumerate(records):
            audio = (converted_dir / f"{clip_id}.wav").resolve()
            split = "validation" if index < validation_count else "train"
            writer.writerow([audio, text, "ljspeech", split])
    print(f"Wrote {len(records)} rows to {args.manifest}")


if __name__ == "__main__":
    main()
