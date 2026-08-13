#!/usr/bin/env python3
"""Generate tiny tonal WAV fixtures. They test I/O only and are not useful speech data."""

from __future__ import annotations

import argparse
import csv
import math
import struct
import wave
from pathlib import Path

SAMPLES = [
    ("demo_001.wav", "This tiny sample checks the training pipeline.", "train", 180.0),
    ("demo_002.wav", "It is not intended to produce a useful voice.", "train", 220.0),
    ("demo_003.wav", "Use clean recordings for a real experiment.", "validation", 260.0),
]


def write_fixture(path: Path, frequency: float, seconds: float = 1.0, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    fade = int(rate * 0.05)
    for index in range(int(seconds * rate)):
        envelope = min(1.0, index / fade, (seconds * rate - index) / fade)
        value = int(0.15 * envelope * 32767 * math.sin(2 * math.pi * frequency * index / rate))
        frames.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames)


def generate(output_dir: Path) -> None:
    wav_dir = output_dir / "wavs"
    for filename, _, _, frequency in SAMPLES:
        write_fixture(wav_dir / filename, frequency)
    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["audio_path", "text", "speaker_id", "split"])
        for filename, text, split, _ in SAMPLES:
            writer.writerow([f"wavs/{filename}", text, "demo_speaker", split])
    print(f"Wrote {len(SAMPLES)} fixtures to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
