from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np
import soundfile as sf

from .config import ProjectConfig, load_config
from .manifest import read_manifest, split_rows


def validate(config: ProjectConfig) -> list[str]:
    rows = read_manifest(config.data.manifest)
    split_rows(rows)
    errors: list[str] = []
    seen_paths: set[str] = set()
    durations: list[float] = []

    for index, row in enumerate(rows, start=2):
        if not row.audio_path.is_file():
            errors.append(f"line {index}: missing audio: {row.audio_path}")
            continue
        key = str(row.audio_path)
        if key in seen_paths:
            errors.append(f"line {index}: duplicate audio path: {row.audio_path}")
        seen_paths.add(key)
        try:
            info = sf.info(row.audio_path)
        except RuntimeError as exc:
            errors.append(f"line {index}: cannot decode {row.audio_path}: {exc}")
            continue
        duration = info.frames / info.samplerate
        durations.append(duration)
        if info.channels != 1:
            errors.append(f"line {index}: expected mono audio, found {info.channels} channels")
        if info.samplerate != config.data.sampling_rate:
            errors.append(
                f"line {index}: expected {config.data.sampling_rate} Hz, found {info.samplerate} Hz"
            )
        if not config.data.min_duration_seconds <= duration <= config.data.max_duration_seconds:
            errors.append(
                f"line {index}: duration {duration:.2f}s outside "
                f"[{config.data.min_duration_seconds}, {config.data.max_duration_seconds}]s"
            )
        if any(char.isdigit() for char in row.text):
            errors.append(f"line {index}: text contains digits; normalize numbers to words")
        try:
            audio, _ = sf.read(row.audio_path, dtype="float32", always_2d=False)
            audio = np.asarray(audio)
            if not np.isfinite(audio).all():
                errors.append(f"line {index}: audio contains NaN or infinite values")
            elif audio.size:
                rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
                clipped_fraction = float(np.mean(np.abs(audio) >= 0.999))
                if rms < 0.001:
                    errors.append(f"line {index}: audio is silent or nearly silent")
                if clipped_fraction > 0.005:
                    errors.append(
                        f"line {index}: {clipped_fraction:.1%} of samples appear clipped"
                    )
        except (RuntimeError, ValueError) as exc:
            errors.append(f"line {index}: failed quality scan: {exc}")

    if not errors:
        splits = Counter(row.split for row in rows)
        speakers = {row.speaker_id for row in rows}
        total_seconds = sum(durations)
        print(
            f"OK: {len(rows)} clips, {len(speakers)} speaker(s), "
            f"{total_seconds / 60:.1f} minutes, splits={dict(splits)}"
        )
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a TTS manifest and its audio files")
    parser.add_argument("--config", default="configs/base.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        errors = validate(load_config(args.config))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
