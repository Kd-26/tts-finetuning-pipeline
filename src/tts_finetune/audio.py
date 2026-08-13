from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


def load_audio(path: str | Path, sampling_rate: int) -> np.ndarray:
    """Load mono floating-point audio and resample to the model rate."""
    waveform, _ = librosa.load(str(path), sr=sampling_rate, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    peak = float(np.max(np.abs(waveform)))
    if peak > 1.0:
        waveform = waveform / peak
    return waveform


def save_audio(path: str | Path, waveform: np.ndarray, sampling_rate: int) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, np.asarray(waveform, dtype=np.float32), sampling_rate, subtype="PCM_16")


def speaker_embedding(speaker_model: object, waveform: np.ndarray) -> np.ndarray:
    """Create a normalized 512-dimensional SpeechBrain x-vector."""
    tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    with torch.no_grad():
        embedding = speaker_model.encode_batch(tensor)
        embedding = torch.nn.functional.normalize(embedding, dim=2)
    result = embedding.squeeze().cpu().numpy().astype(np.float32)
    if result.shape != (512,):
        raise ValueError(f"Expected a 512-dimensional speaker embedding, got {result.shape}")
    return result
