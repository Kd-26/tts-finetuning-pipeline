# Data layout

The training manifest belongs at `data/manifest.csv` by default. Do not commit private recordings.

```text
data/
├── manifest.csv
└── wavs/
    ├── clip_0001.wav
    └── clip_0002.wav
```

Required columns:

| Column | Meaning |
|---|---|
| `audio_path` | Absolute path or path relative to the manifest |
| `text` | Exact spoken transcript, with numbers written as words |
| `speaker_id` | Stable non-sensitive speaker label |
| `split` | `train`, `validation`, or `test` |

The tracked `demo/` data is a set of generated tones for software tests only. It contains no human
voice and cannot train a useful TTS model.
