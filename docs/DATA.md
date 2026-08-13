# Dataset preparation notes

## Permission first

Keep written evidence that every speaker knowingly authorized model training and the intended forms
of use. Permission to publish a recording is not automatically permission to train a voice model.
Define retention, deletion, access, redistribution, and revocation rules before collection. Avoid
names or other personal information in `speaker_id`.

## Recording target

For a single-speaker adaptation, aim for at least one to three hours of clean, accurately transcribed
speech; more varied high-quality data usually improves prosody and coverage. Use the same microphone,
room, gain, and distance across a session. Capture natural variation without shouting, clipping,
background music, reverberation, or aggressive denoising.

Useful clip properties:

- mono PCM WAV;
- 16 kHz for this pipeline (resample once, before training);
- roughly two to twelve seconds per utterance;
- peak below 0 dBFS with no clipping;
- leading/trailing silence trimmed consistently, but not so tightly that phonemes are cut;
- transcript exactly matches spoken words, including disfluencies you intend the model to learn.

Do not train on the included tones. Do not mix unrelated speakers under one ID. For multi-speaker
training, seek balanced hours per speaker and ensure each speaker has held-out evaluation clips.

## Convert audio with FFmpeg

```bash
mkdir -p data/wavs
ffmpeg -i input.wav -ac 1 -ar 16000 -c:a pcm_s16le data/wavs/clip_0001.wav
```

Batch conversion should preserve a mapping back to transcripts. Inspect a random sample by ear after
conversion.

## Text normalization

SpeechT5 tokenization is character-oriented and has no number tokens. Convert `42` to `forty two`,
expand abbreviations consistently, normalize Unicode punctuation, and decide how to pronounce symbols.
Do not silently “correct” a transcript away from what the audio actually says.

The validator catches digits but cannot tell whether the words match the speech. Automated ASR checks
can help find large mismatches, but final quality control should include human listening.

## Splits and leakage

Hold out at least five percent for validation. If several clips came from one long source file, keep
all of those clips in the same split to prevent acoustic leakage. Keep a separate test set when making
model-selection decisions from validation loss. Never train on the reference clips used for a public
similarity evaluation.

## Quality report to keep with a run

Record the source/license, consent status, number of clips, total duration, speakers, sample rate,
language/accent, transcript process, excluded data, train/validation/test rule, and known biases. Hash
the final manifest and preserve it alongside the run configuration.
