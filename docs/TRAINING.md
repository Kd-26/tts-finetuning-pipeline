# Training and evaluation notes

## Before a long run

1. Run `tts_finetune.validate` and resolve every error.
2. Listen to random clips from every speaker and both splits.
3. Run `configs/smoke.yaml` to test model downloads, preprocessing, collation, backward pass, save, and
   evaluation on the actual machine.
4. Start a short real-data experiment (for example, one hundred update steps) and synthesize a fixed
   evaluation sentence before committing to thousands of steps.

The preprocessing cache contains token IDs, log-mel targets, and x-vectors. A key derived from the
manifest, audio file metadata, sample rate, checkpoint, and speaker encoder automatically selects a
version. Content-preserving changes with unchanged size/mtime are unusual but can still require manual
cache removal.

## Memory controls

The effective batch size is:

```text
per_device_train_batch_size × gradient_accumulation_steps × number_of_GPUs
```

On out-of-memory errors, reduce the per-device batch first and increase accumulation to preserve the
effective size. Gradient checkpointing saves activation memory at a compute cost. FP16 is enabled
automatically only on CUDA by the base config. Shorter clips also materially reduce memory usage.

## What gets saved

- `artifacts/speecht5-checkpoints/checkpoint-*`: resumable trainer checkpoints;
- `artifacts/speecht5-final`: final model, processor, and run metadata;
- `artifacts/preprocessed`: cached model inputs and speaker embeddings.

The vocoder and speaker encoder are referenced by model ID and downloaded separately. For an offline
deployment, pin and mirror those assets too.

## Evaluation gates

Loss alone is insufficient. Use a fixed, held-out sentence suite and multiple reference clips. At
minimum assess:

- intelligibility and pronunciation, ideally with human ratings and ASR word error rate;
- speaker similarity, with blinded human comparisons and an embedding metric;
- prosody, pacing, silence, noise, clicks, and truncation;
- memorization by prompting training sentences and near-neighbors;
- performance on numbers written as words, names, questions, abbreviations, and long sentences;
- demographic/language coverage for the intended users;
- misuse controls, disclosure, watermarking/provenance, and access policy.

Do not select a model solely because it sounds convincing on one sentence.

## Reproducibility

The code sets Python, NumPy, PyTorch/Transformers seeds and records package/CUDA details. Complete GPU
determinism is not guaranteed across architectures and kernels. Preserve the Git commit, Docker image
digest, config, manifest hash, and model dependency revisions for an auditable release.
