# Troubleshooting

## CUDA out of memory

Set `per_device_train_batch_size: 1`, increase `gradient_accumulation_steps`, keep gradient
checkpointing enabled, and shorten/remove unusually long clips. Restart the container after an OOM to
release memory.

## `docker compose` cannot see the GPU

Confirm `nvidia-smi` works on the host and the NVIDIA Container Toolkit is installed. Test Docker with
an NVIDIA CUDA image before debugging this repository. Docker Desktop on macOS cannot expose an NVIDIA
CUDA GPU; use the CPU services for validation and a Linux CUDA host for real training.

## SpeechBrain download or symlink error

Ensure `.cache` is writable and that outbound access to Hugging Face is available. In restricted
environments, pre-download `speechbrain/spkrec-xvect-voxceleb`, the SpeechT5 checkpoint, and vocoder to
the mounted cache.

## Validator reports digits

Write numbers as spoken words. For example, use `twenty twenty six`, not `2026`. Apply one consistent
normalization policy to train, validation, and inference text.

## Loss becomes NaN

Check for clipped, empty, near-silent, corrupt, or extreme-duration clips. Temporarily disable FP16,
lower the learning rate, and reproduce with a small subset. Do not suppress NaNs without finding the
bad data or unstable configuration.

## Generated voice is wrong or unstable

Try several clean reference clips from the authorized speaker. Verify the correct checkpoint and
processor were loaded together. Improve transcript accuracy and recording consistency, then train
longer while monitoring held-out samples. SpeechT5's English-pretrained x-vector conditioning may be
weak for some languages and voices.

## Cache appears stale

Remove only the relevant subdirectory under `artifacts/preprocessed` and rerun. The cache is versioned
from the manifest, configuration, and audio size/mtime; manual removal is useful when file metadata was
preserved despite a content change.
