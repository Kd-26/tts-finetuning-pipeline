from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any
    reduction_factor: int = 2

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        batch = self.processor.pad(
            input_ids=input_ids,
            labels=label_features,
            return_tensors="pt",
        )
        batch["labels"] = batch["labels"].masked_fill(
            batch["decoder_attention_mask"].unsqueeze(-1).ne(1), -100
        )
        del batch["decoder_attention_mask"]

        if self.reduction_factor > 1:
            lengths = torch.tensor([len(item["input_values"]) for item in label_features])
            lengths = lengths - lengths % self.reduction_factor
            batch["labels"] = batch["labels"][:, : int(lengths.max().item())]

        batch["speaker_embeddings"] = torch.tensor(speaker_features, dtype=torch.float32)
        return batch
