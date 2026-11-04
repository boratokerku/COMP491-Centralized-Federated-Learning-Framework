"""sms_hash_embed: the text model every client trains (managed or independent implementation)."""

from __future__ import annotations

import torch
from torch import nn

from cfl.core.contract import ModelContract
from cfl.tasks.text.tokenizer import NUM_BUCKETS, PAD_ID

ARCHITECTURE_ID = "sms_hash_embed"
ARCHITECTURE_VERSION = "1.0.0"

EMBED_DIM = 64
NUM_CLASSES = 2  # class 1 = spam


class SmsHashEmbed(nn.Module):
    """ids[B, L] -> emb(16384, 64) -> mean over non-padding positions -> fc1(32) -> ReLU
    -> fc2(2 logits). Plain Embedding + masked mean (not EmbeddingBag) so ONNX export stays
    simple."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(NUM_BUCKETS, EMBED_DIM, padding_idx=PAD_ID)
        self.fc1 = nn.Linear(EMBED_DIM, 32)
        self.fc2 = nn.Linear(32, NUM_CLASSES)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != PAD_ID).unsqueeze(-1).to(self.emb.weight.dtype)
        summed = (self.emb(ids) * mask).sum(dim=1)
        pooled = summed / mask.sum(dim=1).clamp(min=1.0)
        return self.fc2(torch.relu(self.fc1(pooled)))


def contract() -> ModelContract:
    return ModelContract.from_model(SmsHashEmbed(), ARCHITECTURE_ID, ARCHITECTURE_VERSION)
