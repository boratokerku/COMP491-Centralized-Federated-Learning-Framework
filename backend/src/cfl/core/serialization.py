"""Weight (de)serialization with safetensors. Never unpickle data received over the network."""

from __future__ import annotations

import hashlib

import torch
from safetensors.torch import load, save

StateDict = dict[str, torch.Tensor]


def state_dict_to_bytes(weights: StateDict) -> bytes:
    return save({k: v.detach().cpu().contiguous() for k, v in weights.items()})


def bytes_to_state_dict(data: bytes) -> StateDict:
    return load(data)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
