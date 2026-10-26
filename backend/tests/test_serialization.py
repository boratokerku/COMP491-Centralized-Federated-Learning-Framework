import torch

from cfl.core.serialization import bytes_to_state_dict, sha256, state_dict_to_bytes


def test_roundtrip_and_checksum():
    weights = {"layer.weight": torch.randn(3, 4), "layer.bias": torch.randn(3)}
    data = state_dict_to_bytes(weights)
    restored = bytes_to_state_dict(data)
    assert restored.keys() == weights.keys()
    for k in weights:
        assert torch.equal(restored[k], weights[k])
    assert sha256(data) == sha256(state_dict_to_bytes(restored))
