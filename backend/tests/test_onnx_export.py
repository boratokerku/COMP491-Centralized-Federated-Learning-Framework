import io

import numpy as np
import onnxruntime as ort
import pytest
import torch

from cfl.core.contract import initial_weights
from cfl.tasks.tabular.model import INPUT_DIM, BankMLP
from cfl.tasks.text.model import SmsHashEmbed
from cfl.tasks.text.tokenizer import encode_batch


def _export_and_run(model: torch.nn.Module, x: torch.Tensor, dynamic_axes: dict) -> np.ndarray:
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        (x,),
        buf,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    session = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    return session.run(None, {"input": x.numpy()})[0]


@pytest.mark.parametrize("batch", [1, 5])
def test_tabular_onnx_matches_pytorch(batch):
    model = BankMLP().eval()
    model.load_state_dict(initial_weights(BankMLP, seed=0))
    x = torch.randn(batch, INPUT_DIM, generator=torch.Generator().manual_seed(1))
    onnx_out = _export_and_run(model, x, {"input": {0: "batch"}, "logits": {0: "batch"}})
    np.testing.assert_allclose(onnx_out, model(x).detach().numpy(), atol=1e-5)


def test_text_onnx_matches_pytorch():
    model = SmsHashEmbed().eval()
    model.load_state_dict(initial_weights(SmsHashEmbed, seed=0))
    x = torch.from_numpy(encode_batch(["WINNER! claim your free prize now", "see you at 6", ""]))
    onnx_out = _export_and_run(
        model, x, {"input": {0: "batch", 1: "seq"}, "logits": {0: "batch"}}
    )
    np.testing.assert_allclose(onnx_out, model(x).detach().numpy(), atol=1e-5)
