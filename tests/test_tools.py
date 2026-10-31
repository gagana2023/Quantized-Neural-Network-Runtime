"""M0 tests: model definition, op listing and the export/quantise pipeline on a tiny model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from tools import envinfo, list_ops
from tools.export_quantize import export_quantize, ort_session
from tools.train_cifar_cnn import CifarCNN


def test_cifar_cnn_shape_and_depth() -> None:
    model = CifarCNN().eval()
    assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 10)
    assert sum(isinstance(m, nn.Conv2d) for m in model.modules()) == 4
    assert sum(isinstance(m, nn.Linear) for m in model.modules()) == 1


def test_export_quantize_and_list_ops(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 8 * 8, 5)
    )
    rng = np.random.default_rng(0)
    calib = [rng.standard_normal((1, 3, 8, 8)).astype(np.float32) for _ in range(8)]
    fp32, int8 = export_quantize(model, torch.zeros(1, 3, 8, 8), calib, tmp_path, "tiny")

    counts = list_ops.count_ops(onnx.load(str(int8)))
    assert counts[("ai.onnx", "QuantizeLinear")] == 1
    assert any(op == "QLinearConv" for _, op in counts)

    x = calib[0]
    for path in (fp32, int8):
        out = ort_session(path).run(None, {"input": x})[0]
        assert out.shape == (1, 5)
        assert out.dtype == np.float32


def test_envinfo_has_required_fields() -> None:
    info = envinfo.collect("pytest")
    for key in ("date", "git_commit", "cpu", "os", "python", "packages", "command"):
        assert key in info
    for pkg in ("numpy", "onnx", "onnxruntime"):
        assert info["packages"][pkg] != "not installed"
