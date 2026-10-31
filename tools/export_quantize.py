"""Export a PyTorch model to ONNX and quantise it (spec Section 3.1).

Pipeline: torch.onnx.export (opset 17, batch 1, static shapes) -> ORT quant pre-process
(shape inference + graph optimisation) -> quantize_static (QOperator, QInt8 activations and
weights, per-channel, MinMax calibration on 200 training samples).

`export_quantize()` works for any torch.nn.Module. The CLI wires it up for the CIFAR CNN,
lists the ops of the result and records FP32 / INT8 accuracy measured by ORT.

Usage: python -m tools.export_quantize --arch cifar
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from tools import envinfo, list_ops

ROOT = Path(__file__).resolve().parent.parent
OPSET = 17


class ArrayReader(CalibrationDataReader):
    """Feeds a fixed list of single-sample arrays (shape (1, ...)) to the calibrator."""

    def __init__(self, input_name: str, samples: Iterable[np.ndarray]) -> None:
        self._it = iter([{input_name: s.astype(np.float32)} for s in samples])

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._it, None)


def export_quantize(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    calib_samples: Iterable[np.ndarray],
    out_dir: Path,
    name: str,
) -> tuple[Path, Path]:
    """Return (fp32_path, int8_path). Files: <name>_fp32.onnx, <name>_pre.onnx, <name>_int8.onnx."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32, pre, int8 = (out_dir / f"{name}_{s}.onnx" for s in ("fp32", "pre", "int8"))

    model.eval()
    torch.onnx.export(
        model,
        (example_input,),
        str(fp32),
        input_names=["input"],
        output_names=["logits"],
        opset_version=OPSET,
        dynamo=False,  # legacy TorchScript exporter: plain static ONNX, no external functions
    )
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=False)
    quantize_static(
        str(pre),
        str(int8),
        ArrayReader("input", calib_samples),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
    )
    return fp32, int8


def ort_session(path: Path) -> ort.InferenceSession:
    # Graph optimisations off: the reference must run exactly the nodes in the file.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_accuracy(path: Path, images: np.ndarray, labels: np.ndarray) -> float:
    sess = ort_session(path)
    name = sess.get_inputs()[0].name
    correct = 0
    for img, lab in zip(images, labels, strict=True):
        out = sess.run(None, {name: img[None]})[0]
        correct += int(out.argmax() == lab)
    return correct / len(labels)


def _cifar(args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader

    from tools.train_cifar_cnn import CifarCNN, cifar_transform, load_cifar

    model = CifarCNN()
    model.load_state_dict(torch.load(args.out / "cifar_cnn.pt"))

    train = load_cifar(args.data, train=True)
    # Calibration comes from the *training* set, without augmentation: 200 fixed random samples.
    train.transform = cifar_transform()  # type: ignore[attr-defined]
    idx = np.random.default_rng(args.seed).choice(len(train), 200, replace=False)  # type: ignore[arg-type]
    calib = [train[int(i)][0].numpy()[None] for i in idx]

    fp32, int8 = export_quantize(model, torch.zeros(1, 3, 32, 32), calib, args.out, "cifar")

    ops_txt = ROOT / "results" / "ops_cifar.txt"
    ops_txt.parent.mkdir(exist_ok=True)
    text = list_ops.format_ops(onnx.load(str(int8)), int8.name)
    ops_txt.write_text(text)
    print(text)

    test = load_cifar(args.data, train=False)
    dl = DataLoader(test, 1000)
    images = np.concatenate([x.numpy() for x, _ in dl])
    labels = np.concatenate([y.numpy() for _, y in dl])
    acc_fp32 = ort_accuracy(fp32, images, labels)
    acc_int8 = ort_accuracy(int8, images, labels)
    print(f"ORT FP32 acc={acc_fp32:.4f}  ORT INT8 acc={acc_int8:.4f}")

    result = {
        "env": envinfo.collect("python -m tools.export_quantize " + shlex.join(sys.argv[1:])),
        "what": "Test accuracy measured by ONNX Runtime CPU (graph optimisations disabled)",
        "test_images": int(len(labels)),
        "calibration": {"samples": 200, "method": "MinMax", "source": "CIFAR-10 train set"},
        "quantization": "QOperator, QInt8 activations, QInt8 weights, per-channel",
        "ort_fp32_accuracy": acc_fp32,
        "ort_int8_accuracy": acc_int8,
        "int8rt_accuracy": None,
    }
    (ROOT / "results" / "accuracy_cifar.json").write_text(json.dumps(result, indent=2) + "\n")
    envinfo.write_env_json(command="python -m tools.envinfo")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=["cifar"], default="cifar")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "cifar")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    _cifar(args)


if __name__ == "__main__":
    main()
