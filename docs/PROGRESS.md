# Progress

## M0 — Setup and models (done)

**Built:** `pyproject.toml`, pinned `requirements.txt`, GitHub Actions CI (ruff, mypy, pytest),
`tools/train_cifar_cnn.py`, `tools/export_quantize.py`, `tools/list_ops.py`, `tools/envinfo.py`.

**Checks (local, Python 3.12.10):** `ruff check` clean, `ruff format --check` clean, `mypy` clean,
`pytest`: 3 passed. CI has not run yet because there is no GitHub remote; the workflow runs the same three commands.

**Results (all from committed scripts, see `results/`):**

| Quantity | Value | File |
| --- | --- | --- |
| FP32 test accuracy, PyTorch, 8 epochs | 76.90% | `train_cifar.json` |
| FP32 test accuracy, ORT (10,000 images) | 76.90% | `accuracy_cifar.json` |
| INT8 test accuracy, ORT (10,000 images) | 76.94% | `accuracy_cifar.json` |

Ops in the quantised CIFAR model (`results/ops_cifar.txt`): QuantizeLinear 1, QLinearConv 4, MaxPool 2,
QLinearGlobalAveragePool 1, Flatten 1, com.microsoft QGemm 1, DequantizeLinear 1.
ReLU nodes do not appear: ORT folded them into QLinearConv (as a clip on the output range).

**Notes / open points**
- The CNN is deliberately small and trained for only 8 epochs; accuracy is a baseline for parity, not a target.
- The KWS model from Project 2 is already in `models/kws/kws/` (`SOURCE.json`); it is first used in M3.
- ORT warns that QOperator with QInt8/QInt8 is slow on x64. That is expected; the spec requires this format.
- INT8 is 0.04 points above FP32 on this test set; that is noise, not a real gain.
