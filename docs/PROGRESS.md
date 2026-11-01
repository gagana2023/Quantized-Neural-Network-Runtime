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

## M1 — Requantisation and core kernels (done)

**Built:** `int8rt/requant.py` (Mode A float and Mode B fixed-point requantisation, `QParams`, int32 overflow check),
`int8rt/kernels/conv.py` (`qlinear_conv`: im2col, groups/depthwise, padding, stride, per-channel scales and zero points),
`int8rt/kernels/matmul.py` (`qgemm`, `qlinear_matmul`). Dilation other than 1 raises `NotImplementedError`.

**Tests (`tests/test_requant.py`, `tests/test_kernels.py`):** hand-computed cases for negative values, saturation at -128/127,
non-zero input and weight zero points, per-channel scales, odd sizes, padding, stride 2, depthwise (no channel mixing), bias,
tie rounding in both modes, the gemmlowp double-rounding case, int32 overflow detection and im2col layout. Random conv and gemm
cases are checked against an independent naive-loop implementation in both modes. Every kernel's output dtype is asserted int8.

**Checks (local, Python 3.12.10):** `ruff check` and `ruff format --check` clean, `mypy` clean, `pytest`: 43 passed
(40 new). CI still has not run (no GitHub remote).

**Not yet done / caveats**
- No comparison with ORT yet; that is M2. The multiplier order `(x_scale * w_scale) / y_scale` in float32 is an assumption until then (DESIGN D8).
- Mode B deviation from Mode A has only been unit-tested (<= 1 LSB on random accumulators); the real measurement is M3.
- Open decision for you: single vs two-step rounding in Mode B (DESIGN D10).

## M2 — Importer, executor and single-op parity (done)

**Built:** `int8rt/graph.py`, `importer.py`, `ops.py` (op registry), `executor.py`; kernels `quant.py`, `pool.py` (MaxPool,
QLinearGlobalAveragePool, ReduceMean); `tools/make_op_tests.py` (300 seeded single-op models run through ORT and int8rt);
`tools/probe_ort.py` (which float32 formulations ORT uses). Ops implemented: exactly those in the two models (QuantizeLinear,
DequantizeLinear, QLinearConv, QGemm, QLinearGlobalAveragePool, MaxPool, Flatten, ReduceMean) plus QLinearMatMul.
The KWS model adds only ReduceMean (and a second DequantizeLinear/QuantizeLinear pair) to the CIFAR op list.

**Checks (local, Python 3.12.10):** `ruff check` and `ruff format --check` clean, `mypy` clean, `pytest`: 375 passed
(332 new, of which 300 are one test per parity case). CI still has not run (no GitHub remote).

**Results (`results/op_parity.json`, `results/ort_probe.json`; float mode, ORT 1.31.1):**

| Op family | Cases | Bit-exact vs ORT |
| --- | --- | --- |
| QLinearConv | 80 | 80 |
| QLinearConv sweep (100,000 channels each) | 10 | 10 |
| QGemm | 30 | 30 |
| QGemm sweep (300,000 accumulators each) | 20 | 20 |
| QLinearMatMul | 20 | 20 |
| QLinearGlobalAveragePool | 30 | 30 |
| MaxPool | 25 | 25 |
| QuantizeLinear | 25 | 25 |
| QuantizeLinear sweep (200,000 values each) | 10 | 10 |
| DequantizeLinear | 15 | 15 |
| Flatten | 10 | 10 |
| ReduceMean | 25 | 25 |
| **Total** | **300** | **300** |

**Mismatches found:** one, ReduceMean (summation order); fixed, with a restriction to trailing axes. See `docs/MISMATCHES.md` M-001.
The first run was 251/260 before the fix, so the suite does detect real differences. Also logged there: the evidence for ORT's
multiplier order (DESIGN D8 now verified) and for QuantizeLinear dividing by the scale.

**Smoke check beyond the spec:** `tests/test_executor.py` runs both full models through int8rt and ORT on 5 inputs each (random
CIFAR inputs, real KWS features) and requires identical logits bytes; they match. This is only a smoke test. The per-layer parity
over 100 inputs per model and the accuracy comparison are M3.

**Caveats / open points**
- The multiplier-order evidence is a handful of differing elements out of millions (rare events); consistent, not overwhelming.
- `ReduceMean` parity assumes a 16-byte-aligned input buffer; M3 will exercise it inside the real KWS model.
- Mode B is not compared with ORT yet (M3). `cli.py` is not written yet (arrives with the planner, M4).
- Not implemented because they appear in neither model: QLinearAdd, QLinearAveragePool, Reshape, N-D QLinearMatMul, dilation != 1.
- Kernel signatures differ slightly between ops with and without weights (DESIGN D14).
