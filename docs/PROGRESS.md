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

## M3 — Full-model parity (done)

**Built:** `tools/compare_ort.py` (ORT with every intermediate tapped, byte comparison in execution order, first-divergence report,
Mode B isolated and cumulative deviation, full-test-set accuracy) and `tests/test_compare_ort.py`.
Run: `python -m tools.compare_ort --model cifar` and `--model kws`. The KWS model is Project 2's (`models/kws/kws/SOURCE.json`, commit f7b2a39).

**Numerical correctness (float mode vs ORT 1.31.1; `results/parity_<model>.json`):**

| Model | Inputs | Layers | Tensor comparisons exact | Final logits exact (whole test set) |
| --- | --- | --- | --- | --- |
| CIFAR CNN | 100 | 11 | 1100 / 1100 | 10,000 / 10,000 |
| KWS DS-CNN | 100 | 15 | 1500 / 1500 | 4,888 / 4,888 |

**Accuracy (`results/accuracy_<model>.json`, whole test sets):**

| Model | ORT FP32 | ORT INT8 | int8rt float | int8rt fixed |
| --- | --- | --- | --- | --- |
| CIFAR (10,000) | 76.90% | 76.94% | 76.94% | 76.88% |
| KWS (4,888) | 92.82% | 92.78% | 92.78% | 92.74% |

int8rt float accuracy equals ORT INT8 on both models. KWS ORT logits are also identical to the logits Project 2 stored.

**Mode B (fixed-point) deviation from Mode A:**
- *Isolated*, per node: QLinearConv differs in 0.04-0.27% of elements, never by more than 1 LSB. CIFAR's QLinearGlobalAveragePool differs in 6.97% of elements (1 LSB), the largest single-node effect. (Likely cause: averaging produces many exact ties, and float mode rounds half to even while gemmlowp rounds half up; I have not isolated this by experiment.)
- *Cumulative*: differences compound: max 2 LSB (CIFAR conv), up to 9 LSB (KWS last conv layers); final logits differ in 11.5% (CIFAR, max 0.12) and 22.6% (KWS, max 0.44) of elements.
- *End to end*: predicted class changes on 45/10,000 CIFAR inputs (accuracy -0.06 points) and 18/4,888 KWS clips (-0.04 points).

**Checks (local):** `ruff`, `mypy` clean; `pytest`: 385 passed (10 new); CI still has not run (no GitHub remote).

**Caveats**
- Per-layer parity covers the first 100 inputs per model (the spec's minimum); only the final logits are checked on every test input.
- Results JSON records `+dirty` commits because earlier results files were untracked while later runs started; the code commit is `3ec7e48`. The `what` label in `accuracy_cifar.json` was corrected by hand after the run (a tool bug kept the old M0 label, fixed in the tool afterwards); no numbers were edited.
- Host CPU wall times in the JSON are incidental, not performance claims.

## M4 — Memory planner (done)

**Built:** `int8rt/planner.py` (liveness, greedy-by-size arena, lower bound, overlap check), `Executor(planner="greedy")` with real arena views, `int8rt/cli.py` (`run`, `memreport`), `tools/run_memory_report.py`, `tests/test_planner.py`.
Run: `python -m tools.run_memory_report --model cifar` and `--model kws`; `python -m int8rt memreport --model <onnx>`.

**Activation memory (`results/memory_<model>.json`; a property of the graph, not a host measurement):**

| Model | Naive KiB | Planned KiB | Lower bound KiB | Naive to planned | Weights KiB |
| --- | --- | --- | --- | --- | --- |
| CIFAR CNN | 123.17 | 76.00 | 76.00 | -38.3% | 66.28 |
| KWS DS-CNN | 104.33 | 41.00 | 40.98 | -60.7% | 26.73 |

Activations include the pinned float32 graph input (12 KiB CIFAR, 1.9 KiB KWS). KWS planned is 24 bytes over the bound, all of it the 64-byte round-up of the arena (DESIGN D22).

**Planner on vs off:** on 100 real test inputs per model and both requantisation modes, final outputs are byte-identical (100/100 each), and so are all intermediate tensors on the first 5 inputs (CIFAR 60/60, KWS 80/80 per mode), with the arena refilled with random bytes before every run. No live tensors overlap in either plan.

**Checks (local):** `ruff`, `ruff format`, `mypy` clean; `pytest`: 402 passed (17 new; the old "greedy not available" test was removed). CI has still never run (no GitHub remote).

**Caveats**
- Both models are straight chains, so greedy meeting the lower bound is expected; it is not evidence for branchy graphs.
- `im2col_scratch_kib` describes this NumPy kernel only (1152 KiB for CIFAR, far above the activations); it is not part of the arena.
- No in-place reuse and no Flatten aliasing (DESIGN D21), so planned can be higher than a more aggressive planner would give.
- `memory_kws.json` records `+dirty` because `memory_cifar.json` was untracked at that moment; the code commit is `e9a5826`.

## M5 — Accelerator cost model (done)

**Built:** `costmodel/mapping.py`, `costmodel/model.py`, `costmodel/sweep.py`, `costmodel/configs/base.yaml`, `tools/run_cost_model.py`, `tests/test_costmodel.py`, `docs/RESULTS.md` (draft for review).
Run: `python -m tools.run_cost_model --model cifar` and `--model kws`.

**Modelled accelerator cost, base config (not measured; `results/cost_base_<model>.json`):**

| Model | Scheme | Compute cycles | No overlap | Double buffered | DMA stalls cut | Utilisation (DB) | Compute-bound / transfer-bound layers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CIFAR CNN | WS | 99,217 | 150,877 | 110,848 | 77.49% | 86.3% | 3 / 7 |
| CIFAR CNN | OS | 99,217 | 166,815 | 109,583 | 84.67% | 87.3% | 3 / 7 |
| KWS DS-CNN | WS | 29,018 | 59,014 | 42,483 | 55.11% | 24.4% | 4 / 11 |
| KWS DS-CNN | OS | 29,018 | 62,678 | 43,747 | 56.24% | 23.7% | 4 / 11 |

Per-layer tables, sweep tables and the draft interpretation are in `docs/RESULTS.md`; plots and CSVs are in `results/`.
KWS depthwise layers run at 5.6% utilisation (one of 16 rows busy).

**Checks (local):** `ruff`, `ruff format`, `mypy` clean; `pytest`: 421 passed (19 new, hand-computed cost cases included). CI has still never run (no GitHub remote).

**Caveats**
- Layers are modelled in isolation (every layer's input and output goes through DRAM); this probably overstates transfer-boundness.
- The no-overlap column reuses the DB-optimal tiling, so "DMA stalls cut" is an upper estimate (DESIGN D27).
- My reading of the spec's double-buffering formula differs slightly from its literal wording (DESIGN D27); please confirm.
- Every machine parameter is an assumption; none comes from a real chip. Results JSON records `+dirty` commits because new results files were untracked when the runs started; the code commit is `ae9388d`.
