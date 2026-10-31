# Project 1 Spec: INT8 Inference Runtime with Accelerator Cost Model

Oct 31, 2026 · @Gagana

Build a small, correct Python and NumPy runtime that executes INT8-quantised ONNX models bit-exactly with ONNX Runtime, plans activation memory, and estimates accelerator cost per layer. Correctness first, then memory, then performance modelling. No C++.

## 0. Instructions for Claude Code

Read this whole spec before writing code. Then follow these rules for the entire project.

**Working rules**

1. Implement all milestones in Section 5, in order, without waiting for my go-ahead. After each milestone, run all tests, commit, and append a short summary with test results to docs/PROGRESS.md. Stop and ask only if you are blocked or a decision would change scope or numeric behaviour.
2. Clarity over speed. Kernels use NumPy with explicit integer dtypes. Do not use torch, scipy or ORT inside the runtime; ORT is only the reference.
3. Make small commits with clear messages, one logical change each. Never squash history.
4. Keep `docs/DESIGN.md` up to date: each decision, the alternatives and why it was chosen.
5. Ask before changing scope, the internal graph format, or any numeric behaviour.
6. Python 3.10+ with type hints. Code must pass `ruff` and `mypy`.

**Honesty rules**

1. Never invent or estimate results. Every number in the README must come from a committed script that writes to `results/`.
2. Every results file records: date, git commit, CPU model, OS, Python, numpy, onnx and onnxruntime versions, and the exact command.
3. Keep three kinds of numbers apart and label them: numerical correctness, *modelled* accelerator cost, and *measured* host CPU time. Never mix them up.
4. Do not describe the accelerator as any real company's chip. It is a configurable, generic analytical model.
5. Log every numerical mismatch found, its cause and its fix in `docs/MISMATCHES.md`. Do not delete entries once fixed.

**Explanations for me**

I will not write the code myself, so I need to understand it well enough to explain it in an interview. After each milestone, add to `docs/EXPLAIN_BACK.md`:

- A plain-language explanation of each key piece built in that milestone (for example requantisation, liveness analysis, the cost formula), with a tiny worked example.
- 5 likely interview questions about it, each with a short model answer.

## 1. Goal, non-goals and resume claims

**Goal.** A pure Python and NumPy runtime that loads a static-shape subset of INT8-quantised ONNX models and produces outputs identical, byte for byte, to ONNX Runtime (ORT) on CPU. Add a static activation memory planner and an analytical accelerator cost model on top.

**Target models**

1. The DS-CNN keyword-spotting model from Project 2 (required).
2. A small CIFAR-10 CNN trained in this repo, about 4 conv layers plus FC (required, for a second, vision workload).
3. MobileNetV2 from torchvision (stretch only).

**Non-goals:** dynamic shapes, training, GPU, multi-threading, a general compiler, matching ORT's speed, C++. This project is about numerics, memory and dataflow, not low-level speed.

**Resume claims this repo must support.** Final wording is adjusted to what the results actually show.

| Resume line | Must be proven by |
| --- | --- |
| Built a NumPy INT8 runtime for \<N> ONNX ops with INT32 accumulation and fixed-point requantisation | Op list in README; tests per op; requantisation modes from Section 3.2 |
| Matched ONNX Runtime bit-for-bit on \<N> layers of \<model>; traced \<N> mismatches to \<cause> | `tools/compare_ort.py` report in `results/`; `docs/MISMATCHES.md` |
| Cut peak activation memory \<A>→\<B> KiB via buffer reuse; double buffering cut modelled DMA stalls \<X>% | `results/memory_*.json`; `results/cost_*.csv` and plots |

If ORT parity only holds in float-scale mode (Section 3.2), the first line must say "float and fixed-point requantisation" and state the measured fixed-point deviation honestly.

## 2. Stack

- **Python 3.10+:** numpy, onnx and onnxruntime for the runtime and the reference; torch and torchvision for training and export only; pyyaml, matplotlib, pytest, ruff and mypy.
- **Pinning:** pin exact versions in `requirements.txt` at M0. ORT parity is defined against that pinned onnxruntime version.
- **CI:** GitHub Actions runs `ruff`, `mypy` and `pytest` on every push.

## 3. Key design decisions

### 3.1 Model format and quantisation pipeline

1. Export PyTorch models to ONNX with opset 17, batch size 1 and static shapes.
2. Run `python -m onnxruntime.quantization.preprocess` for shape inference and graph optimisations such as BatchNorm folding.
3. Quantise with `onnxruntime.quantization.quantize_static`: `quant_format=QuantFormat.QOperator`, `activation_type=QuantType.QInt8`, `weight_type=QuantType.QInt8`, `per_channel=True`. Use 200 calibration samples from the training set with MinMax calibration.
4. Write `tools/list_ops.py` to print every op type and domain in the quantised model. **Implement exactly the ops that appear.** Likely candidates are QuantizeLinear, DequantizeLinear, QLinearConv (including depthwise via `group`), QLinearMatMul or com.microsoft QGemm, com.microsoft QLinearAdd, QLinearGlobalAveragePool or QLinearAveragePool, MaxPool, Reshape and Flatten. Check the real list rather than trusting this one.

### 3.2 Numerics

- **Types:** INT8 activations and weights, INT32 bias and accumulators, float32 scales, INT8 zero points. Activation layout is NCHW and weight layout is OIHW, both matching ONNX.
- **Core formula:** `acc = Σ (x − x_zp)(w − w_zp) + bias` in INT32. Then `y = saturate_int8(round(acc × M) + y_zp)`, where `M = x_scale × w_scale[c] / y_scale` for output channel `c`.
- **NumPy dtype discipline:**
  - Cast INT8 inputs to int32 *before* subtracting zero points.
  - Implement convolution as im2col plus `np.matmul` on int32 arrays. Accumulate in int64, assert the result fits in the int32 range, then cast to int32.
  - Never let NumPy silently promote to float64. Add a test that checks the dtype of every kernel's output.
- **Mode A, float requantisation (default):** compute `M` and the multiply in float32, round half to even with `np.rint`, then saturate. This follows the ONNX operator spec and is the mode used for ORT parity. float32 versus float64 and the order of operations can change the last bit. Find ORT's behaviour empirically and record it in `docs/MISMATCHES.md`.
- **Mode B, fixed-point requantisation:** precompute `M` offline as an INT32 multiplier plus a right shift. Apply it with integer-only maths: a rounding high multiply and a rounding right shift, gemmlowp style, implemented in int64 NumPy. This is how integer-only hardware works. Measure its deviation from Mode A: percentage of elements that differ, maximum difference in LSBs, and end-to-end accuracy change on the test set.

### 3.3 Importer

`int8rt/importer.py` loads the `.onnx` file with `onnx.load` and reads initialisers with `onnx.numpy_helper.to_array`. It builds an internal `Graph` dataclass: nodes in topological order, plus tensors with shape, dtype and quantisation parameters. It rejects unsupported ops and dynamic shapes with a clear error that lists them.

### 3.4 Executor

Run nodes sequentially in topological order. Each kernel is a plain function that writes into a preallocated output array. With the planner on, that array is a view into the arena. A `dump_dir` option writes every intermediate tensor as `.npy`, which the parity tool uses.

### 3.5 Memory planner

- **Scope:** activation tensors only. Report weights and scratch separately.
- **Liveness:** each tensor lives from the index of the node that produces it to the index of its last consumer. Graph inputs and outputs are pinned.
- **Real reuse, not just bookkeeping:** the arena is one `np.int8` buffer. Each activation tensor is a NumPy view at its assigned offset, so tensors with non-overlapping lifetimes really share memory.
- **Three numbers to report:**
  - Naive: every tensor gets its own buffer, so peak is the sum of all sizes.
  - Planned: one arena, with offsets assigned greedy-by-size. Sort tensors by size, largest first, and place each at the lowest 64-byte-aligned offset that does not overlap any already-placed tensor with an overlapping lifetime.
  - Lower bound: the maximum, over all steps, of the total size of tensors live at that step.
- **Checks:** outputs are identical with the planner on and off. A test asserts that no two tensors with overlapping lifetimes overlap in memory.

### 3.6 Accelerator cost model

Write this in Python, in `costmodel/`. It reads `the imported Graph` and a YAML config. It is an analytical model, not cycle-accurate, and not a model of any real chip.

- **Config:** MAC array of R×C INT8 MACs per cycle, weight SRAM size, activation SRAM size, DMA bandwidth in bytes per cycle, vector unit lanes and clock frequency.
- **Mapping:**
  - Convolution becomes a GEMM with M = output channels, K = (input channels / group) × kernel height × kernel width, and N = output height × output width.
  - Depthwise convolution maps per group.
  - FC layers are GEMMs.
  - Elementwise ops and pooling run on the vector unit.
- **Loop orders:** tile each layer to fit SRAM and compare at least weight-stationary against output-stationary.
- **Per layer and per scheme, report:**
  - Bytes moved at each memory level.
  - Compute cycles.
  - Transfer cycles, which are bytes divided by bandwidth.
  - Total cycles without overlap, which is compute plus transfer.
  - Total cycles with double buffering, which is the first tile's transfer plus the sum over tiles of max(compute, transfer).
  - Utilisation, which is useful MACs divided by (R × C × total cycles).
  - Whether the layer is compute-bound or transfer-bound.
- **Validation:** unit tests on 2–3 tiny layers with hand-computed expected values.
- **Sweeps:** vary SRAM size, bandwidth and array shape. Write CSV files and plots to `results/`.

## 4. Repo layout and interfaces

```
int8rt/
  pyproject.toml  requirements.txt  README.md
  int8rt/         graph.py  importer.py  requant.py  executor.py  planner.py  cli.py
                  kernels/  conv.py  matmul.py  pool.py  eltwise.py  quant.py
  costmodel/      model.py  mapping.py  sweep.py  configs/*.yaml
  tools/          train_cifar_cnn.py  export_quantize.py  list_ops.py
                  make_op_tests.py  compare_ort.py  run_memory_report.py
  tests/          pytest: one file per component + ORT parity tests
  models/         (gitignored; scripts regenerate them)
  results/        committed JSON/CSV/PNG outputs + environment info
  docs/           DESIGN.md  MISMATCHES.md  EXPLAIN_BACK.md  RESULTS.md
```

**Interfaces (adjust names if needed, keep the roles)**

- `Graph`, `Node` and `TensorInfo` dataclasses: shape, dtype and quantisation parameters.
- `load_model(path) -> Graph`: the importer from Section 3.3.
- `Executor(graph, requant="float" | "fixed", planner="naive" | "greedy")` with `run(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]` and an optional `dump_dir`.
- Kernels: `def qlinear_conv(x, w, bias, attrs, qparams, requant, out) -> None`, and the same shape for every op.

**CLI**

```
python -m int8rt run --model m.onnx --input x.npy --output y.npy
                     [--requant float|fixed] [--planner naive|greedy] [--dump-all DIR]
python -m int8rt memreport --model m.onnx   # naive / planned / lower-bound KiB as JSON
```

## 5. Milestones

Each milestone ends with tests, a commit and a summary in docs/PROGRESS.md, then the next milestone starts. “Done when” is the acceptance test.

**M0 — Setup and models**

- `pyproject.toml`, pinned `requirements.txt`, GitHub Actions CI running ruff, mypy and pytest.
- `tools/train_cifar_cnn.py`: a small CNN with about 4 conv layers, BatchNorm, ReLU, pooling and one FC layer. Train it for a few epochs on CPU and record the test accuracy.
- `tools/export_quantize.py`: the Section 3.1 pipeline for any PyTorch model. Run `list_ops.py` on its output.
- **Done when:** CI is green, `results/ops_cifar.txt` exists, and FP32 and INT8 accuracy as measured by ORT are recorded in `results/`.

**M1 — Requantisation and core kernels**

- Both requantisation modes. QLinearConv (normal and depthwise via `group`, with padding, stride and dilation 1) built on im2col, and QLinearMatMul or QGemm.
- **Tests with hand-computed cases.** Cover negative values, saturation at −128 and 127, non-zero zero points, per-channel scales, odd sizes, padding, stride 2, and an output-dtype check for every kernel.
- **Done when:** all `pytest` tests pass.

**M2 — Importer, executor and single-op parity**

- The importer, the executor, and the remaining ops from `list_ops.py`.
- `tools/make_op_tests.py` builds single-node ONNX models with `onnx.helper`, using random shapes, scales and zero points with fixed seeds, for at least 200 cases across all ops. Each case is run through ORT and int8rt and compared exactly.
- **Done when:** float mode matches 100% of the cases, or every failure is logged in `docs/MISMATCHES.md` with its status.

**M3 — Full-model parity**

- `tools/compare_ort.py` exposes every intermediate tensor in ORT by adding node outputs to the graph outputs. It runs int8rt with `dump_dir`, compares tensor by tensor and reports the first divergent node, the element index and both values.
- Run it on at least 100 real test-set inputs per model: the CIFAR CNN and the KWS DS-CNN from Project 2.
- Measure the fixed-point (Mode B) deviation and its end-to-end accuracy effect.
- **Done when:** float mode is bit-exact on every layer of both models for every input, int8rt's test accuracy equals ORT's, and the results JSON is committed.

**M4 — Memory planner**

- Liveness analysis, the greedy-by-size arena with real NumPy views, and `memreport`.
- **Done when:** `results/memory_<model>.json` reports naive, planned and lower-bound KiB for both models, outputs are identical with the planner on and off, and the overlap test passes.

**M5 — Accelerator cost model**

- Implement Section 3.6, with hand-computed tests and the default config in `costmodel/configs/base.yaml`.
- Sweep SRAM size, bandwidth and array shape. Compare the loop orders, and compare runs with and without double buffering.
- **Done when:** CSV files and plots are in `results/`, and a per-layer table shows which layers are compute-bound or transfer-bound. Claude Code writes the numbers and a draft interpretation in `docs/RESULTS.md`, marked "draft – for my review".

**M6 — Documentation and stretch goals**

- README: what the project is and is not, op support table, install and run commands, a results table generated from `results/`, limitations, and what did not work.
- Optional: MobileNetV2 as a stretch model.
- **Done when:** a fresh clone installs and passes all tests with the README's commands.

## 6. Results, build order and self-check

**Build order across both projects:** Project 2 M1 produces the quantised KWS model, which Project 1 M3 needs. Run P1 M0–M2 → P2 M0–M1 → P1 M3–M4 → P2 M2–M3 → P1 M5–M6 → P2 M4. Copy the KWS `.onnx` file into `models/` and record its source commit.

**Results files to commit**

| File | Contents |
| --- | --- |
| `results/env.json` | CPU, OS, Python and package versions, git commit |
| `results/parity_<model>.json` | Inputs tested, layers compared, exact matches, first mismatch if any, Mode B deviation stats |
| `results/accuracy_<model>.json` | FP32 (ORT), INT8 (ORT) and INT8 (int8rt, both modes) test accuracy |
| `results/memory_<model>.json` | Naive, planned and lower-bound activation KiB; weights KiB |
| `results/cost_<config>.csv` + `.png` | Per-layer cycles, bytes, utilisation and bound type for each scheme and sweep point |

**Fill in the resume from these files only.** Before submitting, check each placeholder against the matching file.

**Interview self-check.** I should be able to answer these without notes:

- Walk through one QLinearConv output element: every multiply, the zero-point handling, the scale and the rounding.
- Why can float and fixed-point requantisation differ by 1 LSB, and when does it matter?
- What was the hardest mismatch, how did I find it, and what was the fix?
- How does greedy-by-size work, why is it not optimal, and how far is it from the lower bound on my models?
- Why is depthwise convolution hard to keep busy on a MAC array? Show it with my numbers.
- Which layers are transfer-bound under the base config, and what change to the hardware would help most?
- What does my cost model ignore, and how could that change the conclusions?
- What did ORT give me, and what did I write myself?
