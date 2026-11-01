# Design decisions

Each entry: the decision, the alternatives, and why.

## M0

### D1. Python 3.12 venv, exact pins
- **Decision:** develop and run CI on Python 3.12; all packages pinned with `==` in `requirements.txt`.
- **Alternatives:** the system Python 3.14 (works, but torch/onnx wheels are less battle-tested there); loose pins.
- **Why:** ORT parity is defined against the pinned onnxruntime, so the environment must be reproducible. pyproject allows >=3.10 for the runtime itself (numpy/onnx/pyyaml only).

### D2. Legacy TorchScript ONNX exporter (`dynamo=False`)
- **Decision:** `torch.onnx.export(..., opset_version=17, dynamo=False)`.
- **Alternatives:** the default torch.export-based exporter.
- **Why:** the spec requires opset 17; the dynamo exporter targets newer opsets and may emit local functions. The legacy exporter gives plain static graphs with BatchNorm folded into Conv in eval mode. It is deprecated, so this may need revisiting.

### D3. ORT reference session has graph optimisations disabled
- **Decision:** `ORT_DISABLE_ALL` for every reference run.
- **Why:** the reference must execute exactly the nodes in the file, so intermediate tensors can be compared node by node (M3). Note: the spec also says to expose intermediates by adding node outputs to graph outputs.

### D4. Input normalisation lives outside the model
- **Decision:** mean/std normalisation is done in the data transform; the ONNX model takes the normalised float tensor.
- **Why:** keeps the graph minimal (no Sub/Div before the first QuantizeLinear) and makes inputs for int8rt identical to what ORT sees.

### D5. CIFAR CNN architecture
- 4 x (3x3 conv, no bias, BN, ReLU), 2 max-pools, global average pool, 1 FC (64 -> 10). Global average pool instead of flatten keeps the FC small and exercises the average-pool op.

### D6. Ops found in the quantised CIFAR model
- QuantizeLinear, QLinearConv, MaxPool, QLinearGlobalAveragePool, Flatten, com.microsoft.QGemm, DequantizeLinear. The int8rt op list is exactly this plus whatever the KWS model adds (checked in M2).
- ReLU is absorbed by ORT's quantiser into the QLinearConv output range, so no separate relu kernel is needed for this model.
- The model's `opset_import` lists many extra ORT domains (nchwc, training, ...); the importer must ignore opset domains and look only at node domains.

## M1

### D7. Kernel signature and parameter objects
- **Decision:** every kernel is `kernel(x, w, bias, attrs, qparams, requant, out) -> None`. `attrs` is a small frozen dataclass (`ConvAttrs`, `GemmAttrs`); `qparams` is a `QParams` (float32 scales, int zero points); the result is written into `out`.
- **Alternatives:** dicts of ONNX attributes; kernels that return new arrays.
- **Why:** typed attrs are checked by mypy; writing into `out` is what the planner needs in M4 (the output is a view into the arena). The importer (M2) builds these objects from ONNX nodes.

### D8. Multiplier is computed in float32 as `(x_scale * w_scale) / y_scale`
- **Decision:** `M` is float32, evaluated in that order (`requant.multiplier`).
- **Alternatives:** float64, or `x_scale * (w_scale / y_scale)`.
- **Why:** this is my best understanding of how ORT's QLinearConv computes its output scale. It is **not yet verified**: M2's single-op tests decide it, and any change goes in `docs/MISMATCHES.md`.

### D9. Mode A (float): float32 multiply, `np.rint`, clamp in float domain
- `acc` is converted to float32 first (exact only up to 2^24; larger accumulators lose low bits), multiplied by float32 `M`, rounded half to even, clamped to `[-128 - y_zp, 127 - y_zp]` and then shifted by `y_zp`. Clamping before the integer cast avoids undefined casts of huge values.

### D10. Mode B (fixed): gemmlowp's two-step rounding, as the spec asks
- **Decision:** `M = m0 * 2^-31 * 2^exp` with `m0` an int32 in [2^30, 2^31). Apply `SaturatingRoundingDoublingHighMul` (round at bit 31) and then `RoundingDivideByPOT` (round at the shift), in int64 NumPy. `M >= 1` uses a saturating left shift first.
- **Consequence, covered by a unit test:** two roundings can differ from one exact rounding. With `M = 0.25`, `acc = 25`: 25 / 2 = 12.5 -> 13, then 13 / 2 = 6.5 -> 7, but the exact 6.25 rounds to 6. Ties also differ from Mode A (the high multiply sends ties toward +inf, the shift sends them away from zero, Mode A sends them to even).
- **Open question for the author:** a single rounding of `acc * m0 / 2^(31 - exp)` would remove the double-rounding error. I kept the spec's two-step form because that is what gemmlowp / TFLite hardware does. Say if you want the single-rounding variant added as a third option (it changes numeric behaviour, so I have not done it).

### D11. Convolution: im2col per group, int64 matmul, int32 check
- Cast int8 to int32 before subtracting zero points; pad with zeros *after* the subtraction (equivalent to padding with `x_zp` in the int8 domain); unfold with kh*kw strided-slice copies; `np.matmul` on int64; add bias; `check_int32` raises `OverflowError` if the sum leaves the int32 range; narrow to int32; requantise.
- int64 (not int32) matmul is deliberate so overflow is detected instead of wrapping. NumPy has no BLAS path for integers, so this is slow; M3 will show whether it matters for 100 inputs per model.
- Per-channel weight zero points are supported (broadcast over output channels), although the quantiser emits 0 for int8 weights, so they are exercised only by tests.

### D12. Gemm / MatMul share one kernel
- `qgemm` handles A (M,K) x B (K,N) with `trans_a/trans_b`, optional INT32 bias, per-column `w_scale`/`w_zp`, and `alpha` folded into `M` (QGemm in the CIFAR model has `alpha=1`, `transB=1`). `qlinear_matmul` is a thin wrapper with no bias and default attrs. N-D batched QLinearMatMul is not implemented; M2 adds it only if `list_ops` shows the KWS model needs it.
