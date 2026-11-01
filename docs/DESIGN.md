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
- **Why:** this is my best understanding of how ORT's QLinearConv computes its output scale.
- **Verified in M2** (see `docs/MISMATCHES.md`, `results/ort_probe.json`): 0 wrong elements out of 6,000,000 swept accumulators, against 2-6 for each alternative. The evidence is rare events, so it is consistent rather than overwhelming; `(x_scale / y_scale) * w_scale` was the closest alternative (4 wrong).

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

## M2

### D13. Internal graph format
- **Decision:** `Graph(nodes, tensors, inputs, outputs)`; `Node(name, op_type, domain, inputs, outputs, attrs)`; `TensorInfo(name, shape, dtype, scale, zero_point, data)`. Initialisers are tensors with `data` set. An activation tensor's `scale` / `zero_point` are filled in by the importer from the node that produces it (MaxPool and Flatten inherit from their input); float tensors have none.
- **Alternatives:** keep the `onnx.ModelProto` and query it; store scale / zero point only on nodes.
- **Why:** later milestones (planner, cost model) need shapes, dtypes and byte sizes without touching ONNX, and parity tooling needs each tensor's quantisation parameters to compare dumps.

### D14. Op registry and kernel signatures
- **Decision:** `int8rt/ops.py` holds `OPS[(domain, op_type)] -> OpDef(allowed attrs, shape/dtype rule, prepare)`. `prepare` resolves weights, scales, zero points and attributes once and returns a `Runner(dynamic_inputs, out)`. The importer and the executor both use this one table.
- **Deviation from the spec's interface:** the spec shows one kernel signature for every op. Kernels with weights keep the M1 signature `(x, w, bias, attrs, qparams, requant, out)`; kernels without weights drop what they do not have (`max_pool(x, attrs, out)`, `quantize_linear(x, scale, zp, out)`, `qlinear_global_average_pool(x, x_scale, x_zp, y_scale, y_zp, requant, out)`). Padding the unused arguments would add noise and no safety. The common signature that matters is the `Runner`.
- **Constraint:** scales, zero points and weights must be initialisers (constants); every supported op has exactly one dynamic input, and the importer/prepare step raises `UnsupportedError` otherwise.

### D15. Shapes come from int8rt's own rules; GAP multiplier order
- **Decision:** the importer computes every output shape and dtype with per-op rules in `ops.py`, then compares with the file's `value_info` where the file declares one and raises on disagreement.
- **Alternatives:** `onnx.shape_inference` (does not know the com.microsoft ops, so it cannot shape QGemm or QLinearGlobalAveragePool).
- **Side effect:** the single-op test models declare their output shape independently, so every parity case also tests the shape rules.
- **QLinearGlobalAveragePool:** `acc = sum(x) - x_zp * H * W` (int32), `M = x_scale / (y_scale * H * W)` in float32, then the shared requantiser, so it also honours `requant="fixed"`. 30/30 cases bit-exact in float mode.

### D16. Executor: naive buffers, returned outputs are copies
- **Decision:** one preallocated `np.empty` per node output; kernels write into it; `run` returns copies of the graph outputs. `planner="greedy"` raises `NotImplementedError` until M4, when the buffers become arena views (copying the outputs is what will make that safe).
- `dump_dir` writes `NNN_<sanitised name>.npy` for every input and intermediate plus `index.json` (original name -> file, in execution order), because ONNX tensor names contain `/` and `:`.

### D17. Single-op parity tests
- **Decision:** `tools/make_op_tests.py` builds 300 seeded one-node ONNX models: QLinearConv 80 (normal, depthwise, grouped, pointwise, K > 1000 with extreme data so |acc| can pass 2^24; strides 1-3, asymmetric pads, per-tensor and per-channel scales, non-zero weight zero points, optional bias), QGemm 30, QLinearMatMul 20, QLinearGlobalAveragePool 30, MaxPool 25, QuantizeLinear 25, DequantizeLinear 15, Flatten 10, ReduceMean 25, plus three sweep families (QGemmSweep 20, QLinearConvSweep 10, QuantizeSweep 10).
- **Why the sweeps:** with random data, a 1-ULP difference in `M` almost never changes an output, so a wrong multiplier order passed all 80 random conv cases. The sweeps use `x == x_zp` so that `acc == bias`, which lets the bias vector carry hundreds of thousands of arbitrary accumulators per model. A wrong multiplier order (and `x * (1/scale)` for QuantizeLinear) then fails 2 of 30 and 2 of 10 sweep cases respectively; this was checked once by hand, it is not a committed test.
- **Details that matter:** models need `ir_version = 10` (the default for onnx 1.23 is 14, which the pinned ORT rejects); the ORT reference runs with graph optimisations disabled (D3); comparison is on bytes, so float outputs are compared bit for bit.
- Fixed-point mode (Mode B) is *not* asserted against ORT here (it is expected to differ); its deviation is a M3 measurement.

### D18. Scope choices in M2
- **Ops:** exactly the ops found by `list_ops` in the two models (QuantizeLinear, DequantizeLinear, QLinearConv, QGemm, QLinearGlobalAveragePool, MaxPool, Flatten, ReduceMean), plus QLinearMatMul (the M1 kernel; 2-D only, constant B). QLinearAdd, QLinearAveragePool and Reshape are in the spec's "likely" list but appear in neither model, so they are not implemented; the importer reports them as unsupported if a model contains them.
- **ReduceMean:** only a contiguous run of trailing axes (see MISMATCHES M-001).
- **Not done:** `cli.py` (`python -m int8rt run|memreport`) is not part of the M2 definition of done; `run` needs nothing new and arrives with the planner in M4.
