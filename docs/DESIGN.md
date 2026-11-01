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
- **Decision:** one preallocated `np.empty` per node output; kernels write into it; `run` returns copies of the graph outputs. `planner="greedy"` (M4, D21-D23) makes the buffers arena views; copying the outputs is what makes that safe.
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

## M3

### D19. How full-model parity is measured
- **Decision:** `tools/compare_ort.py` adds every node output to the ORT graph outputs (`make_empty_tensor_value_info`), runs ORT with graph optimisations disabled so nothing is fused away, runs int8rt with `dump_dir`, and compares the first output of every node as raw bytes (floats by bit pattern) in execution order. Every node is compared even after a divergence, so the report shows whether later nodes recover; the first divergence is reported with node, tensor, element index and both values.
- **Inputs:** per-layer parity uses the first 100 test-set inputs in dataset order (CIFAR: 10,000-image test split; KWS: the 4,888 Project 2 test clips). Final logits are compared byte for byte on the whole test set as part of the accuracy pass.
- **Alternatives:** compare only final logits (hides compensating errors); random inputs (not what the spec asks).
- **Negative control:** `tests/test_compare_ort.py` plants a fault and checks it is reported at the right node, element and values.

### D20. Two ways to measure Mode B deviation
- **Isolated:** each node run in fixed mode on Mode A's input tensors. Only that node's requantiser differs, so this is the cleanest number for "how different is one fixed-point requantiser".
- **Cumulative:** the whole network free-running in fixed mode. This is what a fixed-point accelerator would actually produce; earlier differences propagate and grow.
- Mode A is bit-exact with ORT, so both are also deviations from ORT. D10 (single vs two-step rounding in Mode B) is unchanged and still open for the user.

## M4

### D21. What counts as an activation, and how time is measured
- **Decision:** activations are the graph inputs plus every node output (`int8rt/planner.py`). Initialisers (weights, biases, scales) are reported separately as `weights_kib`; kernel scratch is `im2col_scratch_kib`. Time is the node index; graph inputs are defined at step -1; a tensor's last use is its last consumer; graph inputs and outputs are pinned to the last node, as the spec says.
- **The float32 graph input is in the arena** and is pinned, so `run` copies the caller's array into it. This is honest but costly: on CIFAR it is 12 KiB of the 76 KiB peak. The alternative (leave it outside) would hide that memory and flatter the result.
- **Lifetimes are closed intervals.** A node's output therefore never shares bytes with that node's own inputs. Reusing an input's bytes for the output (in-place) is not attempted: the NumPy kernels read their inputs while writing the output (im2col, pooling windows), so it would be unsafe without per-kernel proofs.
- **Flatten copies.** It could alias its input (same bytes, new shape), which would save a few bytes; it is left as a real copy to keep one rule: every tensor owns its bytes.

### D22. Greedy-by-size, and why the three numbers are what they are
- **Algorithm:** sort by size descending (ties: earlier definition, then name, so plans are deterministic); place each tensor at the lowest 64-byte-aligned offset where it clears every already placed tensor whose lifetime overlaps. Arena size is the highest end rounded up to 64.
- **Naive** = sum of all sizes. **Lower bound** = max over steps of the total live size. Planned >= lower bound always; planned is not guaranteed optimal (this is dynamic storage allocation, NP-hard), because placing large tensors first can fragment space for small ones.
- **Measured:** CIFAR planned == lower bound (76 KiB). KWS planned exceeds it by 24 bytes: 40,000 + 1,960 = 41,960 bytes, rounded up to the 64-byte multiple 41,984. So on these two chains greedy is optimal up to alignment; this says little about branchy graphs such as MobileNetV2 residuals.
- **Alternatives:** a time-ordered first-fit (simpler, worse on mixed sizes); exact ILP (not needed here, not in the spec).

### D23. Real sharing and how it is kept safe
- **Decision:** the arena is one `np.int8` buffer; each tensor is `arena[off:off+nbytes].view(dtype).reshape(shape)`. Float32 tensors are safe because every offset is a multiple of 64, which also covers the 16-byte alignment that ReduceMean parity assumes (MISMATCHES M-001).
- **Dump:** with an arena a later tensor overwrites an earlier one, so `dump_dir` now writes each tensor right after its node runs (not at the end of `run`). `index.json` is written when the run completes.
- **Safety checks:** `find_overlaps` (no two simultaneously-live tensors share bytes; a test also feeds it a deliberately bad plan); outputs are returned as copies (D16); tests fill the arena with random bytes before each run, so a kernel that did not write its whole output would differ from the planner-off result.

### D24. CLI
- `python -m int8rt run --model m.onnx --input x.npy --output y.npy [--requant float|fixed] [--planner naive|greedy] [--dump-all DIR]` and `python -m int8rt memreport --model m.onnx` (JSON, KiB). Single-input / single-output models use `.npy`; otherwise `.npz` keyed by tensor name.
- `im2col_scratch_kib` is only the largest int32 im2col matrix of this runtime's NumPy conv kernel (CIFAR 1152 KiB, KWS 31.25 KiB). It is not a hardware scratchpad and not the total of NumPy temporaries; it is reported so that nobody mistakes the activation arena for the host's whole working set.

## M5

### D25. What the model is, and what it is not
- **Decision:** a generic analytical model in `costmodel/` (`mapping.py` graph -> work, `model.py` tiling and cycle formulas, `sweep.py` sweeps). Config is `costmodel/configs/base.yaml`: 16x16 MACs/cycle, 32 KiB weight SRAM, 64 KiB activation SRAM, 8 B/cycle DMA, 16 vector lanes, 500 MHz. Every value is an assumption chosen to look like a small edge device; none is taken from a real chip.
- **Layer isolation:** each layer reads its input from DRAM and writes its output to DRAM. Cross-layer residency and fusion are not modelled (see RESULTS.md for the likely effect).

### D26. Mapping and array dataflow
- **Decision:** convolution -> per-group GEMM with M = output channels per group (rows of the array), K = Cin/group x kh x kw (streamed, one step per cycle), N = output H x W (columns). A tile of (mt, nt, kt) takes `ceil(mt/R) * ceil(nt/C) * kt` cycles. Depthwise = `groups` GEMMs with M = 1 run one after another; QGemm / QLinearMatMul = GEMM with the constant matrix as the weights. MaxPool, GAP, ReduceMean, QuantizeLinear, DequantizeLinear run on the vector unit (one element-op per lane per cycle; GAP/ReduceMean charge one op per input and one per output); Flatten is free.
- **Consequence:** a layer with M < R or N < C leaves array rows or columns idle. This is what makes depthwise (M = 1) and batch-1 FC (N = 1) slow, and it is deliberate: the spec asks me to show it.
- **Conv input traffic:** a K x N tile of the im2col matrix costs `tile * unique_input / (K*N)` bytes of DRAM traffic (a 3x3 conv reads each pixel once, not nine times). No halo re-fetch between neighbouring tiles.

### D27. Tiling, loop orders, and the two schedules
- **Loop orders:** WS = for mt: for kt: load W once; for nt: stream X. OS = for mt: for nt: for kt: load W and X; store Y once. WS reads weights once and spills INT32 partial sums to DRAM only if K is cut; OS re-reads weights per N tile and keeps accumulators on chip.
- **Capacity:** a tile's weights (+ INT32 bias) must fit half the weight SRAM; its input tile plus INT32 accumulators half the activation SRAM. Half, so the same tiling is double-bufferable. The accumulators are charged to the activation SRAM.
- **Search:** per layer and scheme, over mt in {R x 2^j, M}, nt in {C x 2^j, N}, kt in {2^j, K}; the winner has the fewest double-buffered cycles (ties: fewer no-overlap cycles, fewer DRAM bytes, bigger tiles). Vector layers search their tile count the same way.
- **No overlap** = sum of compute + load + store cycles. **Double buffered** = `load[0] + sum_i max(compute[i], load[i+1] + store[i-1]) + store[last]`: the first tile's load is exposed, then each step costs the longer of its compute and the DMA work running beside it. This is my reading of the spec's "first tile's transfer plus the sum over tiles of max(compute, transfer)": the spec formula taken literally counts tile 0's transfer twice, so I pair each compute with the transfers that actually run during it.
- **Known weakness:** the no-overlap column reuses the DB-optimal tiling, so it is not independently optimised and the "DMA stalls cut" percentages are an upper estimate. It shows up as a non-monotone no-overlap total versus bandwidth.
- **Bound:** compute-bound if a layer's total compute cycles >= its total transfer cycles. This says nothing about utilisation (KWS depthwise layers are "compute-bound" at 5.6% utilisation).
- **Alternatives not taken:** exhaustive tile search (slow, little gain); allocating full SRAM to the no-overlap case (changes tiling between the two columns, which would confound the comparison); a third loop order (input stationary).

### D28. Validation
- `tests/test_costmodel.py` holds hand-computed cases on a 4x4 array with 4 B/cycle DMA: a single tile, two tiles under WS and OS, a K-cut with partial-sum spill, bias loading, a depthwise layer, edge tiles, a vector layer, and the pipeline formula. Further tests check that the search result fits and cannot beat an exhaustive search, that nothing-fits raises, and invariants on both real models (DB <= no overlap, utilisation in (0, 1], every MAC computed).

### D29. Results files and sweeps
- `python -m tools.run_cost_model --model cifar|kws` writes `results/cost_base_<model>.{csv,json,png}`, `_sweeps.{csv,png}`, `_layers.md`, `_summary.md`. Sweeps change one parameter at a time around the base: SRAM (both SRAMs scaled together, 1/4x to 8x), DMA bandwidth (1-64 B/cycle), array shape (4x4 ... 64x64, 8x32, 32x8). The sweep CSV holds every layer at every point.
- **Bug caught on the way:** vector layers first used the largest tile that fits, so a bigger SRAM gave fewer tiles and *less* overlap (totals got worse as SRAM grew). Vector layers now search their tile count like GEMMs; the small remaining SRAM wobble (8 cycles on CIFAR WS) is because the tile-count candidates depend on the SRAM size.

## M6

### D30. README numbers are generated, and the fresh-clone contract
- **Decision:** the README's results section is produced by `tools/make_readme_tables.py` from `results/*.json` (also written to `results/readme_tables.md`) and spliced between two markers. `tests/test_readme.py` fails if the committed README differs from what the script builds, so a table cannot drift from the results files. Prose in the README quotes no numbers except the ones in generated tables.
- **Alternatives:** typing the numbers in by hand (violates the honesty rule and goes stale); a Jinja/Markdown templating dependency (more moving parts than one `split`).
- **Fresh-clone contract:** `models/` and `data/` stay git-ignored (spec section 4: scripts regenerate them; KWS comes from Project 2). A fresh clone therefore installs, passes ruff and mypy, and passes pytest with the model-dependent tests skipped by `skipif`. The README says so, and lists the commands that regenerate the models. The committed `results/` are what the README tables are built from, so they are present on a fresh clone.
- **Stretch goal (MobileNetV2) not attempted:** it needs QLinearAdd, QLinearAveragePool and other ops absent from both required models, i.e. a scope increase the spec says to ask about. The README says it was not done.
