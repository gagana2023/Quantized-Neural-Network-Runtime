# Numerical mismatches vs ONNX Runtime

Every mismatch found: its cause and its fix. Entries are never deleted once fixed.

Status values: **fixed** (now bit-exact), **restricted** (not reproduced; int8rt refuses the case), **open**.

## M-001: ReduceMean differs from ORT in the last float32 bits (M2) — fixed, with a restriction

- **Found by:** `tools/make_op_tests.py`, first run: 251 of 260 cases matched, and all 9 failures were `ReduceMean`
  (float32 outputs differing by 1 to a few ULP, e.g. ORT `0.4982323944568634` vs int8rt `0.4982324540615082`).
  Every integer op (conv, gemm, matmul, pooling, quantise, dequantise, flatten) matched on the first run.
- **Cause:** float32 addition is not associative, so the *order* of the sum matters. int8rt first used `np.sum` (pairwise).
  ORT sums each contiguous row with Eigen's vectorised `redux`: 4-lane SSE2 packets, two accumulators (even/odd packets),
  a final horizontal add `(l0+l2)+(l1+l3)`, then scalar leftovers. The start of the vectorised part is the first 16-byte-aligned
  element, and the leading scalar elements are added *after* the horizontal add. Row `k` of a `(K, R)` reduction starts
  `k*R` elements into the buffer, so the order differs from row to row.
- **How it was pinned down:** `tools/probe_ort.py` (`results/ort_probe.json`) compares candidate orders with ORT on the 25
  ReduceMean cases: `np.sum` and left-to-right both fail 19 of 25; the Eigen emulation fails 0. Before that, an ad-hoc search over
  lane widths (4, 8, 16) and unroll factors found only SSE2 width 4, 2x unrolled, matches for every length 1..69, 125, 200, 250, 1000.
- **Fix:** `int8rt/kernels/pool.py::_eigen_sum_rows` and `reduce_mean` reproduce that order. After the fix the 25 ReduceMean cases
  match. (An equivalent scratch emulation also matched ORT on 1500 random trailing-axes shapes; that script was not committed.)
- **Restriction:** ORT picks a different reduction routine for other axis patterns (leading, middle, or non-contiguous axes,
  and some shapes containing size-1 dims). An ad-hoc probe reproduced leading-axes and most middle-axes reductions but
  not the rest, so int8rt supports only a contiguous run of trailing axes, which is the only pattern in the models
  (`axes=[2, 3]` of NCHW). The importer rejects anything else with a clear error.
- **Assumption:** the input buffer is 16-byte aligned. This holds for ORT's arena (64-byte) and for NumPy arrays here, and the
  ORT result would otherwise depend on the caller's pointer alignment. M3 checks it on the real KWS model.

## Non-mismatches worth recording (decisions that parity confirmed)

These are not mismatches, but the spec asks for ORT's behaviour to be found empirically (section 3.2), so the evidence is here.
All counts are from `results/ort_probe.json`; `QGemmSweep` feeds arbitrary INT32 accumulators through ORT's requantiser
(x equal to its zero point makes `acc == bias`).

- **Multiplier order.** `M = (x_scale * w_scale) / y_scale` in float32 gave 0 wrong elements out of 6,000,000 accumulators.
  `x_scale * (w_scale / y_scale)` gave 2, `(x_scale / y_scale) * w_scale` gave 4, float64 `M` rounded to float32 gave 2, and a float64
  multiply with float64 `M` gave 6. The effect is rare (a 1-ULP change in `M` only flips a rounding when `acc * M` lies within about
  one ULP of a tie), so the evidence is a handful of elements, consistent but not overwhelming. Random single-op cases alone could
  not tell the orders apart; the sweep was added for that reason. (DESIGN D8 is now verified.)
- **QuantizeLinear** divides: `x / scale` gave 0 wrong out of 2,000,000; `x * (1 / scale)` gave 4.
- **QLinearGlobalAveragePool**: `M = x_scale / (y_scale * H*W)`, accumulator `sum(x) - x_zp*H*W`; 30/30 cases bit-exact (DESIGN D15).

## M3: full-model parity found no new mismatches

Float mode matched ORT bit for bit on every node output of both models for all 100 inputs (CIFAR 1100/1100, KWS 1500/1500 tensor
comparisons) and on the final logits of every test input (10,000 and 4,888). Nothing to log. This also exercised the M-001 alignment
assumption (ReduceMean input buffer 16-byte aligned) inside the real KWS model on 4,888 clips: it held.
Mode B differing from Mode A is expected behaviour of integer-only requantisation, not a mismatch; its size is in `results/parity_<model>.json`.
