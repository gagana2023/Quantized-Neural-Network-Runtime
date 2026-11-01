# Explain-back notes

Plain-language explanations and interview questions, one section per milestone.

## M0 — Setup and models

### What was built
A small CIFAR-10 CNN is trained in PyTorch, exported to ONNX, and quantised to INT8 with ONNX Runtime's
static quantiser. `list_ops` then tells us which ops the runtime must implement. ORT measures FP32 and
INT8 accuracy, which becomes the baseline int8rt must match later.

### Key pieces

**1. Static post-training quantisation (PTQ).**
Each float tensor `r` is stored as an int8 `q` with `r ≈ scale × (q − zero_point)`.
"Static" means the scale and zero point of every *activation* are fixed ahead of time by running
calibration data through the float model and recording each tensor's range (MinMax: the observed
min and max). Weights have known ranges, so they need no calibration data.
*Worked example:* an activation seen in [−2.0, 3.0]. int8 has 256 levels, so scale = 5.0/255 ≈ 0.0196.
Real 0.0 maps to q = round(0/0.0196) + zp, where zp is chosen so −2.0 → −128: zp = −128 − round(−2.0/0.0196) = −26.
So q=−26 means exactly 0.0, which matters because padding and ReLU zeros must be representable.

**2. Per-channel weight quantisation.**
Each output channel of a conv gets its own weight scale (and zero point 0 for int8 weights).
Channels with small weights keep precision instead of being crushed by one large channel.
Cost: requantisation later uses a different multiplier `M[c] = x_scale × w_scale[c] / y_scale` per channel.

**3. QOperator vs QDQ format.**
QDQ keeps float ops wrapped in QuantizeLinear/DequantizeLinear pairs. QOperator fuses them into
integer ops like `QLinearConv` that take int8 in and give int8 out. QOperator is what an integer-only
runtime executes, so that is what we use.

**4. Calibration set.** 200 fixed random *training* images (no augmentation). Using the training set
avoids tuning quantisation on the test set. The seed is fixed so the model is reproducible.

**5. `list_ops`.** Counts nodes by (domain, op_type). The runtime implements exactly this list —
no more — so scope stays small and every op can be tested against ORT.

**6. Why ORT graph optimisations are disabled for the reference.** ORT may fuse nodes (e.g. conv + relu)
at load time. If it does, intermediate tensors vanish and node-by-node comparison is impossible.

### Likely interview questions

1. **What is the difference between symmetric and asymmetric quantisation, and what did you use?**
   Symmetric forces zero point to 0 (range centred on zero); asymmetric allows a non-zero zero point to cover
   skewed ranges (e.g. post-ReLU data). Here both activations and weights are QInt8; ORT's MinMax calibration
   picks the zero point from the observed range, so activations can have non-zero zero points and weights are
   symmetric with zp 0 when per-channel int8 is used.

2. **Why per-channel weights?** Output channels can have very different weight magnitudes; one scale per tensor
   wastes resolution on small channels. Per-channel typically recovers most of the accuracy lost with per-tensor.

3. **Why QOperator rather than QDQ?** QDQ is a float graph with markers; a runtime must fuse them to run integer
   maths. QOperator already has int8-in/int8-out nodes, matching how an integer accelerator executes.

4. **How do you know INT8 accuracy is trustworthy?** It is measured by ORT on all 10,000 test images, using a
   calibration set drawn only from the training data; numbers come from `results/accuracy_cifar.json`, written by
   a committed script together with the environment and exact command.

5. **Why pin exact package versions?** ORT's kernels (and their rounding behaviour) can change between releases.
   Bit-exact parity is only meaningful against a fixed version.

## M1 — Requantisation and core kernels

### What was built
The arithmetic core of the runtime: a convolution and a matrix product that work on int8 data with int32
accumulators, and the step that converts the int32 result back to int8 ("requantisation"), in two flavours.

### Key pieces

**1. One output element of QLinearConv.**
`acc = Σ (x − x_zp)(w − w_zp) + bias`, summed over input channels and the kernel window, all in int32.
Then `y = clamp(round(acc × M) + y_zp, −128, 127)` with `M = x_scale × w_scale[c] / y_scale`.
*Worked example:* x = [4, 6] with x_zp = 2, w = [3, −1] with w_zp = 0, bias = 5. Then (4−2)·3 + (6−2)·(−1) + 5 = 6 − 4 + 5 = 7.
With x_scale = 0.5, w_scale = 0.2, y_scale = 0.1: M = 1.0, so round(7 × 1.0) = 7, and y_zp = −3 gives y = 4.
Check: the real value is x_scale·w_scale·acc = 0.7, and y = 4 decodes to (4 − (−3))·0.1 = 0.7. Same number.

**2. Why cast to int32 before subtracting the zero point.** int8 arithmetic wraps: in int8, 100 − (−100) = −56, not 200. So
inputs are widened first. For the same reason the matmul runs in int64 and the result is then checked against the int32 range.

**3. im2col.** A convolution is a dot product of each kernel with each window. im2col copies every window into one column of a
matrix, so the whole convolution becomes one matrix multiply: weights (O × K) times columns (K × output pixels), with
K = in_channels/group × kh × kw. *Example:* a 3×3 image and 2×2 kernel gives 4 windows, so the column matrix has 4 rows (kernel
positions) and 4 columns (windows). Depthwise convolution is the same thing done once per channel (group = channels).

**4. Padding with the zero point.** Real 0.0 is stored as `x_zp`, not as 0. Padding must mean real zero, so padded pixels
must equal `x_zp` in int8, which is 0 after subtracting `x_zp`. We pad the already-subtracted array with plain zeros.

**5. Mode A, float requantisation.** Multiply the int32 by the float32 `M`, round half to even (2.5 → 2, 3.5 → 4), add `y_zp`,
saturate. This is what the ONNX spec describes and what ORT is expected to do (to be confirmed in M2).

**6. Mode B, fixed-point requantisation (gemmlowp).** Write `M = m0 × 2^−31 × 2^exp` with `m0` an integer in [2^30, 2^31).
*Example:* M = 0.25 = 0.5 × 2^−1, so m0 = 2^30 and exp = −1. Applying it to acc takes two integer steps: multiply by m0 and
shift right by 31 (a "rounding high multiply"), then shift right by 1 (a "rounding right shift"). No floats are needed, which
is how an integer-only accelerator works.

**7. Why the modes can differ by 1.** Mode A rounds once and sends ties to even. Mode B rounds twice (once at each shift) and
sends ties away from zero or toward +inf. *Example:* acc = 25, M = 0.25: exact value 6.25. Mode A gives 6. Mode B gives
25/2 = 12.5 → 13, then 13/2 = 6.5 → 7. It matters when the output is compared bit for bit, and it can shift accuracy slightly;
M3 measures how much on the real models.

### Likely interview questions

1. **Why subtract zero points before multiplying instead of after?**
   `(x − x_zp)(w − w_zp)` is proportional to the product of the real values. Expanding it gives four terms
   (xw, x·w_zp, w·x_zp, zp·zp); hardware often computes the cross terms separately for speed, but a reference implementation
   subtracts first because it is simpler and exact in int32.
2. **What stops int32 overflow in the accumulator?** Nothing in the format itself: the worst case is 255 × 255 × K. We accumulate in int64
   and raise if the result leaves int32 range, so a layer that would overflow real hardware is caught rather than wrapped.
3. **Why is `M` computed per output channel?** Per-channel weight quantisation gives each output channel its own `w_scale`, so
   `M = x_scale × w_scale[c] / y_scale` differs per channel; input and output scales are per tensor.
4. **Why can float and fixed-point requantisation differ by one LSB, and when does it matter?** Different tie rules and two roundings
   instead of one (example above). It matters when you require bit-exact parity with a float reference. For accuracy the error is at
   most 1 LSB per layer and is usually negligible; the M3 results quantify it for these models.
5. **How does im2col handle depthwise and strided convolutions?** Stride only changes which windows are copied (the slice step).
   Depthwise runs the same unfold-and-multiply once per group with one input channel and one filter, which gives a matrix with
   K = kh×kw and a single row: a poor fit for a big MAC array, which is the point of the M5 analysis.

## M2 — Importer, executor and single-op parity

### What was built
`int8rt/graph.py` (the internal graph), `int8rt/importer.py` (ONNX file to graph), `int8rt/ops.py` (the table of supported ops),
`int8rt/executor.py` (runs the graph), kernels for QuantizeLinear, DequantizeLinear, MaxPool, QLinearGlobalAveragePool and ReduceMean,
and `tools/make_op_tests.py` (300 one-op models checked bit for bit against ORT). `tools/probe_ort.py` records which float32
formulas ORT really uses.

### Key pieces

**1. Importer and graph.** An ONNX file lists nodes, constants ("initialisers") and tensor names. The importer turns that into a
`Graph`: nodes in order, and for every tensor its shape, dtype and (for int8 activations) scale and zero point. It refuses
anything it cannot run (unknown op, unknown attribute, dynamic shape) and lists *all* problems at once.
*Example:* `QuantizeLinear(x, scale=0.5, zp=-3) -> MaxPool` gives both output tensors scale 0.5 and zero point -3, because max-pooling
does not change the quantisation.

**2. Executor.** For each node in order: take the inputs that are not constants, call the kernel, write into the node's
preallocated output buffer. Weights, scales and zero points are looked up once when the executor is built, not on every run.
`dump_dir` saves every tensor so M3 can find the first node where int8rt and ORT disagree.

**3. QuantizeLinear and DequantizeLinear.** Quantise: `q = clamp(round_half_even(x / scale) + zp)`. Dequantise: `x = (q - zp) * scale`.
*Example:* scale 0.5, zp 1: x = 0.75 gives 0.75 / 0.5 = 1.5, which rounds to 2 (ties go to even), plus 1 is 3.

**4. QLinearGlobalAveragePool.** The average of H×W int8 values is itself requantised: `acc = sum(x) - zp*H*W`, multiplier
`x_scale / (y_scale * H*W)`. So it reuses the same requantiser as convolution (and therefore has both float and fixed modes).
*Example:* x = [11, 12, 13, 14], zp 10, all scales 1: acc = 50 - 40 = 10, multiplier 1/4, exact 2.5. Float mode rounds to even: 2.
Fixed mode rounds the tie up: 3. That is a real 1-LSB difference between the two modes, and it is a unit test.

**5. MaxPool padding.** Padded positions must never win the max. Treating them as 0 would be wrong for all-negative
windows (the max would become 0), so they are treated as -128, the smallest int8.

**6. ReduceMean and why float order matters.** Float addition is not associative: in float32, `(1e8 + 1) - 1e8` is 0 but
`(1e8 - 1e8) + 1` is 1. The KWS model has one float op, ReduceMean, between a DequantizeLinear and a QuantizeLinear. ORT adds the
values in a specific vectorised order, so to be bit-exact int8rt adds them in the same order: 4-wide groups, two running totals
(even and odd groups), combine, add the four lanes as `(l0+l2)+(l1+l3)`, then the leftover scalars.
*Example:* [1e8, 1, -1e8, 1, 1, 1, 1, 1] sums to 4 in ORT's order (mean 0.5) but 6 left to right (mean 0.75). It is a unit test.
This was the only mismatch found in M2 (MISMATCHES M-001).

**7. How the parity test works.** Each case is a one-node ONNX model with random shapes, scales and zero points (fixed seed).
ORT runs it, int8rt runs it, and the outputs must be identical bytes. Random data is a weak test of the *multiplier formula*, because
changing `M` by one float ULP almost never changes an output. So extra "sweep" cases force `x == x_zp`, which makes the accumulator
equal to the bias, and pass hundreds of thousands of chosen accumulators straight through the requantiser. With those, a different
order of the multiplication was caught.

### Likely interview questions

1. **How do you know your runtime matches ONNX Runtime, not just on the final accuracy?** Single-op models compared byte for byte
   (300 cases, all ops), then, in M3, every intermediate tensor of both full models. Accuracy alone can hide 1-LSB differences.
2. **What was the one mismatch you found, and why did it happen?** ReduceMean. Float32 addition is not associative, and ORT sums with
   Eigen's SSE2 vectorised reduction, whose order differs from `np.sum`. I found the order by testing candidate orders against ORT with
   large inputs, then reproduced it, including how the alignment of each row's start changes the order.
3. **Why did you restrict ReduceMean to trailing axes?** ORT uses different reduction routines for other axis patterns and I could
   only reproduce some of them. The models only need `axes=[2,3]`, so I reject the rest with a clear error rather than claim a
   parity I have not shown.
4. **Why add the sweep cases instead of more random ones?** A wrong multiplier order differs by one float ULP, which only changes a
   rounded result when `acc * M` is almost exactly on a tie. Random cases passed with the wrong formula. Putting the accumulators in the
   bias lets me choose enough of them to hit ties. Even so the evidence is a few elements out of millions, so I describe it as consistent,
   not proven.
5. **Why does the importer compute shapes itself rather than call ONNX shape inference?** ONNX does not know com.microsoft ops such as
   QGemm and QLinearGlobalAveragePool, so it cannot shape them. Writing the rules myself also means the single-op tests, which declare the
   output shape independently, check those rules.
