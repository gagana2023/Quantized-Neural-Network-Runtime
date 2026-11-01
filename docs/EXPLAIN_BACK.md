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
