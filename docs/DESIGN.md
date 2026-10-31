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
