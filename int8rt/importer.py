"""Load a quantised ONNX file into the internal ``Graph``.

Rejects unsupported ops, unsupported attributes and dynamic shapes with one error that lists
every problem found. Shapes are computed by int8rt's own per-op rules (ONNX shape inference does
not know the com.microsoft ops) and cross-checked against the file's ``value_info`` when present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

from int8rt.graph import Graph, Node, TensorInfo
from int8rt.ops import OPS, UnsupportedError, op_def

# Output quantisation parameters live in these node input positions: (scale, zero point).
_OUT_QPARAM_INPUTS: dict[str, tuple[int, int]] = {
    "QuantizeLinear": (1, 2),
    "QLinearConv": (6, 7),
    "QLinearMatMul": (6, 7),
    "QGemm": (7, 8),
    "QLinearGlobalAveragePool": (3, 4),
}
_INHERIT_QPARAMS = {"MaxPool", "Flatten"}  # output has the same scale / zero point as input 0


def _attr_value(attr: onnx.AttributeProto) -> Any:
    value = helper.get_attribute_value(attr)
    return value.decode() if isinstance(value, bytes) else value


def _static_shape(vi: onnx.ValueInfoProto) -> tuple[int, ...] | None:
    """The shape of a value_info, or None if any dimension is dynamic / unknown."""
    if not vi.type.HasField("tensor_type") or not vi.type.tensor_type.HasField("shape"):
        return None
    dims = vi.type.tensor_type.shape.dim
    if any(not d.HasField("dim_value") or d.dim_value <= 0 for d in dims):
        return None
    return tuple(d.dim_value for d in dims)


def _np_dtype(vi: onnx.ValueInfoProto) -> np.dtype[Any]:
    return np.dtype(helper.tensor_dtype_to_np_dtype(vi.type.tensor_type.elem_type))


def _scalar(info: TensorInfo) -> Any:
    assert info.data is not None
    return info.data.reshape(()) if info.data.size == 1 else None


def load_model(path: str | Path) -> Graph:
    """Import ``path`` (a static-shape INT8 ONNX model) into a ``Graph``."""
    model = onnx.load(str(path))
    return graph_from_model(model)


def graph_from_model(model: onnx.ModelProto) -> Graph:
    g = model.graph
    errors: list[str] = []

    tensors: dict[str, TensorInfo] = {}
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        tensors[init.name] = TensorInfo(init.name, tuple(arr.shape), arr.dtype, data=arr)

    inputs: list[str] = []
    for vi in g.input:
        if vi.name in tensors:  # initialiser listed as a graph input (old IR versions)
            continue
        shape = _static_shape(vi)
        if shape is None:
            errors.append(f"graph input {vi.name!r} has a dynamic or unknown shape")
            continue
        tensors[vi.name] = TensorInfo(vi.name, shape, _np_dtype(vi))
        inputs.append(vi.name)

    nodes: list[Node] = []
    for i, n in enumerate(g.node):
        nodes.append(
            Node(
                name=n.name or f"{n.op_type}_{i}",
                op_type=n.op_type,
                domain=n.domain or "ai.onnx",
                inputs=list(n.input),
                outputs=list(n.output),
                attrs={a.name: _attr_value(a) for a in n.attribute},
            )
        )

    unsupported = sorted({f"{n.domain}.{n.op_type}" for n in nodes if n.key not in OPS})
    if unsupported:
        errors.append("unsupported ops: " + ", ".join(unsupported))
    for n in nodes:
        if n.key in OPS:
            extra = sorted(set(n.attrs) - OPS[n.key].attrs)
            if extra:
                errors.append(f"{n.name} ({n.op_type}): unsupported attributes {extra}")
    if errors:
        raise UnsupportedError("cannot import model:\n  " + "\n  ".join(errors))

    declared = {vi.name: vi for vi in [*g.value_info, *g.output]}
    for n in nodes:
        missing = [t for t in n.inputs if t and t not in tensors]
        if missing:
            raise UnsupportedError(f"{n.name}: inputs {missing} are not produced earlier")
        shapes = [tensors[t].shape if t else () for t in n.inputs]
        try:
            shape, dtype = op_def(n).infer(n, shapes)
        except (IndexError, ValueError) as e:
            raise UnsupportedError(f"{n.name} ({n.op_type}): {e}") from e
        for out in n.outputs:
            vi = declared.get(out)
            if vi is not None and (decl := _static_shape(vi)) is not None and decl != shape:
                raise UnsupportedError(
                    f"{n.name}: computed shape {shape} of {out!r} disagrees with the file's {decl}"
                )
            tensors[out] = TensorInfo(out, shape, dtype)
        _set_output_qparams(n, tensors)

    outputs = [vi.name for vi in g.output]
    missing_out = [o for o in outputs if o not in tensors]
    if missing_out:
        raise UnsupportedError(f"graph outputs {missing_out} are not produced by any node")
    return Graph(nodes=nodes, tensors=tensors, inputs=inputs, outputs=outputs)


def _set_output_qparams(n: Node, tensors: dict[str, TensorInfo]) -> None:
    out = tensors[n.outputs[0]]
    if n.op_type in _OUT_QPARAM_INPUTS:
        si, zi = _OUT_QPARAM_INPUTS[n.op_type]
        scale, zp = tensors[n.inputs[si]], tensors[n.inputs[zi]]
        out.scale = np.float32(_scalar(scale))
        out.zero_point = int(_scalar(zp))
    elif n.op_type in _INHERIT_QPARAMS:
        src = tensors[n.inputs[0]]
        out.scale, out.zero_point = src.scale, src.zero_point
