"""Operator registry: for each supported (domain, op_type), its allowed attributes, its shape/dtype
rule, and a ``prepare`` that binds a node to a kernel call.

``prepare`` resolves all constants (weights, scales, zero points, attributes) once. The returned
``Runner`` takes the values of the node's *dynamic* inputs and writes into a preallocated ``out``.
The importer uses the registry to reject anything unsupported; the executor uses it to run nodes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from int8rt.graph import Graph, Node
from int8rt.kernels.conv import ConvAttrs, conv_output_hw, qlinear_conv
from int8rt.kernels.matmul import GemmAttrs, qgemm, qlinear_matmul
from int8rt.kernels.pool import (
    PoolAttrs,
    max_pool,
    pool_output_hw,
    qlinear_global_average_pool,
    reduce_mean,
)
from int8rt.kernels.quant import dequantize_linear, quantize_linear
from int8rt.requant import QParams, RequantMode, make_qparams

Shape = tuple[int, ...]
Runner = Callable[[Sequence[npt.NDArray[Any]], npt.NDArray[Any]], None]
Prepare = Callable[[Node, Graph, RequantMode], Runner]
Infer = Callable[[Node, Sequence[Shape]], tuple[Shape, np.dtype[Any]]]

INT8 = np.dtype(np.int8)
FLOAT32 = np.dtype(np.float32)


class UnsupportedError(ValueError):
    """The model uses an op, attribute or shape that int8rt does not support."""


@dataclass(frozen=True)
class OpDef:
    attrs: frozenset[str]
    infer: Infer
    prepare: Prepare


# ---------------------------------------------------------------- helpers


def _const(graph: Graph, node: Node, idx: int) -> npt.NDArray[Any]:
    if idx >= len(node.inputs) or not node.inputs[idx]:
        raise UnsupportedError(f"{node.name}: input {idx} is required")
    t = graph.tensors[node.inputs[idx]]
    if t.data is None:
        raise UnsupportedError(f"{node.name}: input {idx} ({t.name}) must be a constant")
    return t.data


def _scalar_scale(graph: Graph, node: Node, idx: int) -> np.float32:
    arr = _const(graph, node, idx)
    if arr.size != 1:
        raise UnsupportedError(f"{node.name}: input {idx} must be a per-tensor scale")
    return np.float32(arr.reshape(()))


def _scalar_zp(graph: Graph, node: Node, idx: int) -> int:
    arr = _const(graph, node, idx)
    if arr.size != 1 or arr.dtype != np.int8:
        raise UnsupportedError(f"{node.name}: input {idx} must be a per-tensor int8 zero point")
    return int(arr.reshape(()))


def _optional_bias(graph: Graph, node: Node, idx: int) -> npt.NDArray[np.int32] | None:
    if idx >= len(node.inputs) or not node.inputs[idx]:
        return None
    bias = _const(graph, node, idx)
    if bias.dtype != np.int32:
        raise UnsupportedError(f"{node.name}: bias must be int32, got {bias.dtype}")
    return bias


def _pair(attrs: dict[str, Any], name: str, default: tuple[int, int]) -> tuple[int, int]:
    v = attrs.get(name)
    return default if v is None else (int(v[0]), int(v[1]))


def _per_channel(arr: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """A scalar or length-n vector as a 1-D array (so QParams broadcasting works)."""
    return arr.reshape(-1) if arr.size > 1 else arr.reshape(())


def _require_attr(node: Node, name: str, allowed: tuple[Any, ...]) -> None:
    value = node.attrs.get(name, allowed[0])
    if isinstance(value, bytes):
        value = value.decode()
    if value not in allowed:
        raise UnsupportedError(f"{node.name}: attribute {name}={value!r} is not supported")


# ---------------------------------------------------------------- QuantizeLinear / DequantizeLinear


def _infer_quantize(node: Node, shapes: Sequence[Shape]) -> tuple[Shape, np.dtype[Any]]:
    return shapes[0], INT8


def _prepare_quantize(node: Node, graph: Graph, requant: RequantMode) -> Runner:
    _require_attr(node, "axis", (1,))
    _require_attr(node, "saturate", (1,))
    scale, zp = _scalar_scale(graph, node, 1), _scalar_zp(graph, node, 2)
    return lambda ins, out: quantize_linear(ins[0], scale, zp, out)


def _infer_dequantize(node: Node, shapes: Sequence[Shape]) -> tuple[Shape, np.dtype[Any]]:
    return shapes[0], FLOAT32


def _prepare_dequantize(node: Node, graph: Graph, requant: RequantMode) -> Runner:
    _require_attr(node, "axis", (1,))
    scale, zp = _scalar_scale(graph, node, 1), _scalar_zp(graph, node, 2)
    return lambda ins, out: dequantize_linear(ins[0], scale, zp, out)


# ---------------------------------------------------------------- QLinearConv


def _conv_attrs(node: Node) -> ConvAttrs:
    a = node.attrs
    pads = a.get("pads", [0, 0, 0, 0])
    return ConvAttrs(
        strides=_pair(a, "strides", (1, 1)),
        pads=(int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])),
        group=int(a.get("group", 1)),
        dilations=_pair(a, "dilations", (1, 1)),
    )


def _check_conv(node: Node) -> ConvAttrs:
    _require_attr(node, "auto_pad", ("NOTSET",))
    attrs = _conv_attrs(node)
    if attrs.dilations != (1, 1):
        raise UnsupportedError(f"{node.name}: dilation {attrs.dilations} is not supported")
    return attrs


def _infer_conv(node: Node, shapes: Sequence[Shape]) -> tuple[Shape, np.dtype[Any]]:
    attrs = _check_conv(node)
    (n, _, h, w), (o, _, kh, kw) = shapes[0], shapes[3]
    oh, ow = conv_output_hw(h, w, kh, kw, attrs)
    return (n, o, oh, ow), INT8


def _prepare_conv(node: Node, graph: Graph, requant: RequantMode) -> Runner:
    attrs = _check_conv(node)
    w = _const(graph, node, 3)
    qp = make_qparams(
        _scalar_scale(graph, node, 1),
        _scalar_zp(graph, node, 2),
        _per_channel(_const(graph, node, 4)),
        _per_channel(_const(graph, node, 5)),
        _scalar_scale(graph, node, 6),
        _scalar_zp(graph, node, 7),
    )
    bias = _optional_bias(graph, node, 8)
    return lambda ins, out: qlinear_conv(ins[0], w, bias, attrs, qp, requant, out)


# ---------------------------------------------------------------- QGemm / QLinearMatMul


def _gemm_attrs(node: Node) -> GemmAttrs:
    return GemmAttrs(
        alpha=float(node.attrs.get("alpha", 1.0)),
        trans_a=bool(node.attrs.get("transA", 0)),
        trans_b=bool(node.attrs.get("transB", 0)),
    )


def _infer_gemm(node: Node, shapes: Sequence[Shape]) -> tuple[Shape, np.dtype[Any]]:
    attrs = _gemm_attrs(node)
    if len(node.inputs) < 9 or not node.inputs[7]:
        raise UnsupportedError(f"{node.name}: QGemm without y_scale (float output) is unsupported")
    a, b = shapes[0], shapes[3]
    m = a[1] if attrs.trans_a else a[0]
    n = b[0] if attrs.trans_b else b[1]
    return (m, n), INT8


def _prepare_gemm(node: Node, graph: Graph, requant: RequantMode) -> Runner:
    attrs = _gemm_attrs(node)
    w = _const(graph, node, 3)
    qp = make_qparams(
        _scalar_scale(graph, node, 1),
        _scalar_zp(graph, node, 2),
        _per_channel(_const(graph, node, 4)),
        _per_channel(_const(graph, node, 5)),
        _scalar_scale(graph, node, 7),
        _scalar_zp(graph, node, 8),
    )
    bias = _optional_bias(graph, node, 6)
    return lambda ins, out: qgemm(ins[0], w, bias, attrs, qp, requant, out)


def _infer_matmul(node: Node, shapes: Sequence[Shape]) -> tuple[Shape, np.dtype[Any]]:
    a, b = shapes[0], shapes[3]
    if len(a) != 2 or len(b) != 2:
        raise UnsupportedError(f"{node.name}: only 2-D QLinearMatMul is supported")
    return (a[0], b[1]), INT8


def _prepare_matmul(node: Node, graph: Graph, requant: RequantMode) -> Runner:
    w = _const(graph, node, 3)
    qp: QParams = make_qparams(
        _scalar_scale(graph, node, 1),
        _scalar_zp(graph, node, 2),
        _per_channel(_const(graph, node, 4)),
        _per_channel(_const(graph, node, 5)),
        _scalar_scale(graph, node, 6),
        _scalar_zp(graph, node, 7),
    )
    return lambda ins, out: qlinear_matmul(ins[0], w, None, GemmAttrs(), qp, requant, out)


# ---------------------------------------------------------------- pooling, flatten, reduce


def _pool_attrs(node: Node) -> PoolAttrs:
    _require_attr(node, "auto_pad", ("NOTSET",))
    _require_attr(node, "ceil_mode", (0,))
    _require_attr(node, "storage_order", (0,))
    a = node.attrs
    if _pair(a, "dilations", (1, 1)) != (1, 1):
        raise UnsupportedError(f"{node.name}: pooling dilation is not supported")
    if "kernel_shape" not in a or len(a["kernel_shape"]) != 2:
        raise UnsupportedError(f"{node.name}: only 2-D pooling is supported")
    pads = a.get("pads", [0, 0, 0, 0])
    return PoolAttrs(
        kernel=_pair(a, "kernel_shape", (1, 1)),
        strides=_pair(a, "strides", (1, 1)),
        pads=(int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])),
    )


def _infer_max_pool(node: Node, shapes: Sequence[Shape]) -> tuple[Shape, np.dtype[Any]]:
    n, c, h, w = shapes[0]
    oh, ow = pool_output_hw(h, w, _pool_attrs(node))
    return (n, c, oh, ow), INT8


def _prepare_max_pool(node: Node, graph: Graph, requant: RequantMode) -> Runner:
    attrs = _pool_attrs(node)
    return lambda ins, out: max_pool(ins[0], attrs, out)


def _infer_gap(node: Node, shapes: Sequence[Shape]) -> tuple[Shape, np.dtype[Any]]:
    _require_attr(node, "channels_last", (0,))
    n, c = shapes[0][:2]
    return (n, c, 1, 1), INT8


def _prepare_gap(node: Node, graph: Graph, requant: RequantMode) -> Runner:
    _require_attr(node, "channels_last", (0,))
    xs, xz = _scalar_scale(graph, node, 1), _scalar_zp(graph, node, 2)
    ys, yz = _scalar_scale(graph, node, 3), _scalar_zp(graph, node, 4)
    return lambda ins, out: qlinear_global_average_pool(ins[0], xs, xz, ys, yz, requant, out)


def _infer_flatten(node: Node, shapes: Sequence[Shape]) -> tuple[Shape, np.dtype[Any]]:
    shape = shapes[0]
    axis = int(node.attrs.get("axis", 1))
    if axis < 0:
        axis += len(shape)
    return (int(np.prod(shape[:axis], dtype=np.int64)), int(np.prod(shape[axis:]))), INT8


def _prepare_flatten(node: Node, graph: Graph, requant: RequantMode) -> Runner:
    def run(ins: Sequence[npt.NDArray[Any]], out: npt.NDArray[Any]) -> None:
        out[...] = ins[0].reshape(out.shape)

    return run


def _reduce_axes(node: Node, rank: int) -> tuple[int, ...]:
    _require_attr(node, "noop_with_empty_axes", (0,))
    axes = node.attrs.get("axes")
    if not axes:
        raise UnsupportedError(f"{node.name}: ReduceMean needs explicit axes")
    norm = tuple(sorted(int(a) % rank for a in axes))
    if norm != tuple(range(rank - len(norm), rank)):
        raise UnsupportedError(
            f"{node.name}: ReduceMean over non-trailing axes {norm} is not supported "
            "(ORT's summation order is only reproduced for a contiguous run of trailing axes)"
        )
    return norm


def _infer_reduce_mean(node: Node, shapes: Sequence[Shape]) -> tuple[Shape, np.dtype[Any]]:
    shape = shapes[0]
    axes = _reduce_axes(node, len(shape))
    keep = bool(node.attrs.get("keepdims", 1))
    out = tuple(1 if i in axes else d for i, d in enumerate(shape) if keep or i not in axes)
    return out, FLOAT32


def _prepare_reduce_mean(node: Node, graph: Graph, requant: RequantMode) -> Runner:
    rank = len(graph.tensors[node.inputs[0]].shape)
    axes = _reduce_axes(node, rank)
    keep = bool(node.attrs.get("keepdims", 1))
    return lambda ins, out: reduce_mean(ins[0], axes, keep, out)


# ---------------------------------------------------------------- registry

_CONV_ATTRS = frozenset({"auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"})
_POOL_ATTRS = frozenset(
    {"auto_pad", "ceil_mode", "dilations", "kernel_shape", "pads", "storage_order", "strides"}
)

OPS: dict[tuple[str, str], OpDef] = {
    ("ai.onnx", "QuantizeLinear"): OpDef(
        frozenset({"axis", "saturate"}), _infer_quantize, _prepare_quantize
    ),
    ("ai.onnx", "DequantizeLinear"): OpDef(
        frozenset({"axis"}), _infer_dequantize, _prepare_dequantize
    ),
    ("ai.onnx", "QLinearConv"): OpDef(_CONV_ATTRS, _infer_conv, _prepare_conv),
    ("ai.onnx", "QLinearMatMul"): OpDef(frozenset(), _infer_matmul, _prepare_matmul),
    ("com.microsoft", "QGemm"): OpDef(
        frozenset({"alpha", "transA", "transB"}), _infer_gemm, _prepare_gemm
    ),
    ("com.microsoft", "QLinearGlobalAveragePool"): OpDef(
        frozenset({"channels_last"}), _infer_gap, _prepare_gap
    ),
    ("ai.onnx", "MaxPool"): OpDef(_POOL_ATTRS, _infer_max_pool, _prepare_max_pool),
    ("ai.onnx", "Flatten"): OpDef(frozenset({"axis"}), _infer_flatten, _prepare_flatten),
    ("ai.onnx", "ReduceMean"): OpDef(
        frozenset({"axes", "keepdims", "noop_with_empty_axes"}),
        _infer_reduce_mean,
        _prepare_reduce_mean,
    ),
}


def op_def(node: Node) -> OpDef:
    try:
        return OPS[node.key]
    except KeyError:
        raise UnsupportedError(f"unsupported op {node.domain}.{node.op_type}") from None
