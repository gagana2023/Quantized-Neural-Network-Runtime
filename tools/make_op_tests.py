"""Single-op parity tests: build one-node ONNX models, run them in ORT and int8rt, compare exactly.

Cases are random (fixed seed) over shapes, scales, zero points and data distributions, for every
op int8rt supports. A case passes only if the output bytes are identical to ORT's.

Usage: python -m tools.make_op_tests [--seed 0] [--out results/op_parity.json] [--keep DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
from onnx import TensorProto as TP
from onnx import helper, numpy_helper

from int8rt.executor import Executor
from int8rt.importer import graph_from_model
from int8rt.requant import RequantMode
from tools.envinfo import collect

ROOT = Path(__file__).resolve().parent.parent
IR_VERSION = 10  # the newest IR version the pinned onnxruntime accepts
Rng = np.random.Generator


@dataclass
class Case:
    case_id: str
    op: str
    params: dict[str, Any]
    model: onnx.ModelProto
    feeds: dict[str, npt.NDArray[Any]]
    inits: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- random helpers


def _scale(rng: Rng, lo: float = 0.003, hi: float = 0.3) -> np.float32:
    return np.float32(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _zp(rng: Rng) -> np.int8:
    return np.int8(0 if rng.random() < 0.3 else rng.integers(-128, 128))


def _int8_data(rng: Rng, shape: tuple[int, ...], zp: int, dist: str) -> npt.NDArray[np.int8]:
    if dist == "full":
        a = rng.integers(-128, 128, shape)
    elif dist == "narrow":
        a = np.rint(rng.normal(zp, 20, shape))
    else:  # "extreme": saturating corners only
        a = rng.choice(np.array([-128, 127, zp]), shape)
    return np.clip(a, -128, 127).astype(np.int8)


def _dist(rng: Rng) -> str:
    return str(rng.choice(["full", "narrow", "extreme"], p=[0.4, 0.4, 0.2]))


def _output_scale(rng: Rng, real_std: float) -> np.float32:
    """A y_scale putting the output std near 40 LSB x f, so some cases saturate and some don't."""
    f = rng.uniform(0.3, 3.0)
    return np.float32(max(real_std, 1e-6) / (40.0 * f))


def _build(
    case_id: str,
    op: str,
    node: onnx.NodeProto,
    dyn: dict[str, npt.NDArray[Any]],
    out_name: str,
    out_dtype: int,
    out_shape: tuple[int, ...],
    inits: dict[str, npt.NDArray[Any]],
    params: dict[str, Any],
) -> Case:
    graph = helper.make_graph(
        [node],
        case_id,
        [helper.make_tensor_value_info(k, _elem(v.dtype), list(v.shape)) for k, v in dyn.items()],
        [helper.make_tensor_value_info(out_name, out_dtype, list(out_shape))],
        [numpy_helper.from_array(v, k) for k, v in inits.items()],
    )
    opsets = [helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    model = helper.make_model(graph, opset_imports=opsets)
    model.ir_version = IR_VERSION
    return Case(case_id, op, params, model, dyn, inits)


def _elem(dtype: np.dtype[Any]) -> int:
    return int(helper.np_dtype_to_tensor_dtype(dtype))


# ---------------------------------------------------------------- case builders


def conv_case(rng: Rng, i: int) -> Case:
    kind = str(rng.choice(["normal", "depthwise", "grouped", "pointwise", "big_k"]))
    n = int(rng.integers(1, 3))
    kh, kw = (1, 1) if kind == "pointwise" else (int(rng.integers(1, 6)), int(rng.integers(1, 6)))
    if kind == "depthwise":
        g = c = int(rng.integers(1, 9))
        o = c * int(rng.integers(1, 3))
    elif kind == "grouped":
        g = int(rng.choice([2, 3, 4]))
        c, o = g * int(rng.integers(1, 4)), g * int(rng.integers(1, 4))
    elif kind == "big_k":  # K > 1000, so |acc| can exceed 2**24 (float32 precision limit)
        g, c, o = 1, int(rng.integers(64, 129)), int(rng.integers(1, 5))
        kh = kw = 5
    else:
        g, c, o = 1, int(rng.integers(1, 9)), int(rng.integers(1, 9))
    sh, sw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    pads = [int(rng.integers(0, kh)), int(rng.integers(0, kw))]
    pads += [int(rng.integers(0, kh)), int(rng.integers(0, kw))]  # top, left, bottom, right
    h = int(rng.integers(max(1, kh - pads[0] - pads[2]), 14))
    w = int(rng.integers(max(1, kw - pads[1] - pads[3]), 14))
    h, w = max(h, 1), max(w, 1)
    while (h + pads[0] + pads[2] - kh) < 0:
        h += 1
    while (w + pads[1] + pads[3] - kw) < 0:
        w += 1
    oh, ow = (h + pads[0] + pads[2] - kh) // sh + 1, (w + pads[1] + pads[3] - kw) // sw + 1

    xs, ys_zp, x_zp = _scale(rng, 0.01, 0.2), _zp(rng), _zp(rng)
    ws = np.array([_scale(rng, 0.002, 0.05) for _ in range(o)], dtype=np.float32)
    per_tensor_w = rng.random() < 0.2
    if per_tensor_w:
        ws = ws[:1]
    w_zp_nonzero = rng.random() < 0.3
    w_zp = (
        rng.integers(-20, 21, ws.shape).astype(np.int8)
        if w_zp_nonzero
        else np.zeros_like(ws, np.int8)
    )
    xd, wd = (
        ("extreme", "extreme")
        if kind == "big_k" and rng.random() < 0.5
        else (_dist(rng), _dist(rng))
    )
    x = _int8_data(rng, (n, c, h, w), int(x_zp), xd)
    wt = _int8_data(rng, (o, c // g, kh, kw), 0, wd)
    k = (c // g) * kh * kw
    real_std = float(xs) * float(ws.mean()) * np.sqrt(k) * 40.0 * 40.0
    ys = _output_scale(rng, real_std)
    with_bias = rng.random() < 0.6
    inits: dict[str, npt.NDArray[Any]] = {
        "xs": np.asarray(xs), "xz": np.asarray(x_zp), "w": wt,
        "ws": ws if not per_tensor_w else np.asarray(ws[0]),
        "wz": w_zp if not per_tensor_w else np.asarray(w_zp[0]),
        "ys": np.asarray(ys), "yz": np.asarray(ys_zp),
    }  # fmt: skip
    names = ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz"]
    if with_bias:
        scale = 1 if rng.random() < 0.7 else 1000
        inits["b"] = rng.integers(-500 * scale, 500 * scale + 1, (o,)).astype(np.int32)
        names.append("b")
    node = helper.make_node(
        "QLinearConv", names, ["y"], kernel_shape=[kh, kw], strides=[sh, sw], pads=pads, group=g
    )
    params = dict(kind=kind, x=(n, c, h, w), w=(o, c // g, kh, kw), strides=[sh, sw], pads=pads,
                  group=g, bias=with_bias, w_zp_nonzero=w_zp_nonzero, x_zp=int(x_zp), dist=[xd, wd],
                  per_tensor_w=per_tensor_w)  # fmt: skip
    return _build(f"QLinearConv_{i:03d}", "QLinearConv", node, {"x": x}, "y", TP.INT8,
                  (n, o, oh, ow), inits, params)  # fmt: skip


def gemm_case(rng: Rng, i: int) -> Case:
    m, k, n = int(rng.integers(1, 9)), int(rng.integers(1, 301)), int(rng.integers(1, 33))
    trans_a, trans_b = bool(rng.random() < 0.3), bool(rng.random() < 0.5)
    alpha = 1.0 if rng.random() < 0.6 else float(rng.choice([0.5, 2.0, 0.37, 1.25]))
    xs, x_zp, y_zp = _scale(rng, 0.01, 0.2), _zp(rng), _zp(rng)
    ws = np.array([_scale(rng, 0.002, 0.05) for _ in range(n)], dtype=np.float32)
    per_tensor_w = rng.random() < 0.4
    if per_tensor_w:
        ws = ws[:1]
    w_zp_nonzero = rng.random() < 0.3
    w_zp = (
        rng.integers(-20, 21, ws.shape).astype(np.int8)
        if w_zp_nonzero
        else np.zeros_like(ws, np.int8)
    )
    a = _int8_data(rng, (k, m) if trans_a else (m, k), int(x_zp), _dist(rng))
    b = _int8_data(rng, (n, k) if trans_b else (k, n), 0, _dist(rng))
    real_std = float(xs) * float(ws.mean()) * alpha * np.sqrt(k) * 40.0 * 40.0
    ys = _output_scale(rng, real_std)
    inits: dict[str, npt.NDArray[Any]] = {
        "xs": np.asarray(xs), "xz": np.asarray(x_zp), "b": b,
        "ws": np.asarray(ws[0]) if per_tensor_w else ws,
        "wz": np.asarray(w_zp[0]) if per_tensor_w else w_zp,
        "ys": np.asarray(ys), "yz": np.asarray(y_zp),
    }  # fmt: skip
    with_bias = rng.random() < 0.7
    names = ["a", "xs", "xz", "b", "ws", "wz", "", "ys", "yz"]
    if with_bias:
        inits["c"] = rng.integers(-2000, 2001, (n,)).astype(np.int32)
        names[6] = "c"
    node = helper.make_node(
        "QGemm", names, ["y"], domain="com.microsoft", alpha=alpha,
        transA=int(trans_a), transB=int(trans_b),
    )  # fmt: skip
    params = dict(m=m, k=k, n=n, trans_a=trans_a, trans_b=trans_b, alpha=alpha, bias=with_bias,
                  w_zp_nonzero=w_zp_nonzero, per_tensor_w=per_tensor_w)  # fmt: skip
    return _build(f"QGemm_{i:03d}", "QGemm", node, {"a": a}, "y", TP.INT8, (m, n), inits, params)


def matmul_case(rng: Rng, i: int) -> Case:
    m, k, n = int(rng.integers(1, 9)), int(rng.integers(1, 201)), int(rng.integers(1, 17))
    xs, x_zp, y_zp = _scale(rng, 0.01, 0.2), _zp(rng), _zp(rng)
    ws = np.array([_scale(rng, 0.002, 0.05) for _ in range(n)], dtype=np.float32)
    per_tensor_w = rng.random() < 0.5
    if per_tensor_w:
        ws = ws[:1]
    w_zp = (
        rng.integers(-20, 21, ws.shape).astype(np.int8)
        if rng.random() < 0.3
        else np.zeros_like(ws, np.int8)
    )
    a = _int8_data(rng, (m, k), int(x_zp), _dist(rng))
    b = _int8_data(rng, (k, n), 0, _dist(rng))
    ys = _output_scale(rng, float(xs) * float(ws.mean()) * np.sqrt(k) * 40.0 * 40.0)
    inits: dict[str, npt.NDArray[Any]] = {
        "xs": np.asarray(xs), "xz": np.asarray(x_zp), "b": b,
        "ws": np.asarray(ws[0]) if per_tensor_w else ws,
        "wz": np.asarray(w_zp[0]) if per_tensor_w else w_zp,
        "ys": np.asarray(ys), "yz": np.asarray(y_zp),
    }  # fmt: skip
    node = helper.make_node("QLinearMatMul", ["a", "xs", "xz", "b", "ws", "wz", "ys", "yz"], ["y"])
    params = dict(m=m, k=k, n=n, per_tensor_w=per_tensor_w, w_zp_nonzero=bool(w_zp.any()))
    return _build(f"QLinearMatMul_{i:03d}", "QLinearMatMul", node, {"a": a}, "y", TP.INT8,
                  (m, n), inits, params)  # fmt: skip


def gap_case(rng: Rng, i: int) -> Case:
    n, c = int(rng.integers(1, 3)), int(rng.integers(1, 9))
    h, w = int(rng.integers(1, 17)), int(rng.integers(1, 17))
    xs, ys, x_zp, y_zp = _scale(rng, 0.01, 0.3), _scale(rng, 0.01, 0.3), _zp(rng), _zp(rng)
    dist = _dist(rng)
    x = _int8_data(rng, (n, c, h, w), int(x_zp), dist)
    inits = {
        "xs": np.asarray(xs),
        "xz": np.asarray(x_zp),
        "ys": np.asarray(ys),
        "yz": np.asarray(y_zp),
    }
    node = helper.make_node(
        "QLinearGlobalAveragePool", ["x", "xs", "xz", "ys", "yz"], ["y"],
        domain="com.microsoft", channels_last=0,
    )  # fmt: skip
    params = dict(x=(n, c, h, w), dist=dist, x_zp=int(x_zp), y_zp=int(y_zp))
    return _build(f"QLinearGlobalAveragePool_{i:03d}", "QLinearGlobalAveragePool", node, {"x": x},
                  "y", TP.INT8, (n, c, 1, 1), inits, params)  # fmt: skip


def maxpool_case(rng: Rng, i: int) -> Case:
    n, c = int(rng.integers(1, 3)), int(rng.integers(1, 6))
    kh, kw = int(rng.integers(1, 5)), int(rng.integers(1, 5))
    sh, sw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    pads = [
        int(rng.integers(0, kh)),
        int(rng.integers(0, kw)),
        int(rng.integers(0, kh)),
        int(rng.integers(0, kw)),
    ]
    h = int(rng.integers(max(1, kh), 14))
    w = int(rng.integers(max(1, kw), 14))
    oh, ow = (h + pads[0] + pads[2] - kh) // sh + 1, (w + pads[1] + pads[3] - kw) // sw + 1
    x = _int8_data(rng, (n, c, h, w), 0, str(rng.choice(["full", "narrow"])))
    node = helper.make_node(
        "MaxPool", ["x"], ["y"], kernel_shape=[kh, kw], strides=[sh, sw], pads=pads
    )
    params = dict(x=(n, c, h, w), kernel=[kh, kw], strides=[sh, sw], pads=pads)
    return _build(
        f"MaxPool_{i:03d}", "MaxPool", node, {"x": x}, "y", TP.INT8, (n, c, oh, ow), {}, params
    )


def quantize_case(rng: Rng, i: int) -> Case:
    shape = tuple(int(d) for d in rng.integers(1, 9, int(rng.integers(1, 5))))
    scale, zp = _scale(rng, 0.001, 1.0), _zp(rng)
    mode = str(rng.choice(["normal", "wide", "ties"]))
    if (
        mode == "ties"
    ):  # exact halves: with a power-of-two scale, x / scale lands exactly on k + 0.5
        scale = np.float32(2.0 ** int(rng.integers(-6, 2)))
        x = ((rng.integers(-200, 200, shape) + 0.5) * scale).astype(np.float32)
    elif mode == "wide":
        x = rng.normal(0, 200 * float(scale), shape).astype(np.float32)
    else:
        x = rng.normal(0, 40 * float(scale), shape).astype(np.float32)
    inits = {"ys": np.asarray(scale), "yz": np.asarray(zp)}
    node = helper.make_node("QuantizeLinear", ["x", "ys", "yz"], ["y"])
    params = dict(shape=shape, mode=mode, scale=float(scale), zp=int(zp))
    return _build(
        f"QuantizeLinear_{i:03d}",
        "QuantizeLinear",
        node,
        {"x": x},
        "y",
        TP.INT8,
        shape,
        inits,
        params,
    )


def dequantize_case(rng: Rng, i: int) -> Case:
    shape = tuple(int(d) for d in rng.integers(1, 9, int(rng.integers(1, 5))))
    scale, zp = _scale(rng, 0.001, 1.0), _zp(rng)
    x = _int8_data(rng, shape, int(zp), str(rng.choice(["full", "extreme"])))
    inits = {"xs": np.asarray(scale), "xz": np.asarray(zp)}
    node = helper.make_node("DequantizeLinear", ["x", "xs", "xz"], ["y"])
    params = dict(shape=shape, scale=float(scale), zp=int(zp))
    return _build(
        f"DequantizeLinear_{i:03d}",
        "DequantizeLinear",
        node,
        {"x": x},
        "y",
        TP.FLOAT,
        shape,
        inits,
        params,
    )


def flatten_case(rng: Rng, i: int) -> Case:
    shape = tuple(int(d) for d in rng.integers(1, 7, int(rng.integers(2, 5))))
    axis = int(rng.integers(-len(shape), len(shape) + 1))
    a = axis + len(shape) if axis < 0 else axis
    x = _int8_data(rng, shape, 0, "full")
    out = (int(np.prod(shape[:a])), int(np.prod(shape[a:])))
    node = helper.make_node("Flatten", ["x"], ["y"], axis=axis)
    return _build(f"Flatten_{i:03d}", "Flatten", node, {"x": x}, "y", TP.INT8, out, {},
                  dict(shape=shape, axis=axis))  # fmt: skip


def reduce_mean_case(rng: Rng, i: int) -> Case:
    rank = int(rng.integers(2, 5))
    shape = tuple(int(d) for d in rng.integers(1, 14, rank))
    k = int(rng.integers(1, rank))  # reduce the last k axes (the only pattern int8rt supports)
    axes = list(range(rank - k, rank))
    keep = int(rng.integers(0, 2))
    x = (rng.random(shape) * float(rng.choice([1.0, 6.0, 100.0]))).astype(np.float32)
    out = tuple(1 if d in axes else s for d, s in enumerate(shape) if keep or d not in axes)
    node = helper.make_node("ReduceMean", ["x"], ["y"], axes=axes, keepdims=keep)
    return _build(f"ReduceMean_{i:03d}", "ReduceMean", node, {"x": x}, "y", TP.FLOAT, out, {},
                  dict(shape=shape, axes=axes, keepdims=keep))  # fmt: skip


def _sweep_acc(rng: Rng, mult: npt.NDArray[np.float32]) -> npt.NDArray[np.int32]:
    """One INT32 accumulator per entry of ``mult``, spread so that acc * M mostly lands in range."""
    limit = np.minimum(150.0 / np.maximum(mult.astype(np.float64), 1e-12), 2.0**31 - 1)
    return np.rint(rng.uniform(-1.0, 1.0, mult.shape) * limit).astype(np.int32)


def gemm_sweep_case(rng: Rng, i: int) -> Case:
    """Requantisation probe: x == x_zp makes acc == bias, so the bias vector feeds arbitrary
    INT32 accumulators (including |acc| > 2**24) through ORT's requantiser. This pins down the
    order in which ORT computes M = x_scale * w_scale / y_scale in float32."""
    n = 300000
    xs, ws, ys = _scale(rng, 1e-4, 0.3), _scale(rng, 1e-4, 0.05), _scale(rng, 0.01, 1.0)
    x_zp, y_zp = _zp(rng), _zp(rng)
    mult = np.full(n, (xs * ws) / ys, dtype=np.float32)
    acc = _sweep_acc(rng, mult)
    inits: dict[str, npt.NDArray[Any]] = {
        "xs": np.asarray(xs), "xz": np.asarray(x_zp), "b": np.ones((1, n), np.int8),
        "ws": np.asarray(ws), "wz": np.asarray(np.int8(0)), "c": acc,
        "ys": np.asarray(ys), "yz": np.asarray(y_zp),
    }  # fmt: skip
    node = helper.make_node(
        "QGemm", ["a", "xs", "xz", "b", "ws", "wz", "c", "ys", "yz"], ["y"], domain="com.microsoft"
    )
    a = np.full((1, 1), x_zp, np.int8)
    params = dict(n=n, M=float(mult[0]), max_abs_acc=int(np.abs(acc).max()), y_zp=int(y_zp))
    return _build(
        f"QGemmSweep_{i:03d}", "QGemmSweep", node, {"a": a}, "y", TP.INT8, (1, n), inits, params
    )


def conv_sweep_case(rng: Rng, i: int) -> Case:
    """The same probe through QLinearConv with 100000 output channels, so every channel has its own
    float32 multiplier: a 1x1 conv on a 1x1 image equal to x_zp, so acc == bias per channel."""
    o = 100000
    xs, ys = _scale(rng, 1e-4, 0.3), _scale(rng, 0.01, 1.0)
    x_zp, y_zp = _zp(rng), _zp(rng)
    ws = np.array(np.exp(rng.uniform(np.log(1e-4), np.log(0.05), o)), dtype=np.float32)
    mult = ((xs * ws) / ys).astype(np.float32)
    inits: dict[str, npt.NDArray[Any]] = {
        "xs": np.asarray(xs), "xz": np.asarray(x_zp), "w": np.ones((o, 1, 1, 1), np.int8),
        "ws": ws, "wz": np.zeros(o, np.int8), "ys": np.asarray(ys), "yz": np.asarray(y_zp),
        "b": _sweep_acc(rng, mult),
    }  # fmt: skip
    node = helper.make_node(
        "QLinearConv",
        ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz", "b"],
        ["y"],
        kernel_shape=[1, 1],
    )
    x = np.full((1, 1, 1, 1), x_zp, np.int8)
    params = dict(o=o, y_zp=int(y_zp))
    return _build(
        f"QLinearConvSweep_{i:03d}",
        "QLinearConvSweep",
        node,
        {"x": x},
        "y",
        TP.INT8,
        (1, o, 1, 1),
        inits,
        params,
    )


def quantize_sweep_case(rng: Rng, i: int) -> Case:
    """200000 random floats spanning the int8 range (and beyond) for one scale: probes whether ORT
    divides by the scale or multiplies by its reciprocal, and the rounding of near-ties."""
    n = 200000
    scale, zp = _scale(rng, 1e-3, 1.0), _zp(rng)
    x = (rng.uniform(-140, 140, n) * float(scale)).astype(np.float32)
    inits = {"ys": np.asarray(scale), "yz": np.asarray(zp)}
    node = helper.make_node("QuantizeLinear", ["x", "ys", "yz"], ["y"])
    params = dict(n=n, scale=float(scale), zp=int(zp))
    return _build(
        f"QuantizeSweep_{i:03d}", "QuantizeSweep", node, {"x": x}, "y", TP.INT8, (n,), inits, params
    )


BUILDERS: list[tuple[Callable[[Rng, int], Case], int]] = [
    (conv_case, 80),
    (gemm_case, 30),
    (matmul_case, 20),
    (gap_case, 30),
    (maxpool_case, 25),
    (quantize_case, 25),
    (dequantize_case, 15),
    (flatten_case, 10),
    (reduce_mean_case, 25),
    (gemm_sweep_case, 20),
    (conv_sweep_case, 10),
    (quantize_sweep_case, 10),
]


def generate_cases(seed: int = 0) -> list[Case]:
    """All cases, deterministic for a given ``seed``. One generator per op family."""
    cases: list[Case] = []
    for fam, (builder, count) in enumerate(BUILDERS):
        rng = np.random.default_rng([seed, fam])
        cases += [builder(rng, i) for i in range(count)]
    return cases


# ---------------------------------------------------------------- running and comparing


def run_ort(case: Case) -> npt.NDArray[Any]:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        case.model.SerializeToString(), so, providers=["CPUExecutionProvider"]
    )
    (out,) = sess.run(None, case.feeds)
    return np.asarray(out)


def run_int8rt(case: Case, requant: RequantMode = "float") -> npt.NDArray[Any]:
    graph = graph_from_model(case.model)
    (out,) = Executor(graph, requant=requant).run(case.feeds).values()
    return out


def compare(ref: npt.NDArray[Any], got: npt.NDArray[Any]) -> dict[str, Any]:
    """Exact comparison: same dtype, shape and bytes."""
    if ref.dtype != got.dtype or ref.shape != got.shape:
        return {
            "match": False,
            "reason": f"ORT {ref.dtype}{ref.shape} vs int8rt {got.dtype}{got.shape}",
        }
    if ref.tobytes() == got.tobytes():
        return {"match": True}
    if ref.dtype.kind == "f":
        a, b = ref.view(np.uint32).astype(np.int64), got.view(np.uint32).astype(np.int64)
        bad = a != b
        diff = float(np.max(np.abs(ref[bad].astype(np.float64) - got[bad].astype(np.float64))))
    else:
        bad = ref != got
        diff = float(np.max(np.abs(ref.astype(np.int64) - got.astype(np.int64))))
    idx = tuple(int(v) for v in np.argwhere(bad)[0])
    return {
        "match": False,
        "elements": int(ref.size),
        "differing": int(bad.sum()),
        "max_abs_diff": diff,
        "first_index": idx,
        "ort": ref[idx].item(),
        "int8rt": got[idx].item(),
    }


def run_all(seed: int, keep: Path | None = None) -> dict[str, Any]:
    cases = generate_cases(seed)
    per_op: dict[str, dict[str, int]] = defaultdict(lambda: {"cases": 0, "matched": 0})
    failures: list[dict[str, Any]] = []
    for case in cases:
        if keep is not None:
            keep.mkdir(parents=True, exist_ok=True)
            onnx.save(case.model, str(keep / f"{case.case_id}.onnx"))
        res = compare(run_ort(case), run_int8rt(case))
        per_op[case.op]["cases"] += 1
        per_op[case.op]["matched"] += int(res["match"])
        if not res["match"]:
            failures.append({"case": case.case_id, "params": case.params, **res})
    total = len(cases)
    matched = sum(v["matched"] for v in per_op.values())
    return {"seed": seed, "requant": "float", "cases": total, "matched": matched,
            "per_op": dict(per_op), "failures": failures}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "op_parity.json")
    ap.add_argument("--keep", type=Path, help="also save every generated .onnx here")
    args = ap.parse_args()
    summary = run_all(args.seed, args.keep)
    record = {
        "env": collect(command="python -m tools.make_op_tests " + " ".join(sys.argv[1:])),
        **summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{summary['matched']}/{summary['cases']} cases bit-exact (float mode)")
    for op, v in sorted(summary["per_op"].items()):
        print(f"  {op:<28}{v['matched']:>4}/{v['cases']}")
    for f in summary["failures"][:10]:
        print("FAIL", f["case"], {k: v for k, v in f.items() if k not in ("case", "params")})


if __name__ == "__main__":
    main()
