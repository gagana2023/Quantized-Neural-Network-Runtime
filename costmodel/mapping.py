"""Map an imported ``Graph`` onto accelerator work: GEMMs for the MAC array, streams for the vector
unit.

Convolution -> per-group GEMM  W[M,K] x X[K,N] -> Y[M,N]
  M = output channels per group, K = (input channels per group) * kh * kw, N = output H * W.
A depthwise convolution is ``groups = C`` GEMMs with M = 1. FC / QGemm / QLinearMatMul are GEMMs
whose weight matrix is the constant operand (M = output features) and N = the batch rows (1 here).
Everything else (quantise, pool, reduce) runs on the vector unit; Flatten is a layout change
and costs nothing.

This module reads shapes and attributes only; no numerics are involved.
"""

from __future__ import annotations

from dataclasses import dataclass

from int8rt.graph import Graph, Node


@dataclass(frozen=True)
class GemmWork:
    """``groups`` independent GEMMs of shape (m x k) by (k x n).

    ``in_unique`` is the number of distinct input elements one group's GEMM reads from DRAM. A 3x3
    convolution reads each input pixel once although the on-the-fly im2col matrix (K*N elements)
    repeats it ~9 times, so a tile of the K x N input matrix costs ``tile * in_unique / (K*N)``
    elements of DRAM traffic. It is capped at K*N (a strided 1x1 convolution never reads more).
    """

    name: str
    op: str
    groups: int
    m: int
    k: int
    n: int
    in_unique: int
    has_bias: bool
    in_itemsize: int
    out_itemsize: int

    @property
    def macs(self) -> int:
        return self.groups * self.m * self.k * self.n


@dataclass(frozen=True)
class VectorWork:
    """A pass over ``in_bytes`` of input producing ``out_bytes`` with ``ops`` element operations."""

    name: str
    op: str
    in_bytes: int
    out_bytes: int
    ops: int


Work = GemmWork | VectorWork


def _elems(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _conv(node: Node, graph: Graph) -> GemmWork:
    x = graph.tensors[node.inputs[0]]
    w = graph.tensors[node.inputs[3]]
    y = graph.tensors[node.outputs[0]]
    groups = int(node.attrs.get("group", 1))
    cout, cin_g, kh, kw = w.shape
    _, _, oh, ow = y.shape
    m, k, n = cout // groups, cin_g * kh * kw, oh * ow
    unique_in = cin_g * x.shape[2] * x.shape[3]
    return GemmWork(
        node.name,
        node.op_type,
        groups,
        m,
        k,
        n,
        min(unique_in, k * n),
        has_bias=len(node.inputs) > 8 and bool(node.inputs[8]),
        in_itemsize=x.dtype.itemsize,
        out_itemsize=y.dtype.itemsize,
    )


def _gemm(node: Node, graph: Graph) -> GemmWork:
    a = graph.tensors[node.inputs[0]]
    y = graph.tensors[node.outputs[0]]
    if node.op_type == "QGemm":
        w = graph.tensors[node.inputs[3]]
        trans_b = int(node.attrs.get("transB", 0))
        m, k = (w.shape[0], w.shape[1]) if trans_b else (w.shape[1], w.shape[0])
        bias = len(node.inputs) > 6 and bool(node.inputs[6])
    else:  # QLinearMatMul: A (rows, k) x constant B (k, cols); the weight matrix is B
        w = graph.tensors[node.inputs[3]]
        k, m = w.shape
        bias = False
    n = _elems(y.shape) // m
    return GemmWork(
        node.name, node.op_type, 1, m, k, n, k * n, bias, a.dtype.itemsize, y.dtype.itemsize
    )


def _vector(node: Node, graph: Graph) -> VectorWork:
    x = graph.tensors[node.inputs[0]]
    y = graph.tensors[node.outputs[0]]
    in_el, out_el = _elems(x.shape), _elems(y.shape)
    in_b, out_b = in_el * x.dtype.itemsize, out_el * y.dtype.itemsize
    if node.op_type == "Flatten":
        return VectorWork(node.name, node.op_type, 0, 0, 0)  # layout only
    if node.op_type == "MaxPool":
        kh, kw = node.attrs["kernel_shape"]
        ops = out_el * kh * kw  # one compare per window element
    elif node.op_type in ("QLinearGlobalAveragePool", "ReduceMean"):
        ops = in_el + out_el  # one add per input, one scale + requantise per output
    else:  # QuantizeLinear / DequantizeLinear: one op per element
        ops = out_el
    return VectorWork(node.name, node.op_type, in_b, out_b, ops)


def map_graph(graph: Graph) -> list[Work]:
    """One ``Work`` per node, in execution order."""
    work: list[Work] = []
    for node in graph.nodes:
        if node.op_type == "QLinearConv":
            work.append(_conv(node, graph))
        elif node.op_type in ("QGemm", "QLinearMatMul"):
            work.append(_gemm(node, graph))
        else:
            work.append(_vector(node, graph))
    return work
