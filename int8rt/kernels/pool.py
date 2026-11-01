"""Pooling and reduction kernels: MaxPool, QLinearGlobalAveragePool, ReduceMean (float)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from int8rt.requant import RequantMode, check_int32, requantize


@dataclass(frozen=True)
class PoolAttrs:
    """MaxPool attributes. ``pads`` is ``(top, left, bottom, right)``; dilation 1, no ceil_mode."""

    kernel: tuple[int, int]
    strides: tuple[int, int] = (1, 1)
    pads: tuple[int, int, int, int] = (0, 0, 0, 0)


def pool_output_hw(h: int, w: int, attrs: PoolAttrs) -> tuple[int, int]:
    kh, kw = attrs.kernel
    top, left, bottom, right = attrs.pads
    sh, sw = attrs.strides
    return (h + top + bottom - kh) // sh + 1, (w + left + right - kw) // sw + 1


def max_pool(x: npt.NDArray[np.int8], attrs: PoolAttrs, out: npt.NDArray[np.int8]) -> None:
    """INT8 max pooling. Padded positions are ignored (treated as -inf), as in ONNX."""
    if x.dtype != np.int8 or out.dtype != np.int8:
        raise TypeError("max_pool expects int8 x and out")
    n, c, h, w = x.shape
    kh, kw = attrs.kernel
    sh, sw = attrs.strides
    top, left, bottom, right = attrs.pads
    oh, ow = pool_output_hw(h, w, attrs)
    if out.shape != (n, c, oh, ow):
        raise ValueError(f"out has shape {out.shape}, expected {(n, c, oh, ow)}")
    padded = np.pad(
        x, ((0, 0), (0, 0), (top, bottom), (left, right)), constant_values=np.iinfo(np.int8).min
    )
    out[...] = np.iinfo(np.int8).min
    for i in range(kh):
        for j in range(kw):
            np.maximum(out, padded[:, :, i : i + sh * oh : sh, j : j + sw * ow : sw], out=out)


def global_average_pool_multiplier(
    x_scale: np.float32, y_scale: np.float32, image_size: int
) -> np.float32:
    """``x_scale / (y_scale * image_size)`` in float32, the ORT / MLAS order of operations."""
    return np.float32(np.float32(x_scale) / (np.float32(y_scale) * np.float32(image_size)))


def qlinear_global_average_pool(
    x: npt.NDArray[np.int8],
    x_scale: np.float32,
    x_zp: int,
    y_scale: np.float32,
    y_zp: int,
    requant: RequantMode,
    out: npt.NDArray[np.int8],
) -> None:
    """Average over H and W: ``acc = sum(x) - x_zp * H * W`` (int32), then requantise by
    ``M = x_scale / (y_scale * H * W)``. ``x`` is NCHW, ``out`` is (N, C, 1, 1)."""
    if x.dtype != np.int8 or out.dtype != np.int8:
        raise TypeError("qlinear_global_average_pool expects int8 x and out")
    n, c, h, w = x.shape
    if out.shape != (n, c, 1, 1):
        raise ValueError(f"out has shape {out.shape}, expected {(n, c, 1, 1)}")
    acc = x.astype(np.int64).sum(axis=(2, 3), keepdims=True) - np.int64(x_zp) * (h * w)
    mult = global_average_pool_multiplier(x_scale, y_scale, h * w)
    out[...] = requantize(check_int32(acc), np.asarray(mult), y_zp, requant)


_PACKET = 4  # float32 lanes in an SSE2 packet, the width ORT's Eigen build vectorises with


def _eigen_sum_rows(v: npt.NDArray[np.float32], start: int) -> npt.NDArray[np.float32]:
    """float32 sum of every row of ``v`` (G, L) in the exact order of Eigen's SSE2 ``redux``.

    ``start`` is how many leading elements precede the first 16-byte-aligned one (the same for all
    rows in ``v``). The order is: 4-lane packets from ``start``, two independent accumulators
    (even / odd packets), their sum, one leftover packet, a horizontal add ``(l0+l2)+(l1+l3)``,
    then the ``start`` leading scalars, then the tail scalars. Too few elements for one packet
    means a plain left-to-right sum.
    """
    g, length = v.shape
    size = length - start
    p = _PACKET
    end1 = max(size, 0) // p * p  # packets cover [start, start + end1)
    end2 = max(size, 0) // (2 * p) * (2 * p)  # the unrolled-by-two part
    if end1 == 0:
        total = v[:, 0].copy()
        for j in range(1, length):
            total = total + v[:, j]
        return total
    acc0 = v[:, start : start + p].copy()
    if end1 > p:
        acc1 = v[:, start + p : start + 2 * p].copy()
        for i in range(start + 2 * p, start + end2, 2 * p):
            acc0 = acc0 + v[:, i : i + p]
            acc1 = acc1 + v[:, i + p : i + 2 * p]
        acc0 = acc0 + acc1
        if end1 > end2:
            acc0 = acc0 + v[:, start + end2 : start + end1]
    total = (acc0[:, 0] + acc0[:, 2]) + (acc0[:, 1] + acc0[:, 3])
    for j in range(start):
        total = total + v[:, j]
    for j in range(start + end1, length):
        total = total + v[:, j]
    return total


def reduce_mean(
    x: npt.NDArray[np.float32], axes: tuple[int, ...], keepdims: bool, out: npt.NDArray[np.float32]
) -> None:
    """float32 mean over a contiguous run of trailing ``axes``, bit-matching ORT on x86.

    ORT sums each contiguous row with Eigen's vectorised reduction, whose association order
    differs from ``np.sum`` (see ``_eigen_sum_rows`` and docs/MISMATCHES.md M-001). Row ``k``
    starts ``k * R`` elements into the buffer, so its alignment (and sum order) depends on ``k``;
    the buffer itself is assumed 16-byte aligned (true for ORT's arena and for NumPy).
    Then ``mean = sum / R`` in float32.
    """
    if x.dtype != np.float32 or out.dtype != np.float32:
        raise TypeError("reduce_mean expects float32 x and out")
    rank = x.ndim
    if axes != tuple(range(rank - len(axes), rank)) or not axes:
        raise NotImplementedError("reduce_mean supports only a contiguous run of trailing axes")
    r = int(np.prod(x.shape[rank - len(axes) :]))
    rows = np.ascontiguousarray(x).reshape(-1, r)
    totals = np.empty(rows.shape[0], dtype=np.float32)
    starts = (-(np.arange(rows.shape[0]) * r)) % _PACKET  # elements before 16-byte alignment
    for start in range(_PACKET):
        sel = starts == start
        if sel.any():
            totals[sel] = _eigen_sum_rows(rows[sel], start)
    out[...] = (totals / np.float32(r)).reshape(out.shape)
