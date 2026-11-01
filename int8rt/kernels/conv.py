"""QLinearConv: INT8 convolution via im2col + integer matmul, INT32 accumulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from int8rt.requant import QParams, RequantMode, check_int32, multiplier, requantize


@dataclass(frozen=True)
class ConvAttrs:
    """ONNX Conv attributes. ``pads`` is ``(top, left, bottom, right)``; only dilation 1 works."""

    strides: tuple[int, int] = (1, 1)
    pads: tuple[int, int, int, int] = (0, 0, 0, 0)
    group: int = 1
    dilations: tuple[int, int] = (1, 1)


def conv_output_hw(h: int, w: int, kh: int, kw: int, attrs: ConvAttrs) -> tuple[int, int]:
    top, left, bottom, right = attrs.pads
    sh, sw = attrs.strides
    return (h + top + bottom - kh) // sh + 1, (w + left + right - kw) // sw + 1


def im2col(xs: npt.NDArray[np.int32], kh: int, kw: int, attrs: ConvAttrs) -> npt.NDArray[np.int32]:
    """Unfold zero-point-corrected activations ``(N, C, H, W)`` into ``(N, C*kh*kw, OH*OW)``.

    Padding is zeros *after* the zero-point subtraction, i.e. padded pixels hold the value ``x_zp``
    in the original INT8 domain, so they contribute nothing to the accumulator.
    """
    n, c, h, w = xs.shape
    top, left, bottom, right = attrs.pads
    sh, sw = attrs.strides
    oh, ow = conv_output_hw(h, w, kh, kw, attrs)
    padded = np.pad(xs, ((0, 0), (0, 0), (top, bottom), (left, right)))
    cols = np.empty((n, c, kh, kw, oh, ow), dtype=np.int32)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = padded[:, :, i : i + sh * oh : sh, j : j + sw * ow : sw]
    return cols.reshape(n, c * kh * kw, oh * ow)


def qlinear_conv(
    x: npt.NDArray[np.int8],
    w: npt.NDArray[np.int8],
    bias: npt.NDArray[np.int32] | None,
    attrs: ConvAttrs,
    qparams: QParams,
    requant: RequantMode,
    out: npt.NDArray[np.int8],
) -> None:
    """``out = requant(sum((x - x_zp) * (w - w_zp)) + bias)``.

    ``x`` is NCHW, ``w`` is OIHW with ``I = C / group``, ``bias`` is INT32 of length O (or None).
    Depthwise convolution is ``group == C == O``. The result is written into ``out`` (NCHW int8).
    """
    if attrs.dilations != (1, 1):
        raise NotImplementedError("dilation != 1 is not supported")
    if x.dtype != np.int8 or w.dtype != np.int8 or out.dtype != np.int8:
        raise TypeError("qlinear_conv expects int8 x, w and out")
    n, c, h, wd = x.shape
    o, ig, kh, kw = w.shape
    g = attrs.group
    if c != ig * g or o % g != 0:
        raise ValueError(f"inconsistent conv shapes: x {x.shape}, w {w.shape}, group {g}")
    oh, ow = conv_output_hw(h, wd, kh, kw, attrs)
    if out.shape != (n, o, oh, ow):
        raise ValueError(f"out has shape {out.shape}, expected {(n, o, oh, ow)}")
    og = o // g

    # Cast to int32 BEFORE subtracting zero points (int8 arithmetic would wrap).
    xs = x.astype(np.int32) - np.int32(qparams.x_zp)
    w_zp = np.broadcast_to(qparams.w_zp, (o,)).reshape(o, 1, 1, 1)
    ws = w.astype(np.int32) - w_zp

    acc = np.empty((n, o, oh * ow), dtype=np.int64)
    for gi in range(g):
        cols = im2col(xs[:, gi * ig : (gi + 1) * ig], kh, kw, attrs)  # (N, K, OH*OW)
        wmat = ws[gi * og : (gi + 1) * og].reshape(og, ig * kh * kw)  # (Og, K)
        # int64 matmul so the sum cannot wrap silently; narrowed to int32 (with a check) below.
        acc[:, gi * og : (gi + 1) * og] = np.matmul(wmat.astype(np.int64), cols.astype(np.int64))
    if bias is not None:
        acc += bias.astype(np.int64).reshape(1, o, 1)

    acc32 = check_int32(acc).reshape(n, o, oh, ow)
    out[...] = requantize(acc32, multiplier(qparams), qparams.y_zp, requant, channel_axis=1)
