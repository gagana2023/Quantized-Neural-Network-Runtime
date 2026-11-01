"""QLinearMatMul and com.microsoft QGemm: INT8 matrix products with INT32 accumulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from int8rt.requant import QParams, RequantMode, check_int32, multiplier, requantize


@dataclass(frozen=True)
class GemmAttrs:
    """QGemm attributes. ``alpha`` scales the product (folded into the requant multiplier)."""

    alpha: float = 1.0
    trans_a: bool = False
    trans_b: bool = False


def qgemm(
    x: npt.NDArray[np.int8],
    w: npt.NDArray[np.int8],
    bias: npt.NDArray[np.int32] | None,
    attrs: GemmAttrs,
    qparams: QParams,
    requant: RequantMode,
    out: npt.NDArray[np.int8],
) -> None:
    """``out = requant(alpha * (A - a_zp) @ (B - b_zp) + bias)`` with A (M,K) and B (K,N).

    ``qparams.w_scale`` / ``w_zp`` are scalars or per-column (length N). With ``trans_a`` /
    ``trans_b`` the stored operand is transposed first, as in ONNX Gemm.
    """
    if x.dtype != np.int8 or w.dtype != np.int8 or out.dtype != np.int8:
        raise TypeError("qgemm expects int8 x, w and out")
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError("qgemm expects 2-D operands")
    a = x.T if attrs.trans_a else x
    b = w.T if attrs.trans_b else w
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"inner dimensions differ: {a.shape} vs {b.shape}")
    if out.shape != (m, n):
        raise ValueError(f"out has shape {out.shape}, expected {(m, n)}")

    # int32 before the zero-point subtraction; per-column B zero point broadcasts over rows of B.
    a32 = a.astype(np.int32) - np.int32(qparams.x_zp)
    b32 = b.astype(np.int32) - np.broadcast_to(qparams.w_zp, (n,)).reshape(1, n)
    acc = np.matmul(a32.astype(np.int64), b32.astype(np.int64))
    if bias is not None:
        acc += np.broadcast_to(bias.astype(np.int64), (n,)).reshape(1, n)

    mult = multiplier(qparams) * np.float32(attrs.alpha)
    out[...] = requantize(check_int32(acc), mult, qparams.y_zp, requant, channel_axis=1)


def qlinear_matmul(
    x: npt.NDArray[np.int8],
    w: npt.NDArray[np.int8],
    bias: npt.NDArray[np.int32] | None,
    attrs: GemmAttrs,
    qparams: QParams,
    requant: RequantMode,
    out: npt.NDArray[np.int8],
) -> None:
    """ONNX QLinearMatMul (2-D): a QGemm with no bias and no transposes."""
    if bias is not None or attrs != GemmAttrs():
        raise ValueError("qlinear_matmul takes no bias and default attrs")
    qgemm(x, w, None, attrs, qparams, requant, out)
