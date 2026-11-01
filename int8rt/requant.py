"""Requantisation: turn an INT32 accumulator into an INT8 output.

Two modes (see docs/DESIGN.md):

* ``"float"`` (Mode A, default): ``y = sat(rint(acc * M) + y_zp)``,
  with ``M`` and the multiply in float32.
* ``"fixed"`` (Mode B): ``M`` is precomputed as an int32 multiplier plus a shift and applied with
  integer-only maths, gemmlowp style: a saturating rounding doubling high multiply followed by a
  rounding right shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

RequantMode = Literal["float", "fixed"]

INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class QParams:
    """Quantisation parameters of one quantised linear op.

    ``w_scale`` / ``w_zp`` are scalars (per-tensor) or 1-D arrays with one entry per output channel.
    """

    x_scale: np.float32
    x_zp: int
    w_scale: npt.NDArray[np.float32]
    w_zp: npt.NDArray[np.int32]
    y_scale: np.float32
    y_zp: int


def make_qparams(
    x_scale: float | np.floating[Any],
    x_zp: int,
    w_scale: float | npt.ArrayLike,
    w_zp: int | npt.ArrayLike,
    y_scale: float | np.floating[Any],
    y_zp: int,
) -> QParams:
    """Build a ``QParams`` with the dtypes the kernels expect."""
    return QParams(
        x_scale=np.float32(x_scale),
        x_zp=int(x_zp),
        w_scale=np.asarray(w_scale, dtype=np.float32),
        w_zp=np.asarray(w_zp, dtype=np.int32),
        y_scale=np.float32(y_scale),
        y_zp=int(y_zp),
    )


def multiplier(qp: QParams) -> npt.NDArray[np.float32]:
    """``M = (x_scale * w_scale) / y_scale`` in float32, shape () or (channels,)."""
    return ((qp.x_scale * qp.w_scale) / qp.y_scale).astype(np.float32)


def check_int32(acc: npt.NDArray[np.int64]) -> npt.NDArray[np.int32]:
    """Assert an int64 accumulator fits in int32, then narrow it."""
    if acc.size == 0:
        return acc.astype(np.int32)
    lo, hi = int(np.min(acc)), int(np.max(acc))
    if lo < INT32_MIN or hi > INT32_MAX:
        raise OverflowError(f"INT32 accumulator overflow: range [{lo}, {hi}]")
    return acc.astype(np.int32)


# ---------------------------------------------------------------- Mode A: float


def requantize_float(
    acc: npt.NDArray[np.int32], mult: npt.NDArray[np.float32], y_zp: int
) -> npt.NDArray[np.int8]:
    """``sat_int8(rint(float32(acc) * M) + y_zp)``; ``mult`` must already broadcast against ``acc``.

    Rounding is half to even (``np.rint``). The clamp is done in the float domain on the
    zero-point-shifted range so that huge products cannot overflow the integer cast.
    """
    scaled = np.rint(acc.astype(np.float32) * mult.astype(np.float32))
    clipped = np.clip(scaled, INT8_MIN - y_zp, INT8_MAX - y_zp)
    return (clipped.astype(np.int32) + y_zp).astype(np.int8)


# ---------------------------------------------------------------- Mode B: fixed point


def quantize_multiplier(
    m: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Split ``M`` into ``(m0, exp)``: ``M ~= m0 * 2**-31 * 2**exp``, ``m0`` in [2**30, 2**31).

    ``exp`` is the power-of-two exponent: negative means a right shift by ``-exp`` after the high
    multiply, positive means a left shift by ``exp`` before it. ``M == 0`` gives ``m0 == 0``.
    """
    m64 = np.asarray(m, dtype=np.float64)
    if np.any(m64 < 0) or not np.all(np.isfinite(m64)):
        raise ValueError("requantisation multiplier must be finite and non-negative")
    sig, exp = np.frexp(m64)  # m = sig * 2**exp, sig in [0.5, 1)
    m0 = np.rint(sig * (1 << 31)).astype(np.int64)
    exp = exp.astype(np.int64)
    overflow = m0 == (1 << 31)  # sig rounded up to 1.0
    m0 = np.where(overflow, m0 // 2, m0)
    exp = np.where(overflow, exp + 1, exp)
    exp = np.where(m64 == 0, 0, exp)
    if np.any(exp > 30) or np.any(exp < -62):
        raise ValueError("requantisation multiplier outside the supported range")
    return m0.astype(np.int32), exp.astype(np.int32)


def saturating_rounding_doubling_high_mul(
    a: npt.NDArray[np.int64], m0: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """gemmlowp ``SaturatingRoundingDoublingHighMul``: ``round(a * m0 / 2**31)``.

    Exact ties round toward +inf (nudge ``2**30`` if the product is >= 0, else ``1 - 2**30``,
    then division truncating toward zero).

    ``a`` is an int32-range value held in int64, ``m0`` is in [0, 2**31). The product fits in int64.
    """
    ab = a * m0
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    num = ab + nudge
    # C-style division truncates toward zero; ``>>`` floors, so handle negatives explicitly.
    return np.where(num >= 0, num >> 31, -((-num) >> 31))


def rounding_right_shift(
    x: npt.NDArray[np.int64], e: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """gemmlowp ``RoundingDivideByPOT``: ``x / 2**e`` rounded to nearest, ties away from zero."""
    mask = (np.int64(1) << e) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (x < 0)
    return (x >> e) + (remainder > threshold)


def requantize_fixed(
    acc: npt.NDArray[np.int32],
    m0: npt.NDArray[np.int32],
    exp: npt.NDArray[np.int32],
    y_zp: int,
) -> npt.NDArray[np.int8]:
    """Integer-only requantisation. ``m0`` / ``exp`` must already broadcast against ``acc``."""
    a = acc.astype(np.int64)
    exp64 = exp.astype(np.int64)
    left = np.maximum(exp64, 0)
    right = np.maximum(-exp64, 0)
    a = np.clip(a << left, INT32_MIN, INT32_MAX)  # saturating left shift (only when M >= 1)
    hi = saturating_rounding_doubling_high_mul(a, m0.astype(np.int64))
    out = rounding_right_shift(hi, right) + y_zp
    return np.clip(out, INT8_MIN, INT8_MAX).astype(np.int8)


# ---------------------------------------------------------------- dispatch


def requantize(
    acc: npt.NDArray[np.int32],
    mult: npt.NDArray[np.float32],
    y_zp: int,
    mode: RequantMode,
    channel_axis: int = 1,
) -> npt.NDArray[np.int8]:
    """Requantise ``acc`` with a scalar or per-channel multiplier ``mult``.

    A per-channel ``mult`` (1-D) is broadcast along ``channel_axis`` of ``acc``.
    """
    if mult.ndim == 1:
        shape = [1] * acc.ndim
        shape[channel_axis] = mult.shape[0]
        mult = mult.reshape(shape)
    if mode == "float":
        return requantize_float(acc, mult, y_zp)
    if mode == "fixed":
        m0, exp = quantize_multiplier(mult)
        return requantize_fixed(acc, m0, exp, y_zp)
    raise ValueError(f"unknown requantisation mode {mode!r}")
