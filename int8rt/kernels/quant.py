"""QuantizeLinear and DequantizeLinear (per-tensor scale and zero point)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from int8rt.requant import INT8_MAX, INT8_MIN


def quantize_linear(
    x: npt.NDArray[np.float32], scale: np.float32, zp: int, out: npt.NDArray[np.int8]
) -> None:
    """``out = sat_int8(rint(x / scale) + zp)``; float32 division, half to even."""
    if x.dtype != np.float32 or out.dtype != np.int8:
        raise TypeError("quantize_linear expects float32 x and int8 out")
    scaled = np.rint(x / np.float32(scale))
    clipped = np.clip(scaled, INT8_MIN - zp, INT8_MAX - zp)
    out[...] = (clipped.astype(np.int32) + zp).astype(np.int8)


def dequantize_linear(
    x: npt.NDArray[np.int8], scale: np.float32, zp: int, out: npt.NDArray[np.float32]
) -> None:
    """``out = float32(x - zp) * scale``; the subtraction is done in int32."""
    if x.dtype != np.int8 or out.dtype != np.float32:
        raise TypeError("dequantize_linear expects int8 x and float32 out")
    out[...] = (x.astype(np.int32) - np.int32(zp)).astype(np.float32) * np.float32(scale)
