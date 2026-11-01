"""M2 tests: the non-conv kernels (quantise, pool, reduce) against hand-computed values."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from int8rt.kernels.pool import PoolAttrs, max_pool, qlinear_global_average_pool, reduce_mean
from int8rt.kernels.quant import dequantize_linear, quantize_linear
from int8rt.requant import RequantMode

F32 = np.float32


def i8(v: npt.ArrayLike) -> npt.NDArray[np.int8]:
    return np.asarray(v, dtype=np.int8)


def f32(v: npt.ArrayLike) -> npt.NDArray[np.float32]:
    return np.asarray(v, dtype=np.float32)


# ---------------------------------------------------------------- QuantizeLinear / DequantizeLinear


def test_quantize_hand_computed() -> None:
    # scale 0.5, zp 1: x/scale = [0.5, 1.5, -0.5, 5, 2000, -2000]
    x = f32([0.25, 0.75, -0.25, 2.5, 1000.0, -1000.0])
    out = np.zeros(6, dtype=np.int8)
    quantize_linear(x, F32(0.5), 1, out)
    # ties go to even: 0.5 -> 0, 1.5 -> 2, -0.5 -> -0; then + zp; then saturate at 127 / -128
    assert out.tolist() == [1, 3, 1, 6, 127, -128]
    assert out.dtype == np.int8


def test_quantize_saturates_exactly_at_the_ends() -> None:
    out = np.zeros(4, dtype=np.int8)
    quantize_linear(f32([127.0, 128.0, -128.0, -129.0]), F32(1.0), 0, out)
    assert out.tolist() == [127, 127, -128, -128]


def test_dequantize_hand_computed() -> None:
    out = np.zeros(3, dtype=np.float32)
    dequantize_linear(i8([-128, 0, 127]), F32(0.5), 10, out)
    # (x - zp) * scale; the subtraction must not wrap in int8 (-128 - 10 = -138)
    assert out.tolist() == [-69.0, -5.0, 58.5]
    assert out.dtype == np.float32


def test_quantize_dequantize_dtype_checks() -> None:
    with pytest.raises(TypeError):
        quantize_linear(f32([1.0]), F32(1), 0, np.zeros(1, dtype=np.int32))
    with pytest.raises(TypeError):
        dequantize_linear(i8([1]), F32(1), 0, np.zeros(1, dtype=np.float64))


# ---------------------------------------------------------------- MaxPool


def test_max_pool_2x2_stride_2() -> None:
    x = i8(np.arange(16).reshape(1, 1, 4, 4) - 8)
    out = np.zeros((1, 1, 2, 2), dtype=np.int8)
    max_pool(x, PoolAttrs(kernel=(2, 2), strides=(2, 2)), out)
    assert out[0, 0].tolist() == [[-3, -1], [5, 7]]
    assert out.dtype == np.int8


def test_max_pool_padding_is_ignored_not_zero() -> None:
    # all values negative: a zero pad would win the max and give 0
    x = i8([[[[-5, -7], [-9, -128]]]])
    out = np.zeros((1, 1, 2, 2), dtype=np.int8)
    max_pool(x, PoolAttrs(kernel=(3, 3), strides=(1, 1), pads=(1, 1, 1, 1)), out)
    assert out.tolist() == [[[[-5, -5], [-5, -5]]]]


def test_max_pool_odd_size_floors() -> None:
    x = i8(np.arange(25).reshape(1, 1, 5, 5))
    out = np.zeros((1, 1, 2, 2), dtype=np.int8)
    max_pool(x, PoolAttrs(kernel=(2, 2), strides=(2, 2)), out)  # last row / column is dropped
    assert out[0, 0].tolist() == [[6, 8], [16, 18]]


# ---------------------------------------------------------------- QLinearGlobalAveragePool


@pytest.mark.parametrize(("mode", "expected"), [("float", 2), ("fixed", 3)])
def test_gap_tie_differs_between_modes(mode: RequantMode, expected: int) -> None:
    # x = [11, 12, 13, 14], zp 10: acc = 50 - 4*10 = 10, M = 1 / (1 * 4): exact value 2.5.
    # float: half to even -> 2. fixed: both gemmlowp roundings send the tie up -> 3.
    out = np.zeros((1, 1, 1, 1), dtype=np.int8)
    qlinear_global_average_pool(i8([[[[11, 12], [13, 14]]]]), F32(1), 10, F32(1), 0, mode, out)
    assert out.item() == expected
    assert out.dtype == np.int8


def test_gap_saturates() -> None:
    x = i8([[[[127, 127], [127, 127]], [[-128, -128], [-128, -128]]]])
    out = np.zeros((1, 2, 1, 1), dtype=np.int8)
    # real means 12.7 and -12.8; y_scale 0.05, y_zp 5 -> 254 + 5 and -256 + 5: both saturate
    qlinear_global_average_pool(x, F32(0.1), 0, F32(0.05), 5, "float", out)
    assert out.reshape(-1).tolist() == [127, -128]


# ---------------------------------------------------------------- ReduceMean


def test_reduce_mean_uses_eigen_summation_order() -> None:
    # 8 elements: packets p0 = [1e8, 1, -1e8, 1], p1 = [1, 1, 1, 1].
    # Eigen: p0 + p1 = [1e8 (the +1 is lost), 2, -1e8, 2]; horizontal (l0+l2) + (l1+l3) = 0 + 4.
    # mean = 4 / 8 = 0.5. A left-to-right sum loses the first +1 as well and gives 6 / 8 = 0.75.
    x = f32([[1e8, 1, -1e8, 1, 1, 1, 1, 1]])
    out = np.zeros(1, dtype=np.float32)
    reduce_mean(x, (1,), False, out)
    assert out.item() == 0.5


def test_reduce_mean_small_exact_cases() -> None:
    x = f32(np.arange(24).reshape(1, 2, 3, 4))
    out = np.zeros((1, 2), dtype=np.float32)
    reduce_mean(x, (2, 3), False, out)
    assert out.tolist() == [[5.5, 17.5]]
    out_k = np.zeros((1, 2, 1, 1), dtype=np.float32)
    reduce_mean(x, (2, 3), True, out_k)
    assert out_k.reshape(-1).tolist() == [5.5, 17.5]


def test_reduce_mean_rejects_bad_axes_and_dtypes() -> None:
    with pytest.raises(NotImplementedError):
        reduce_mean(f32(np.zeros((2, 3, 4))), (0,), False, np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(TypeError):
        reduce_mean(f32([[1, 2, 3]]).astype(np.float64), (1,), False, np.zeros(1, dtype=np.float32))
