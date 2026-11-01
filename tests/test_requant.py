"""M1 tests: requantisation modes, with hand-computed expectations."""

from __future__ import annotations

import numpy as np
import pytest

from int8rt.requant import (
    check_int32,
    make_qparams,
    multiplier,
    quantize_multiplier,
    requantize,
    rounding_right_shift,
    saturating_rounding_doubling_high_mul,
)


def i32(*v: int) -> np.ndarray:
    return np.array(v, dtype=np.int32)


def f32(v: float) -> np.ndarray:
    return np.array(v, dtype=np.float32)


def test_multiplier_is_float32_and_per_channel() -> None:
    qp = make_qparams(0.5, 0, [1.0, 0.5], [0, 0], 0.25, 0)
    m = multiplier(qp)
    assert m.dtype == np.float32
    np.testing.assert_array_equal(m, np.array([2.0, 1.0], dtype=np.float32))


def test_float_mode_hand_computed() -> None:
    # M = 0.25: 10 -> 2.5 -> 2 (half to even), -10 -> -2.5 -> -2, 25 -> 6.25 -> 6,
    # +-1000 -> +-250 saturates to 127 / -128.
    acc = i32(10, -10, 25, 1000, -1000)
    y = requantize(acc, f32(0.25), 0, "float")
    assert y.dtype == np.int8
    np.testing.assert_array_equal(y, [2, -2, 6, 127, -128])
    # with y_zp = 5 the shift is applied after rounding, and saturation is on the final value
    y = requantize(acc, f32(0.25), 5, "float")
    np.testing.assert_array_equal(y, [7, 3, 11, 127, -128])


def test_float_mode_ties_go_to_even() -> None:
    # M = 0.5: 1.5 -> 2, 2.5 -> 2, 3.5 -> 4, -1.5 -> -2, -2.5 -> -2
    y = requantize(i32(3, 5, 7, -3, -5), f32(0.5), 0, "float")
    np.testing.assert_array_equal(y, [2, 2, 4, -2, -2])


def test_quantize_multiplier_hand_computed() -> None:
    m0, exp = quantize_multiplier(np.array([0.25, 0.75, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(m0, [1 << 30, 1610612736, 1 << 30])
    np.testing.assert_array_equal(exp, [-1, 0, 2])
    assert m0.dtype == np.int32 and exp.dtype == np.int32


def test_quantize_multiplier_reconstructs_m() -> None:
    rng = np.random.default_rng(1)
    m = rng.uniform(1e-4, 4.0, size=1000).astype(np.float32)
    m0, exp = quantize_multiplier(m)
    assert np.all(m0 >= 1 << 30) and np.all(m0 < 1 << 31)
    rebuilt = m0.astype(np.float64) / 2**31 * 2.0 ** exp.astype(np.float64)
    np.testing.assert_allclose(rebuilt, m.astype(np.float64), rtol=2**-30)


def test_quantize_multiplier_zero_and_invalid() -> None:
    m0, exp = quantize_multiplier(np.array([0.0], dtype=np.float32))
    assert m0[0] == 0 and exp[0] == 0
    with pytest.raises(ValueError):
        quantize_multiplier(np.array([-1.0], dtype=np.float32))


def test_high_mul_and_rounding_shift_hand_computed() -> None:
    m0 = np.array([1 << 30] * 4, dtype=np.int64)  # multiplies by 0.5
    a = np.array([10, -10, 25, -25], dtype=np.int64)
    # 10*0.5=5, -5; ties go toward +inf here: 12.5 -> 13 but -12.5 -> -12 (nudge + truncation)
    np.testing.assert_array_equal(saturating_rounding_doubling_high_mul(a, m0), [5, -5, 13, -12])
    x = np.array([5, -5, 6, -6, 7, -7], dtype=np.int64)
    one = np.ones(6, dtype=np.int64)
    # x / 2 with ties away from zero: 2.5 -> 3, -2.5 -> -3, 3, -3, 3.5 -> 4, -3.5 -> -4
    np.testing.assert_array_equal(rounding_right_shift(x, one), [3, -3, 3, -3, 4, -4])


def test_fixed_mode_hand_computed_ties_differ_from_float() -> None:
    # M = 0.25 -> m0 = 2**30, exp = -1. The rounding shift sends ties AWAY from zero (3 and -3),
    # float rounds them to even (2 and -2). 25 shows gemmlowp's double rounding:
    # 25 * 0.5 = 12.5 -> 13, then 13 / 2 = 6.5 -> 7, while the exact value 6.25 rounds to 6.
    acc = i32(10, -10, 25, 1000, -1000)
    y = requantize(acc, f32(0.25), 0, "fixed")
    assert y.dtype == np.int8
    np.testing.assert_array_equal(y, [3, -3, 7, 127, -128])


def test_fixed_mode_exact_cases_match_float() -> None:
    # No ties and exactly representable M: both modes agree.
    acc = i32(0, 4, -4, 8, -8, 100, -100)
    for m in (0.5, 0.25, 1.0, 2.0):
        a = requantize(acc, f32(m), 3, "float")
        b = requantize(acc, f32(m), 3, "fixed")
        np.testing.assert_array_equal(a, b)
    # M >= 1 uses the left-shift path: 3 * 2 = 6
    np.testing.assert_array_equal(requantize(i32(3, -3), f32(2.0), 0, "fixed"), [6, -6])


def test_per_channel_multiplier_broadcasts_along_channel_axis() -> None:
    acc = np.full((1, 2, 1, 2), 8, dtype=np.int32)
    mult = np.array([0.5, 0.25], dtype=np.float32)
    for mode in ("float", "fixed"):
        y = requantize(acc, mult, 0, mode, channel_axis=1)
        np.testing.assert_array_equal(y[0, :, 0, 0], [4, 2])


def test_fixed_deviation_from_float_is_small() -> None:
    rng = np.random.default_rng(0)
    acc = rng.integers(-200_000, 200_000, size=50_000).astype(np.int32)
    m = f32(0.0013)
    a = requantize(acc, m, -10, "float").astype(np.int32)
    b = requantize(acc, m, -10, "fixed").astype(np.int32)
    assert np.abs(a - b).max() <= 1


def test_check_int32() -> None:
    ok = np.array([2**31 - 1, -(2**31)], dtype=np.int64)
    assert check_int32(ok).dtype == np.int32
    with pytest.raises(OverflowError):
        check_int32(np.array([2**31], dtype=np.int64))
