"""M1 tests: QLinearConv / QGemm / QLinearMatMul against hand-computed and naive-loop references."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from int8rt.kernels.conv import ConvAttrs, conv_output_hw, im2col, qlinear_conv
from int8rt.kernels.matmul import GemmAttrs, qgemm, qlinear_matmul
from int8rt.requant import QParams, RequantMode, make_qparams, multiplier, requantize

MODES: list[RequantMode] = ["float", "fixed"]


def i8(v: npt.ArrayLike) -> npt.NDArray[np.int8]:
    return np.asarray(v, dtype=np.int8)


def conv(
    x: npt.NDArray[np.int8],
    w: npt.NDArray[np.int8],
    bias: npt.NDArray[np.int32] | None,
    attrs: ConvAttrs,
    qp: QParams,
    mode: RequantMode = "float",
) -> npt.NDArray[np.int8]:
    oh, ow = conv_output_hw(x.shape[2], x.shape[3], w.shape[2], w.shape[3], attrs)
    out = np.zeros((x.shape[0], w.shape[0], oh, ow), dtype=np.int8)
    qlinear_conv(x, w, bias, attrs, qp, mode, out)
    return out


def naive_conv_acc(
    x: npt.NDArray[np.int8],
    w: npt.NDArray[np.int8],
    bias: npt.NDArray[np.int32] | None,
    attrs: ConvAttrs,
    x_zp: int,
    w_zp: npt.NDArray[np.int32],
) -> npt.NDArray[np.int64]:
    """Direct 7-loop convolution on Python ints; independent of im2col."""
    n, _, h, wd = x.shape
    o, ig, kh, kw = w.shape
    og = o // attrs.group
    top, left, _, _ = attrs.pads
    sh, sw = attrs.strides
    oh, ow = conv_output_hw(h, wd, kh, kw, attrs)
    acc = np.zeros((n, o, oh, ow), dtype=np.int64)
    wz = np.broadcast_to(w_zp, (o,))
    for b in range(n):
        for oc in range(o):
            g = oc // og
            for i in range(oh):
                for j in range(ow):
                    s = int(bias[oc]) if bias is not None else 0
                    for ic in range(ig):
                        for ki in range(kh):
                            for kj in range(kw):
                                yy, xx = i * sh - top + ki, j * sw - left + kj
                                if 0 <= yy < h and 0 <= xx < wd:
                                    xv = int(x[b, g * ig + ic, yy, xx]) - x_zp
                                    s += xv * (int(w[oc, ic, ki, kj]) - int(wz[oc]))
                    acc[b, oc, i, j] = s
    return acc


# ------------------------------------------------------------------ hand-computed conv


def test_conv_hand_computed_zero_point_bias_requant() -> None:
    # xs = x - 1 = [[0,1,2],[3,4,5],[6,7,8]], w = [[1,2],[3,4]]
    # acc = [[27,37],[57,67]], + bias 3 = [[30,40],[60,70]]
    # M = 0.5 * 0.1 / 0.25 = 0.2 -> [[6,8],[12,14]]; y_zp = -10 -> [[-4,-2],[2,4]]
    x = i8([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
    w = i8([[[[1, 2], [3, 4]]]])
    qp = make_qparams(0.5, 1, 0.1, 0, 0.25, -10)
    for mode in MODES:
        y = conv(x, w, np.array([3], dtype=np.int32), ConvAttrs(), qp, mode)
        assert y.dtype == np.int8
        np.testing.assert_array_equal(y[0, 0], [[-4, -2], [2, 4]])


def test_conv_padding_pads_with_zero_point() -> None:
    # 3x3 all-ones kernel, pad 1, x_zp = 1 so padded pixels must count as (x - zp) = 0.
    # Sum of in-bounds xs = [[0,1,2],[3,4,5],[6,7,8]] under each window:
    x = i8([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
    w = np.ones((1, 1, 3, 3), dtype=np.int8)
    qp = make_qparams(1.0, 1, 1.0, 0, 1.0, 0)
    attrs = ConvAttrs(pads=(1, 1, 1, 1))
    expected = [[8, 15, 12], [21, 36, 27], [20, 33, 24]]
    for mode in MODES:
        np.testing.assert_array_equal(conv(x, w, None, attrs, qp, mode)[0, 0], expected)


def test_conv_stride2_odd_size_with_padding() -> None:
    # 5x5 input equal to its zero point + 1 everywhere, 3x3 ones kernel, stride 2, pad 1.
    # Windows cover 2, 3, 2 in-bounds rows (and columns): counts are the outer product.
    x = np.full((1, 1, 5, 5), 2, dtype=np.int8)
    w = np.ones((1, 1, 3, 3), dtype=np.int8)
    qp = make_qparams(1.0, 1, 1.0, 0, 1.0, 0)
    y = conv(x, w, None, ConvAttrs(strides=(2, 2), pads=(1, 1, 1, 1)), qp)
    np.testing.assert_array_equal(y[0, 0], [[4, 6, 4], [6, 9, 6], [4, 6, 4]])


def test_conv_depthwise_no_channel_mixing_and_per_channel_scale() -> None:
    # group = C = O = 2. Ch 0: x=[[1,2],[3,4]], w=ones -> 10. Ch 1: x=2s, w=[[1,2],[3,4]] -> 20.
    # Per-channel w_scale [1, 0.5] -> M = [1, 0.5] -> outputs [10, 10]. Mixing would give 30.
    x = i8([[[[1, 2], [3, 4]], [[2, 2], [2, 2]]]])
    w = i8([[[[1, 1], [1, 1]]], [[[1, 2], [3, 4]]]])
    qp = make_qparams(1.0, 0, [1.0, 0.5], [0, 0], 1.0, 0)
    for mode in MODES:
        y = conv(x, w, None, ConvAttrs(group=2), qp, mode)
        assert y.shape == (1, 2, 1, 1)
        np.testing.assert_array_equal(y.reshape(2), [10, 10])


def test_conv_saturation_and_negative_values() -> None:
    # 1x1 conv, M = 1: 127*127 saturates to 127; -128*127 saturates to -128; -3*7 = -21 passes.
    x = i8([[[[127]], [[-128]], [[-3]]]])  # C = 3, 1x1
    w = i8([[[[127]], [[0]], [[0]]], [[[0]], [[127]], [[0]]], [[[0]], [[0]], [[7]]]])
    qp = make_qparams(1.0, 0, 1.0, 0, 1.0, 0)
    for mode in MODES:
        y = conv(x, w, None, ConvAttrs(), qp, mode)
        np.testing.assert_array_equal(y.reshape(3), [127, -128, -21])


def test_conv_nonzero_weight_zero_point_per_channel() -> None:
    # xs = 5 - 0 = 5; ws = [3 - 1, 3 - (-2)] = [2, 5] -> acc = [10, 25]
    x = i8([[[[5]]]])
    w = i8([[[[3]]], [[[3]]]])
    qp = make_qparams(1.0, 0, 1.0, [1, -2], 1.0, 0)
    np.testing.assert_array_equal(conv(x, w, None, ConvAttrs(), qp).reshape(2), [10, 25])


def test_conv_int32_overflow_is_detected() -> None:
    # 40000 * 255 * 255 = 2.6e9 > 2**31 - 1
    c = 40000
    x = np.full((1, c, 1, 1), 127, dtype=np.int8)
    w = np.full((1, c, 1, 1), 127, dtype=np.int8)
    qp = make_qparams(1.0, -128, 1.0, -128, 1.0, 0)
    with pytest.raises(OverflowError):
        conv(x, w, None, ConvAttrs(), qp)


def test_im2col_layout_hand_computed() -> None:
    xs = np.arange(9, dtype=np.int32).reshape(1, 1, 3, 3)
    cols = im2col(xs, 2, 2, ConvAttrs())
    assert cols.shape == (1, 4, 4) and cols.dtype == np.int32
    # row k = (ki, kj) in row-major kernel order, column = output pixel in row-major order
    np.testing.assert_array_equal(cols[0], [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]])


def test_conv_rejects_dilation_and_bad_dtype() -> None:
    x = np.zeros((1, 1, 3, 3), dtype=np.int8)
    w = np.zeros((1, 1, 1, 1), dtype=np.int8)
    qp = make_qparams(1.0, 0, 1.0, 0, 1.0, 0)
    with pytest.raises(NotImplementedError):
        conv(x, w, None, ConvAttrs(dilations=(2, 2)), qp)
    out = np.zeros((1, 1, 3, 3), dtype=np.int32)
    with pytest.raises(TypeError):
        qlinear_conv(x, w, None, ConvAttrs(), qp, "float", out)  # type: ignore[arg-type]


# ------------------------------------------------------------------ random vs naive loops


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "c,o,group,h,w,k,stride,pad",
    [
        (3, 4, 1, 7, 5, 3, 1, 1),
        (4, 6, 2, 9, 9, 3, 2, 1),
        (5, 5, 5, 8, 7, 3, 1, 1),  # depthwise, odd sizes
        (3, 2, 1, 6, 6, 1, 1, 0),
        (2, 3, 1, 8, 11, 5, 2, 2),
        (6, 6, 6, 5, 5, 3, 2, 0),  # depthwise stride 2, no padding
    ],
)
def test_conv_random_matches_naive_loops(
    mode: RequantMode, c: int, o: int, group: int, h: int, w: int, k: int, stride: int, pad: int
) -> None:
    rng = np.random.default_rng(c * 100 + o * 10 + k)
    x = rng.integers(-128, 128, size=(2, c, h, w), dtype=np.int8)
    wt = rng.integers(-128, 128, size=(o, c // group, k, k), dtype=np.int8)
    bias = rng.integers(-5000, 5000, size=o, dtype=np.int32)
    w_scale = rng.uniform(0.001, 0.01, size=o).astype(np.float32)
    w_zp = rng.integers(-3, 4, size=o, dtype=np.int32)
    qp = make_qparams(0.03, -7, w_scale, w_zp, 0.5, 11)
    attrs = ConvAttrs(strides=(stride, stride), pads=(pad,) * 4, group=group)

    acc = naive_conv_acc(x, wt, bias, attrs, -7, w_zp)
    expected = requantize(acc.astype(np.int32), multiplier(qp), 11, mode, channel_axis=1)
    np.testing.assert_array_equal(conv(x, wt, bias, attrs, qp, mode), expected)


# ------------------------------------------------------------------ matmul / gemm


def test_qlinear_matmul_hand_computed_ties() -> None:
    # as = a - 1 = [[0,1],[2,3]], b = [[5,6],[7,8]] -> acc = [[7,8],[31,36]]
    # M = 0.5 -> [3.5, 4, 15.5, 18]; float rounds ties to even -> [4, 4, 16, 18]
    a = i8([[1, 2], [3, 4]])
    b = i8([[5, 6], [7, 8]])
    qp = make_qparams(0.5, 1, 1.0, 0, 1.0, 0)
    out = np.zeros((2, 2), dtype=np.int8)
    qlinear_matmul(a, b, None, GemmAttrs(), qp, "float", out)
    np.testing.assert_array_equal(out, [[4, 4], [16, 18]])
    # fixed point rounds these ties away from zero: 3.5 -> 4 and 15.5 -> 16, the same here
    qlinear_matmul(a, b, None, GemmAttrs(), qp, "fixed", out)
    np.testing.assert_array_equal(out, [[4, 4], [16, 18]])


def test_qgemm_transb_bias_per_column_scale() -> None:
    # A = [[1,-2,3]], B stored (N=2, K=3) with transB: rows [1,1,1] and [2,0,-1]
    # A @ B^T = [2, -1]; + bias [10, -10] = [12, -11]; M = [1, 2] -> [12, -22]
    a = i8([[1, -2, 3]])
    b = i8([[1, 1, 1], [2, 0, -1]])
    qp = make_qparams(1.0, 0, [1.0, 2.0], 0, 1.0, 0)
    for mode in MODES:
        out = np.zeros((1, 2), dtype=np.int8)
        qgemm(a, b, np.array([10, -10], dtype=np.int32), GemmAttrs(trans_b=True), qp, mode, out)
        np.testing.assert_array_equal(out, [[12, -22]])


def test_qgemm_alpha_and_saturation() -> None:
    a = i8([[100, 100]])
    b = i8([[100], [100]])
    qp = make_qparams(1.0, 0, 1.0, 0, 1.0, 0)
    out = np.zeros((1, 1), dtype=np.int8)
    qgemm(a, b, None, GemmAttrs(), qp, "float", out)
    assert out[0, 0] == 127  # 20000 saturates
    qgemm(a, -b, None, GemmAttrs(alpha=0.0001), qp, "float", out)
    assert out[0, 0] == -2  # -20000 * 1e-4 = -2


@pytest.mark.parametrize("mode", MODES)
def test_qgemm_random_matches_python_ints(mode: RequantMode) -> None:
    rng = np.random.default_rng(5)
    m, k, n = 3, 17, 5
    a = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
    b = rng.integers(-128, 128, size=(n, k), dtype=np.int8)  # transB
    bias = rng.integers(-1000, 1000, size=n, dtype=np.int32)
    w_zp = rng.integers(-2, 3, size=n, dtype=np.int32)
    qp = make_qparams(0.02, 3, rng.uniform(0.002, 0.01, size=n), w_zp, 0.4, -5)
    acc = np.zeros((m, n), dtype=np.int32)
    for i in range(m):
        for j in range(n):
            s = sum((int(a[i, t]) - 3) * (int(b[j, t]) - int(w_zp[j])) for t in range(k))
            acc[i, j] = s + int(bias[j])
    expected = requantize(acc, multiplier(qp), -5, mode, channel_axis=1)
    out = np.zeros((m, n), dtype=np.int8)
    qgemm(a, b, bias, GemmAttrs(trans_b=True), qp, mode, out)
    np.testing.assert_array_equal(out, expected)


# ------------------------------------------------------------------ dtype discipline


@pytest.mark.parametrize("mode", MODES)
def test_every_kernel_output_is_int8_and_never_float(mode: RequantMode) -> None:
    rng = np.random.default_rng(9)
    qp = make_qparams(0.05, 2, 0.01, 0, 0.1, -3)
    x = rng.integers(-128, 128, size=(1, 3, 6, 6), dtype=np.int8)
    w = rng.integers(-128, 128, size=(4, 3, 3, 3), dtype=np.int8)
    y = conv(x, w, np.zeros(4, dtype=np.int32), ConvAttrs(pads=(1, 1, 1, 1)), qp, mode)
    assert y.dtype == np.int8

    a = rng.integers(-128, 128, size=(2, 8), dtype=np.int8)
    b = rng.integers(-128, 128, size=(8, 3), dtype=np.int8)
    out = np.zeros((2, 3), dtype=np.int8)
    qlinear_matmul(a, b, None, GemmAttrs(), qp, mode, out)
    assert out.dtype == np.int8
    qgemm(a, b, np.zeros(3, dtype=np.int32), GemmAttrs(), qp, mode, out)
    assert out.dtype == np.int8

    # internal stages: im2col stays int32 and requantisation returns int8
    assert im2col(x.astype(np.int32), 3, 3, ConvAttrs()).dtype == np.int32
    assert requantize(np.zeros((2, 2), np.int32), multiplier(qp), 0, mode).dtype == np.int8
