"""M2 acceptance test: every generated single-op model is bit-exact against ONNX Runtime."""

from __future__ import annotations

import numpy as np
import pytest

from tools.make_op_tests import Case, compare, generate_cases, run_int8rt, run_ort

CASES = generate_cases(seed=0)
OPS = [
    "QLinearConv",
    "QGemm",
    "QLinearMatMul",
    "QLinearGlobalAveragePool",
    "MaxPool",
    "QuantizeLinear",
    "DequantizeLinear",
    "Flatten",
    "ReduceMean",
]


def test_there_are_at_least_200_cases_covering_every_op() -> None:
    assert len(CASES) >= 200
    assert set(OPS) <= {c.op for c in CASES}


def test_generation_is_deterministic() -> None:
    again = generate_cases(seed=0)
    assert [c.case_id for c in again] == [c.case_id for c in CASES]
    assert all(
        np.array_equal(a.feeds[k], b.feeds[k])
        for a, b in zip(again, CASES, strict=True)
        for k in a.feeds
    )


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_float_mode_is_bit_exact_with_ort(case: Case) -> None:
    res = compare(run_ort(case), run_int8rt(case, "float"))
    assert res["match"], f"{case.case_id} {case.params}: {res}"
