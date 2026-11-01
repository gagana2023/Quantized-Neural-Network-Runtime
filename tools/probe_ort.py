"""Empirical probes of ORT's numerics: which float32 formulation does ORT actually use?

For each question, run the same large input through ORT and through several candidate
formulations, and count how many output elements each candidate gets wrong. A candidate is
consistent with ORT only if it gets none wrong. Results go to results/ort_probe.json and back
the entries in docs/MISMATCHES.md and DESIGN.md D8.

Usage: python -m tools.probe_ort [--seed 0] [--out results/ort_probe.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from int8rt.kernels.pool import reduce_mean
from int8rt.requant import requantize_float
from tools.envinfo import collect
from tools.make_op_tests import Case, generate_cases, run_ort

ROOT = Path(__file__).resolve().parent.parent
F32 = np.float32

MULTIPLIERS: dict[str, Callable[[Any, Any, Any], Any]] = {
    "(xs*ws)/ys  [int8rt]": lambda xs, ws, ys: (xs * ws) / ys,
    "xs*(ws/ys)": lambda xs, ws, ys: xs * (ws / ys),
    "(xs/ys)*ws": lambda xs, ws, ys: (xs / ys) * ws,
    "float64 M, rounded to float32": lambda xs, ws, ys: F32(
        np.float64(xs) * np.float64(ws) / np.float64(ys)
    ),
}


def probe_multiplier(cases: list[Case]) -> dict[str, Any]:
    """QGemmSweep: acc == bias, so ORT's output is requantise(acc) for arbitrary acc."""
    counts = dict.fromkeys(MULTIPLIERS, 0)
    counts["float64 multiply with float64 M"] = 0
    total = 0
    for c in (c for c in cases if c.op == "QGemmSweep"):
        xs, ws, ys = (F32(c.inits[k]) for k in ("xs", "ws", "ys"))
        y_zp, acc = int(c.inits["yz"]), c.inits["c"]
        ref = run_ort(c).reshape(-1)
        total += ref.size
        for name, fn in MULTIPLIERS.items():
            got = requantize_float(acc, np.asarray(fn(xs, ws, ys), dtype=F32), y_zp)
            counts[name] += int((got != ref).sum())
        m64 = np.float64(xs) * np.float64(ws) / np.float64(ys)
        prod = np.clip(np.rint(acc.astype(np.float64) * m64), -128 - y_zp, 127 - y_zp) + y_zp
        counts["float64 multiply with float64 M"] += int((prod.astype(np.int8) != ref).sum())
    return {"elements": total, "wrong_elements": counts}


def probe_quantize(cases: list[Case]) -> dict[str, Any]:
    variants: dict[str, Callable[[npt.NDArray[Any], Any], npt.NDArray[Any]]] = {
        "x / scale  [int8rt]": lambda x, s: x / s,
        "x * (1 / scale)": lambda x, s: x * (F32(1) / s),
    }
    counts = dict.fromkeys(variants, 0)
    total = 0
    for c in (c for c in cases if c.op == "QuantizeSweep"):
        x, scale, zp = c.feeds["x"], F32(c.inits["ys"]), int(c.inits["yz"])
        ref = run_ort(c)
        total += ref.size
        for name, fn in variants.items():
            q = np.clip(np.rint(fn(x, scale)), -128 - zp, 127 - zp) + zp
            counts[name] += int((q.astype(np.int8) != ref).sum())
    return {"elements": total, "wrong_elements": counts}


def _sequential_rows(rows: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    total = rows[:, 0].copy()
    for j in range(1, rows.shape[1]):
        total = total + rows[:, j]
    return total


def probe_reduce_mean(cases: list[Case]) -> dict[str, Any]:
    """Sum order of ReduceMean over trailing axes: numpy pairwise vs left-to-right vs Eigen."""

    def numpy_sum(x: npt.NDArray[np.float32], r: int) -> npt.NDArray[np.float32]:
        return (x.reshape(-1, r).sum(axis=1, dtype=np.float32) / F32(r)).astype(np.float32)

    def left_to_right(x: npt.NDArray[np.float32], r: int) -> npt.NDArray[np.float32]:
        return (_sequential_rows(x.reshape(-1, r)) / F32(r)).astype(np.float32)

    def eigen(x: npt.NDArray[np.float32], r: int) -> npt.NDArray[np.float32]:
        out = np.empty(x.size // r, dtype=np.float32)
        reduce_mean(x.reshape(-1, r), (1,), False, out)
        return out

    variants = {
        "np.sum float32 (pairwise)": numpy_sum,
        "left-to-right": left_to_right,
        "Eigen SSE2 redux emulation  [int8rt]": eigen,
    }
    counts = dict.fromkeys(variants, 0)
    cases_run = 0
    for c in (c for c in cases if c.op == "ReduceMean"):
        x = c.feeds["x"]
        axes = c.params["axes"]
        r = int(np.prod([x.shape[a] for a in axes]))
        ref = run_ort(c).reshape(-1)
        cases_run += 1
        for name, fn in variants.items():
            counts[name] += int(fn(x, r).tobytes() != ref.tobytes())
    return {"cases": cases_run, "cases_with_any_wrong_element": counts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ort_probe.json")
    args = ap.parse_args()
    cases = generate_cases(args.seed)
    record = {
        "env": collect(command="python -m tools.probe_ort " + " ".join(sys.argv[1:])),
        "seed": args.seed,
        "requant_multiplier_order": probe_multiplier(cases),
        "quantize_linear_formula": probe_quantize(cases),
        "reduce_mean_sum_order": probe_reduce_mean(cases),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    for key in ("requant_multiplier_order", "quantize_linear_formula", "reduce_mean_sum_order"):
        print(key, json.dumps(record[key], indent=2))


if __name__ == "__main__":
    main()
