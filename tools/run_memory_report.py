"""Activation memory report: naive vs greedy-planned vs lower bound, plus the planner on/off check.

For each model this writes ``results/memory_<model>.json`` with
  * the three headline numbers (KiB), weights and im2col scratch reported separately,
  * the per-tensor plan (size, lifetime, arena offset),
  * the overlap check on the plan,
  * a byte-for-byte comparison of int8rt with the planner off (separate buffers) and on (arena),
    in both requantisation modes, on real test-set inputs: final outputs for every input, and every
    intermediate tensor for the first few. The arena is refilled with random bytes before each run
    so stale data cannot hide a kernel that fails to write its whole output.

Everything here is a property of the model graph and of int8rt's buffers; no hardware is modelled.

Usage: python -m tools.run_memory_report --model cifar|kws [--inputs 100]
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from int8rt.cli import memory_report
from int8rt.executor import Executor
from int8rt.importer import load_model
from int8rt.planner import find_overlaps, plan_greedy
from int8rt.requant import RequantMode
from tools import envinfo
from tools.compare_ort import MODELS, ModelSpec, load_dump

ROOT = Path(__file__).resolve().parent.parent
INTERMEDIATE_INPUTS = 5  # inputs for which every intermediate tensor is compared as well


def planner_equivalence(
    spec: ModelSpec, images: np.ndarray[Any, Any], mode: RequantMode, n_inputs: int
) -> dict[str, Any]:
    graph = load_model(spec.int8_path)
    off = Executor(graph, requant=mode, planner="naive")
    on = Executor(graph, requant=mode, planner="greedy")
    assert on._arena is not None
    rng = np.random.default_rng(0)
    outputs_same = tensors_same = tensors_total = 0
    for i in range(n_inputs):
        feed = {spec.input_name: images[i : i + 1]}
        on._arena[:] = rng.integers(-128, 128, on._arena.shape, dtype=np.int8)
        if i < INTERMEDIATE_INPUTS:
            with tempfile.TemporaryDirectory() as tmp:
                off.dump_dir, on.dump_dir = Path(tmp) / "off", Path(tmp) / "on"
                r_off, r_on = off.run(feed), on.run(feed)
                d_off, d_on = load_dump(off.dump_dir), load_dump(on.dump_dir)
            off.dump_dir = on.dump_dir = None
            tensors_total += len(d_off)
            tensors_same += sum(d_on[k].tobytes() == v.tobytes() for k, v in d_off.items())
        else:
            r_off, r_on = off.run(feed), on.run(feed)
        outputs_same += all(r_on[k].tobytes() == r_off[k].tobytes() for k in r_off)
    return {
        "requant": mode,
        "inputs": n_inputs,
        "final_outputs_identical": outputs_same,
        "intermediate_tensor_comparisons": tensors_total,
        "intermediate_tensors_identical": tensors_same,
    }


def build_report(spec: ModelSpec, n_inputs: int, command: str) -> dict[str, Any]:
    images, _ = spec.load_test_set()
    n_inputs = min(n_inputs, len(images))
    graph = load_model(spec.int8_path)
    plan = plan_greedy(graph)
    overlaps = find_overlaps(plan)
    tensors = [
        {
            "name": lt.name,
            "dtype": str(graph.tensors[lt.name].dtype),
            "shape": list(graph.tensors[lt.name].shape),
            "bytes": lt.nbytes,
            "first_step": lt.first,
            "last_step": lt.last,
            "arena_offset": plan.offsets[lt.name],
        }
        for lt in plan.lifetimes.values()
    ]
    mem = memory_report(spec.int8_path)
    act = mem["activation_kib"]
    return {
        "env": envinfo.collect(command),
        "what": "Activation memory of int8rt on this model (a property of the graph and the "
        "planner; not measured host RSS, not an accelerator figure). Activations = graph input "
        "(float32, pinned) + every node output; weights and scratch are reported separately.",
        "model": spec.name,
        "model_file": str(spec.int8_path.relative_to(ROOT)),
        "activation_kib": act,
        "planned_over_lower_bound_kib": act["planned"] - act["lower_bound"],
        "reduction_naive_to_planned_percent": round(100 * (1 - act["planned"] / act["naive"]), 2),
        "weights_kib": mem["weights_kib"],
        "im2col_scratch_kib": mem["im2col_scratch_kib"],
        "im2col_scratch_note": "largest int32 im2col matrix of one QLinearConv call in int8rt's "
        "NumPy kernel; other NumPy temporaries are not counted; not part of the arena",
        "alignment_bytes": 64,
        "arena_bytes": plan.arena_bytes,
        "overlapping_live_tensor_pairs": [list(p) for p in overlaps],
        "tensors": tensors,
        "planner_on_vs_off": [
            planner_equivalence(spec, images, mode, n_inputs) for mode in ("float", "fixed")
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--inputs", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    spec = MODELS[args.model]
    cmd = "python -m tools.run_memory_report " + shlex.join(sys.argv[1:])
    report = build_report(spec, args.inputs, cmd)
    args.out_dir.mkdir(exist_ok=True)
    path = args.out_dir / f"memory_{spec.name}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    a = report["activation_kib"]
    print(
        f"{spec.name}: naive {a['naive']:.2f} KiB, planned {a['planned']:.2f} KiB, "
        f"lower bound {a['lower_bound']:.2f} KiB; weights {report['weights_kib']:.2f} KiB"
    )
    ok = not report["overlapping_live_tensor_pairs"] and all(
        r["final_outputs_identical"] == r["inputs"]
        and r["intermediate_tensors_identical"] == r["intermediate_tensor_comparisons"]
        for r in report["planner_on_vs_off"]
    )
    print(f"wrote {path}; planner on == off and no overlaps: {ok}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
