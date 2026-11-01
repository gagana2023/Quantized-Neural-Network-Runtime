"""Command line: ``python -m int8rt run|memreport``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from int8rt.executor import Executor
from int8rt.importer import load_model
from int8rt.planner import im2col_scratch_bytes, plan_greedy, weights_bytes

KIB = 1024


def memory_report(model: str | Path) -> dict[str, Any]:
    """Naive / planned / lower-bound activation memory, weights and scratch (KiB)."""
    graph = load_model(model)
    plan = plan_greedy(graph)
    return {
        "model": str(model),
        "activation_kib": {
            "naive": plan.naive_bytes / KIB,
            "planned": plan.arena_bytes / KIB,
            "lower_bound": plan.lower_bound_bytes / KIB,
        },
        "weights_kib": weights_bytes(graph) / KIB,
        "im2col_scratch_kib": im2col_scratch_bytes(graph) / KIB,
        "tensors": len(plan.lifetimes),
    }


def _load_inputs(path: Path, names: list[str]) -> dict[str, npt.NDArray[Any]]:
    if path.suffix == ".npz":
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    if len(names) != 1:
        raise SystemExit(f"model has inputs {names}; pass a .npz with one array per input")
    return {names[0]: np.load(path)}


def _run(args: argparse.Namespace) -> None:
    graph = load_model(args.model)
    ex = Executor(graph, requant=args.requant, planner=args.planner, dump_dir=args.dump_all)
    outputs = ex.run(_load_inputs(Path(args.input), graph.inputs))
    out = Path(args.output)
    if len(outputs) == 1 and out.suffix != ".npz":
        np.save(out, next(iter(outputs.values())))
    else:
        np.savez(out, **outputs)  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m int8rt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run a model on one input")
    run.add_argument("--model", required=True)
    run.add_argument("--input", required=True, help=".npy (single input) or .npz (name -> array)")
    run.add_argument("--output", required=True, help=".npy (single output) or .npz")
    run.add_argument("--requant", choices=["float", "fixed"], default="float")
    run.add_argument("--planner", choices=["naive", "greedy"], default="naive")
    run.add_argument("--dump-all", metavar="DIR", help="write every intermediate tensor here")
    run.set_defaults(func=_run)
    mem = sub.add_parser("memreport", help="activation memory: naive / planned / lower bound")
    mem.add_argument("--model", required=True)
    mem.set_defaults(func=lambda a: print(json.dumps(memory_report(a.model), indent=2)))
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
