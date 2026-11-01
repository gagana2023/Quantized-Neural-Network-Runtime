"""Sequential executor: run nodes in topological order, each writing into a preallocated output."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from int8rt.graph import Graph
from int8rt.ops import Runner, op_def
from int8rt.requant import RequantMode

PlannerMode = Literal["naive", "greedy"]


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


class Executor:
    """Runs a ``Graph``. ``requant`` selects Mode A (``"float"``) or Mode B (``"fixed"``).

    ``planner="naive"`` gives every activation tensor its own buffer. ``"greedy"`` (a shared
    arena) arrives in M4. ``dump_dir`` writes every input and intermediate tensor as ``.npy``,
    plus ``index.json`` (tensor name -> file, in execution order).
    """

    def __init__(
        self,
        graph: Graph,
        requant: RequantMode = "float",
        planner: PlannerMode = "naive",
        dump_dir: str | Path | None = None,
    ) -> None:
        if planner != "naive":
            raise NotImplementedError("the greedy arena planner is implemented in M4")
        self.graph = graph
        self.requant = requant
        self.dump_dir = Path(dump_dir) if dump_dir is not None else None
        self._runners: list[Runner] = [op_def(n).prepare(n, graph, requant) for n in graph.nodes]
        # Per node: the names of its non-constant inputs, in order (what the runner receives).
        self._dynamic: list[list[str]] = [
            [t for t in n.inputs if t and not graph.tensors[t].is_initializer] for n in graph.nodes
        ]
        self._buffers: dict[str, npt.NDArray[Any]] = {
            out: np.empty(graph.tensors[out].shape, dtype=graph.tensors[out].dtype)
            for n in graph.nodes
            for out in n.outputs
        }

    def run(self, inputs: dict[str, npt.NDArray[Any]]) -> dict[str, npt.NDArray[Any]]:
        g = self.graph
        if set(inputs) != set(g.inputs):
            raise ValueError(f"expected inputs {sorted(g.inputs)}, got {sorted(inputs)}")
        values: dict[str, npt.NDArray[Any]] = {}
        for name in g.inputs:
            info, arr = g.tensors[name], inputs[name]
            if arr.shape != info.shape or arr.dtype != info.dtype:
                raise ValueError(
                    f"input {name!r}: expected {info.dtype}{info.shape}, got {arr.dtype}{arr.shape}"
                )
            values[name] = arr
        order = list(g.inputs)

        for node, runner, dyn_names in zip(g.nodes, self._runners, self._dynamic, strict=True):
            dyn = [values[t] for t in dyn_names]
            out_name = node.outputs[0]
            runner(dyn, self._buffers[out_name])
            values[out_name] = self._buffers[out_name]
            order.append(out_name)

        if self.dump_dir is not None:
            self._dump(values, order)
        return {name: values[name].copy() for name in g.outputs}

    def _dump(self, values: dict[str, npt.NDArray[Any]], order: list[str]) -> None:
        assert self.dump_dir is not None
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        index: dict[str, str] = {}
        for i, name in enumerate(order):
            fname = f"{i:03d}_{_safe_name(name)}.npy"
            np.save(self.dump_dir / fname, values[name])
            index[name] = fname
        (self.dump_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
