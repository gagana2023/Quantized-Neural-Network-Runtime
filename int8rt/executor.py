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
from int8rt.planner import MemoryPlan, plan_greedy
from int8rt.requant import RequantMode

PlannerMode = Literal["naive", "greedy"]


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


class Executor:
    """Runs a ``Graph``. ``requant`` selects Mode A (``"float"``) or Mode B (``"fixed"``).

    ``planner="naive"`` gives every node output its own buffer. ``"greedy"`` plans one shared
    ``np.int8`` arena (see ``planner.py``): every activation tensor, graph inputs included, is a
    view at its planned offset, so tensors with disjoint lifetimes really share memory.
    ``dump_dir`` writes every input and intermediate tensor as ``.npy``, plus ``index.json``
    (tensor name -> file, in execution order). Each tensor is written as soon as it is produced,
    because with the arena a later tensor may overwrite it.
    """

    def __init__(
        self,
        graph: Graph,
        requant: RequantMode = "float",
        planner: PlannerMode = "naive",
        dump_dir: str | Path | None = None,
    ) -> None:
        if planner not in ("naive", "greedy"):
            raise ValueError(f"unknown planner {planner!r}")
        self.graph = graph
        self.requant = requant
        self.planner = planner
        self.dump_dir = Path(dump_dir) if dump_dir is not None else None
        self._runners: list[Runner] = [op_def(n).prepare(n, graph, requant) for n in graph.nodes]
        # Per node: the names of its non-constant inputs, in order (what the runner receives).
        self._dynamic: list[list[str]] = [
            [t for t in n.inputs if t and not graph.tensors[t].is_initializer] for n in graph.nodes
        ]
        self._index: dict[str, str] = {}
        self.plan: MemoryPlan | None = None
        self._arena: npt.NDArray[np.int8] | None = None
        self._buffers: dict[str, npt.NDArray[Any]] = {}
        self._input_buffers: dict[str, npt.NDArray[Any]] = {}
        if planner == "greedy":
            self.plan = plan_greedy(graph)
            self._arena = np.zeros(self.plan.arena_bytes, dtype=np.int8)
            views = {name: self._view(name) for name in self.plan.offsets}
            self._input_buffers = {name: views[name] for name in graph.inputs}
            self._buffers = {out: views[out] for n in graph.nodes for out in n.outputs}
        else:
            self._buffers = {
                out: np.empty(graph.tensors[out].shape, dtype=graph.tensors[out].dtype)
                for n in graph.nodes
                for out in n.outputs
            }

    def _view(self, name: str) -> npt.NDArray[Any]:
        """The arena bytes of tensor ``name`` seen with its own dtype and shape."""
        assert self.plan is not None and self._arena is not None
        info, start = self.graph.tensors[name], self.plan.offsets[name]
        raw = self._arena[start : start + info.nbytes]
        return raw.view(info.dtype).reshape(info.shape)

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
            if name in self._input_buffers:
                self._input_buffers[name][...] = arr  # into the arena
                arr = self._input_buffers[name]
            values[name] = arr
        order = list(g.inputs)
        self._dump_new(values, order, 0)

        for node, runner, dyn_names in zip(g.nodes, self._runners, self._dynamic, strict=True):
            dyn = [values[t] for t in dyn_names]
            out_name = node.outputs[0]
            runner(dyn, self._buffers[out_name])
            values[out_name] = self._buffers[out_name]
            order.append(out_name)
            self._dump_new(values, order, len(order) - 1)
        return {name: values[name].copy() for name in g.outputs}

    def _dump_new(self, values: dict[str, npt.NDArray[Any]], order: list[str], start: int) -> None:
        """Write ``order[start:]`` to ``dump_dir``; ``index.json`` follows at the end."""
        if self.dump_dir is None:
            return
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self._index = {} if start == 0 else self._index
        for i in range(start, len(order)):
            fname = f"{i:03d}_{_safe_name(order[i])}.npy"
            np.save(self.dump_dir / fname, values[order[i]])
            self._index[order[i]] = fname
        if len(order) == len(self.graph.inputs) + len(self.graph.nodes):
            (self.dump_dir / "index.json").write_text(json.dumps(self._index, indent=2) + "\n")
