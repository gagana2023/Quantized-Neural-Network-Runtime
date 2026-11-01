"""Internal graph format: nodes in topological order plus tensor metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class TensorInfo:
    """One tensor: static shape, dtype, optional quantisation parameters, optional constant data.

    ``scale`` / ``zero_point`` are the per-tensor quantisation parameters of an *activation* tensor
    (taken from the node that produces it). They are ``None`` for float tensors and for
    initialisers, whose parameters live in their own scale / zero-point initialisers.
    ``data`` is set for initialisers only.
    """

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    scale: np.float32 | None = None
    zero_point: int | None = None
    data: npt.NDArray[Any] | None = None

    @property
    def is_initializer(self) -> bool:
        return self.data is not None

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize


@dataclass
class Node:
    """One operator. ``inputs`` may contain ``""`` for an omitted optional input."""

    name: str
    op_type: str
    domain: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return self.domain, self.op_type


@dataclass
class Graph:
    """Nodes in topological order, every tensor by name, and the graph input / output names.

    ``inputs`` excludes initialisers.
    """

    nodes: list[Node]
    tensors: dict[str, TensorInfo]
    inputs: list[str]
    outputs: list[str]

    def initializer(self, name: str) -> npt.NDArray[Any]:
        data = self.tensors[name].data
        if data is None:
            raise KeyError(f"tensor {name!r} is not an initialiser")
        return data

    def producer_index(self) -> dict[str, int]:
        """Tensor name -> index of the node that produces it."""
        return {out: i for i, n in enumerate(self.nodes) for out in n.outputs}
