"""Static activation memory planner: liveness analysis and a greedy-by-size arena.

Scope is activation tensors only: the graph inputs and every node output. Weights (initialisers)
and kernel scratch are reported separately and are never placed in the arena.

Time is the node index. A tensor is *defined* at the index of the node that produces it (graph
inputs: ``-1``) and is *last used* at the index of its last consumer. Graph inputs and outputs are
pinned: they stay live until the last node has run. Lifetimes are closed intervals, so a node's
output never shares memory with that node's own inputs (kernels read inputs while writing output).
"""

from __future__ import annotations

from dataclasses import dataclass

from int8rt.graph import Graph

ALIGN = 64  # arena offsets are multiples of this many bytes


def _align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


@dataclass(frozen=True)
class Lifetime:
    """One activation tensor: size in bytes and the closed step interval ``[first, last]``."""

    name: str
    nbytes: int
    first: int
    last: int

    def overlaps(self, other: Lifetime) -> bool:
        return self.first <= other.last and other.first <= self.last


@dataclass(frozen=True)
class MemoryPlan:
    """Arena offsets plus the three headline numbers (all in bytes)."""

    lifetimes: dict[str, Lifetime]
    offsets: dict[str, int]
    arena_bytes: int  # planned peak: size of the one shared arena
    naive_bytes: int  # sum of all tensor sizes (every tensor its own buffer)
    lower_bound_bytes: int  # max over steps of the total size of tensors live at that step


def liveness(graph: Graph) -> dict[str, Lifetime]:
    """Lifetime of every activation tensor (graph inputs and node outputs), in definition order."""
    n = len(graph.nodes)
    first: dict[str, int] = {name: -1 for name in graph.inputs}
    for i, node in enumerate(graph.nodes):
        for out in node.outputs:
            first[out] = i
    last = dict(first)
    for i, node in enumerate(graph.nodes):
        for t in node.inputs:
            if t in last:  # skips "" and initialisers
                last[t] = max(last[t], i)
    for name in (*graph.inputs, *graph.outputs):  # pinned
        last[name] = n - 1
    return {
        name: Lifetime(name, graph.tensors[name].nbytes, first[name], last[name]) for name in first
    }


def lower_bound(lifetimes: dict[str, Lifetime]) -> int:
    """Max over steps of the total size of live tensors: no placement can use less."""
    steps = {s for lt in lifetimes.values() for s in (lt.first, lt.last)}
    return max(
        (sum(lt.nbytes for lt in lifetimes.values() if lt.first <= s <= lt.last) for s in steps),
        default=0,
    )


def plan_greedy(graph: Graph) -> MemoryPlan:
    """Greedy-by-size: largest tensor first, each at the lowest aligned offset that does not
    collide with an already placed tensor whose lifetime overlaps."""
    lifetimes = liveness(graph)
    # Ties broken by definition step then name, so the plan is deterministic.
    order = sorted(lifetimes.values(), key=lambda lt: (-lt.nbytes, lt.first, lt.name))
    offsets: dict[str, int] = {}
    for lt in order:
        busy = sorted(
            (offsets[o.name], offsets[o.name] + o.nbytes)
            for o in (lifetimes[name] for name in offsets)
            if lt.overlaps(o)
        )
        candidate = 0
        for start, end in busy:
            if candidate + lt.nbytes <= start:
                break  # fits in the gap before this block
            candidate = max(candidate, _align_up(end))
        offsets[lt.name] = candidate
    arena = _align_up(max((offsets[n] + lifetimes[n].nbytes for n in offsets), default=0))
    return MemoryPlan(
        lifetimes=lifetimes,
        offsets=offsets,
        arena_bytes=arena,
        naive_bytes=sum(lt.nbytes for lt in lifetimes.values()),
        lower_bound_bytes=lower_bound(lifetimes),
    )


def find_overlaps(plan: MemoryPlan) -> list[tuple[str, str]]:
    """Pairs of tensors that are live at the same time *and* share bytes (must be empty)."""
    names = list(plan.offsets)
    bad: list[tuple[str, str]] = []
    for i, a in enumerate(names):
        la = plan.lifetimes[a]
        for b in names[i + 1 :]:
            lb = plan.lifetimes[b]
            if not la.overlaps(lb):
                continue
            a0, b0 = plan.offsets[a], plan.offsets[b]
            if a0 < b0 + lb.nbytes and b0 < a0 + la.nbytes:
                bad.append((a, b))
    return bad


def weights_bytes(graph: Graph) -> int:
    """Total size of the initialisers (weights, biases, scales, zero points)."""
    return sum(t.nbytes for t in graph.tensors.values() if t.is_initializer)


def im2col_scratch_bytes(graph: Graph) -> int:
    """Largest int32 im2col matrix a QLinearConv node builds (the host kernel's main scratch).

    This describes *this runtime's* NumPy kernel, not a hardware scratchpad; NumPy also makes
    other temporaries (padded input, int64 accumulator) that are not counted here.
    """
    best = 0
    for node in graph.nodes:
        if node.op_type != "QLinearConv":
            continue
        n, c, _, _ = graph.tensors[node.inputs[0]].shape
        _, _, kh, kw = graph.tensors[node.inputs[3]].shape  # weight is OIHW: (O, C/g, kh, kw)
        _, o, oh, ow = graph.tensors[node.outputs[0]].shape
        g = int(node.attrs.get("group", 1))
        best = max(best, n * (c // g) * kh * kw * oh * ow * 4)
    return best
