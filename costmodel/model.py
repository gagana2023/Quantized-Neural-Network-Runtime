"""Analytical cost model of a generic INT8 accelerator (not any real chip, not cycle-accurate).

Hardware (``AccelConfig``): an R x C MAC array, a weight SRAM, an activation SRAM, one DMA engine
between DRAM and the SRAMs, and a vector unit. Each layer is run in isolation: its input comes from
DRAM and its output goes back to DRAM (no cross-layer residency or fusion).

A GEMM (M x K by K x N) is cut into tiles (mt, nt, kt). On the array, one cycle does one K step for
an mt x nt output block cut into R x C pieces, so a tile takes ceil(mt/R) * ceil(nt/C) * kt cycles.
Two loop orders decide what stays resident in SRAM (and so what is re-fetched from DRAM):

* ``weight_stationary``: for mt: for kt: [load W tile once]; for nt: load X tile, compute.
  Weights are read from DRAM once; inputs are re-read once per M tile. If K is cut (nK > 1) the
  INT32 partial sums of each output tile are spilled to DRAM and read back.
* ``output_stationary``: for mt: for nt: for kt: load W and X tiles, compute; store Y once.
  Accumulators never leave the chip; weights are re-read once per N tile, inputs once per M tile.

Per tile, ``compute`` is array cycles and ``load`` / ``store`` are DMA cycles (bytes / bandwidth,
rounded up per transfer). Without overlap, total = sum(compute + load + store). With double
buffering the DMA works on the neighbouring tiles while the array computes:

    total_db = load[0] + sum_i max(compute[i], load[i+1] + store[i-1]) + store[last]

(the first tile's transfer is exposed, then each step costs the longer of compute and the transfers
that run beside it). Tile sizes are chosen per layer and scheme to minimise ``total_db`` and are
then reused unchanged for the no-overlap figure, so the two differ only in scheduling. Tiles are
sized to half of each SRAM in both cases, so the same tiling is double-bufferable.

Everything is modelled; nothing here is measured on any hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from costmodel.mapping import GemmWork, VectorWork, Work, map_graph
from int8rt.graph import Graph

SCHEMES = ("weight_stationary", "output_stationary")
ACC_BYTES = 4  # INT32 accumulators and partial sums, INT32 bias

I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class AccelConfig:
    name: str
    array_rows: int
    array_cols: int
    weight_sram_bytes: int
    act_sram_bytes: int
    dma_bytes_per_cycle: int
    vector_lanes: int
    clock_mhz: float

    def __post_init__(self) -> None:
        for key, value in self.as_dict().items():
            if key != "name" and not value > 0:
                raise ValueError(f"config {self.name!r}: {key} must be positive, got {value}")

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @property
    def peak_macs_per_cycle(self) -> int:
        return self.array_rows * self.array_cols


def load_config(path: str | Path) -> AccelConfig:
    data = yaml.safe_load(Path(path).read_text())
    return AccelConfig(**data)


@dataclass(frozen=True)
class LayerCost:
    layer: str
    op: str
    kind: str  # "gemm" | "vector" | "layout"
    scheme: str
    groups: int
    gemm_mnk: tuple[int, int, int] | None
    tile: tuple[int, int, int] | None  # (mt, nt, kt) per group
    tiles: int  # total over groups
    macs: int
    vector_ops: int
    dram_weight_bytes: int
    dram_act_in_bytes: int
    dram_act_out_bytes: int
    dram_psum_bytes: int
    sram_read_bytes: int  # array-side reads of weights and inputs
    sram_write_bytes: int  # array-side writes of final outputs
    compute_cycles: int
    transfer_cycles: int
    total_cycles_no_overlap: int
    total_cycles_double_buffered: int

    @property
    def dram_bytes(self) -> int:
        return (
            self.dram_weight_bytes
            + self.dram_act_in_bytes
            + self.dram_act_out_bytes
            + self.dram_psum_bytes
        )

    @property
    def stall_cycles_no_overlap(self) -> int:
        return self.total_cycles_no_overlap - self.compute_cycles

    @property
    def stall_cycles_double_buffered(self) -> int:
        return self.total_cycles_double_buffered - self.compute_cycles

    @property
    def bound(self) -> str:
        if self.compute_cycles == 0 and self.transfer_cycles == 0:
            return "none"
        return "compute" if self.compute_cycles >= self.transfer_cycles else "transfer"

    def utilisation(self, cfg: AccelConfig, double_buffered: bool) -> float | None:
        """Useful MACs / (R * C * total cycles); ``None`` for layers with no MACs."""
        total = (
            self.total_cycles_double_buffered if double_buffered else self.total_cycles_no_overlap
        )
        if self.macs == 0 or total == 0:
            return None
        return self.macs / (cfg.peak_macs_per_cycle * total)


# ---------------------------------------------------------------- scheduling


def _ceil_div(a: I64, b: int) -> I64:
    return (a + (b - 1)) // b


def schedule(compute: I64, load: I64, store: I64) -> tuple[int, int]:
    """(no-overlap cycles, double-buffered cycles) for per-tile compute / load / store cycles."""
    if len(compute) == 0:
        return 0, 0
    no_overlap = int(compute.sum() + load.sum() + store.sum())
    next_load = np.append(load[1:], 0)
    prev_store = np.insert(store[:-1], 0, 0)
    double = int(load[0] + np.maximum(compute, next_load + prev_store).sum() + store[-1])
    return no_overlap, double


# ---------------------------------------------------------------- GEMM layers


def _sizes(total: int, tile: int) -> I64:
    n = math.ceil(total / tile)
    sizes = np.full(n, tile, dtype=np.int64)
    sizes[-1] = total - tile * (n - 1)
    return sizes


@dataclass(frozen=True)
class _Tiles:
    """Per-tile quantities for one group, in execution order."""

    compute: I64
    load_cycles: I64
    store_cycles: I64
    dram_weight: int
    dram_act_in: int
    dram_act_out: int
    dram_psum: int
    sram_read: int
    sram_write: int


def tile_fits(w: GemmWork, cfg: AccelConfig, mt: int, nt: int, kt: int) -> bool:
    """Weights (+ bias) fit half the weight SRAM; input tile + accumulators half the act SRAM."""
    weight = mt * kt + (ACC_BYTES * mt if w.has_bias else 0)
    act = _input_tile_bytes(w, kt * nt) + ACC_BYTES * mt * nt
    return weight <= cfg.weight_sram_bytes // 2 and act <= cfg.act_sram_bytes // 2


def _input_tile_bytes(w: GemmWork, gemm_elems: int) -> int:
    return -(-gemm_elems * w.in_unique // (w.k * w.n)) * w.in_itemsize


def gemm_tiles(w: GemmWork, cfg: AccelConfig, scheme: str, mt: int, nt: int, kt: int) -> _Tiles:
    """Evaluate one tiling of one group's GEMM under ``scheme``."""
    sm, sn, sk = _sizes(w.m, mt), _sizes(w.n, nt), _sizes(w.k, kt)
    n_m, n_n, n_k = len(sm), len(sn), len(sk)
    if scheme == "weight_stationary":
        mi, ki, ni = (a.ravel() for a in np.indices((n_m, n_k, n_n)))
    elif scheme == "output_stationary":
        mi, ni, ki = (a.ravel() for a in np.indices((n_m, n_n, n_k)))
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    tm, tn, tk = sm[mi], sn[ni], sk[ki]
    first_k, last_k = ki == 0, ki == n_k - 1

    bias = ACC_BYTES * tm * first_k if w.has_bias else np.zeros_like(tm)
    weight_load = tm * tk + bias
    if scheme == "weight_stationary":
        weight_load = weight_load * (ni == 0)  # the weight tile stays put while N tiles stream
    act_load = -(-tk * tn * w.in_unique // (w.k * w.n)) * w.in_itemsize
    spill = n_k > 1 and scheme == "weight_stationary"
    psum_read = ACC_BYTES * tm * tn * ~first_k if spill else np.zeros_like(tm)
    psum_write = ACC_BYTES * tm * tn * ~last_k if spill else np.zeros_like(tm)
    out_store = tm * tn * w.out_itemsize * last_k

    bw = cfg.dma_bytes_per_cycle
    r, c = cfg.array_rows, cfg.array_cols
    return _Tiles(
        compute=_ceil_div(tm, r) * _ceil_div(tn, c) * tk,
        load_cycles=_ceil_div(weight_load + act_load + psum_read, bw),
        store_cycles=_ceil_div(out_store + psum_write, bw),
        dram_weight=int(weight_load.sum()),
        dram_act_in=int(act_load.sum()),
        dram_act_out=int(out_store.sum()),
        dram_psum=int((psum_read + psum_write).sum()),
        sram_read=int((_ceil_div(tn, c) * tm * tk + _ceil_div(tm, r) * tk * tn).sum()),
        sram_write=int(out_store.sum()),
    )


def _candidates(total: int, unit: int) -> list[int]:
    """Tile sizes to try along one dimension: unit * 2^j up to the full size."""
    out = {total}
    size = unit
    while size < total:
        out.add(size)
        size *= 2
    return sorted(out)


def _gemm_cost(
    w: GemmWork, cfg: AccelConfig, scheme: str, tiling: tuple[int, int, int]
) -> LayerCost:
    t = gemm_tiles(w, cfg, scheme, *tiling)
    no_overlap, double = schedule(t.compute, t.load_cycles, t.store_cycles)
    g = w.groups  # groups are independent GEMMs run one after another
    return LayerCost(
        layer=w.name,
        op=w.op,
        kind="gemm",
        scheme=scheme,
        groups=g,
        gemm_mnk=(w.m, w.k, w.n),
        tile=tiling,
        tiles=g * len(t.compute),
        macs=w.macs,
        vector_ops=0,
        dram_weight_bytes=g * t.dram_weight,
        dram_act_in_bytes=g * t.dram_act_in,
        dram_act_out_bytes=g * t.dram_act_out,
        dram_psum_bytes=g * t.dram_psum,
        sram_read_bytes=g * t.sram_read,
        sram_write_bytes=g * t.sram_write,
        compute_cycles=g * int(t.compute.sum()),
        transfer_cycles=g * int(t.load_cycles.sum() + t.store_cycles.sum()),
        total_cycles_no_overlap=g * no_overlap,
        total_cycles_double_buffered=g * double,
    )


@cache
def cost_gemm(w: GemmWork, cfg: AccelConfig, scheme: str) -> LayerCost:
    """Best tiling (fewest double-buffered cycles) of ``w`` under ``scheme``."""
    best: tuple[tuple[int, ...], LayerCost] | None = None
    for mt in _candidates(w.m, cfg.array_rows):
        for nt in _candidates(w.n, cfg.array_cols):
            for kt in _candidates(w.k, 1):
                if not tile_fits(w, cfg, mt, nt, kt):
                    continue
                cost = _gemm_cost(w, cfg, scheme, (mt, nt, kt))
                key = (
                    cost.total_cycles_double_buffered,
                    cost.total_cycles_no_overlap,
                    cost.dram_bytes,
                    -mt,
                    -nt,
                    -kt,
                )
                if best is None or key < best[0]:
                    best = (key, cost)
    if best is None:
        raise ValueError(
            f"{w.name}: not even a 1 x 1 x 1 tile fits config {cfg.name!r} "
            f"(weight SRAM {cfg.weight_sram_bytes} B, activation SRAM {cfg.act_sram_bytes} B)"
        )
    return best[1]


# ---------------------------------------------------------------- vector-unit layers


MAX_VECTOR_TILE_FACTOR = 64  # try up to this many times the minimum tile count


def _vector_cost(w: VectorWork, cfg: AccelConfig, scheme: str, n: int) -> LayerCost:
    """``w`` streamed through the vector unit in ``n`` near-equal tiles (none if no data moves)."""
    total_bytes = w.in_bytes + w.out_bytes
    if total_bytes == 0:
        compute = load = store = np.zeros(0, dtype=np.int64)
    else:

        def split(total: int) -> I64:
            return np.full(n, total // n, dtype=np.int64) + (np.arange(n) < total % n)

        compute = _ceil_div(split(w.ops), cfg.vector_lanes)
        load = _ceil_div(split(w.in_bytes), cfg.dma_bytes_per_cycle)
        store = _ceil_div(split(w.out_bytes), cfg.dma_bytes_per_cycle)
    no_overlap, double = schedule(compute, load, store)
    return LayerCost(
        layer=w.name,
        op=w.op,
        kind="vector" if total_bytes else "layout",
        scheme=scheme,
        groups=1,
        gemm_mnk=None,
        tile=None,
        tiles=len(compute),
        macs=0,
        vector_ops=w.ops,
        dram_weight_bytes=0,
        dram_act_in_bytes=w.in_bytes,
        dram_act_out_bytes=w.out_bytes,
        dram_psum_bytes=0,
        sram_read_bytes=w.in_bytes,
        sram_write_bytes=w.out_bytes,
        compute_cycles=int(compute.sum()),
        transfer_cycles=int(load.sum() + store.sum()),
        total_cycles_no_overlap=no_overlap,
        total_cycles_double_buffered=double,
    )


def cost_vector(w: VectorWork, cfg: AccelConfig, scheme: str) -> LayerCost:
    """Best tile count (fewest double-buffered cycles; fewer tiles on ties) for a vector layer.

    A tile's input plus output must fit half the activation SRAM, which sets the minimum count;
    more, smaller tiles can overlap transfer with compute better.
    """
    total_bytes = w.in_bytes + w.out_bytes
    if total_bytes == 0:
        return _vector_cost(w, cfg, scheme, 0)
    n_min = math.ceil(total_bytes / (cfg.act_sram_bytes // 2))
    counts = sorted(
        {min(n_min * 2**j, total_bytes) for j in range(MAX_VECTOR_TILE_FACTOR.bit_length())}
    )
    costs = [_vector_cost(w, cfg, scheme, n) for n in counts]
    return min(costs, key=lambda c: (c.total_cycles_double_buffered, c.tiles))


# ---------------------------------------------------------------- whole graph


def estimate_work(work: list[Work], cfg: AccelConfig) -> list[LayerCost]:
    """One row per (layer, scheme); vector layers have the same row under both schemes."""
    rows: list[LayerCost] = []
    for w in work:
        for scheme in SCHEMES:
            rows.append(
                cost_gemm(w, cfg, scheme)
                if isinstance(w, GemmWork)
                else cost_vector(w, cfg, scheme)
            )
    return rows


def estimate_graph(graph: Graph, cfg: AccelConfig) -> list[LayerCost]:
    return estimate_work(map_graph(graph), cfg)


def with_config(cfg: AccelConfig, **changes: Any) -> AccelConfig:
    return replace(cfg, **changes)
