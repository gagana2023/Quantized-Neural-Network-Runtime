"""Sweeps over the hardware parameters and flat result rows for CSV output.

Three one-at-a-time sweeps around a base config, each re-tiling every layer from scratch:
``sram`` (weight and activation SRAM scaled together), ``bandwidth`` (DMA bytes per cycle) and
``array`` (R x C shape). Everything is modelled cost; nothing is measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from costmodel.mapping import Work
from costmodel.model import SCHEMES, AccelConfig, LayerCost, estimate_work, with_config

SRAM_FACTORS = (0.25, 0.5, 1, 2, 4, 8)
BANDWIDTHS = (1, 2, 4, 8, 16, 32, 64)
ARRAYS = ((4, 4), (8, 8), (16, 16), (32, 32), (64, 64), (8, 32), (32, 8))


@dataclass(frozen=True)
class SweepPoint:
    sweep: str
    label: str
    x: float  # numeric position for plots
    config: AccelConfig


def sweep_points(base: AccelConfig) -> dict[str, list[SweepPoint]]:
    sram = []
    for f in SRAM_FACTORS:
        w, a = int(base.weight_sram_bytes * f), int(base.act_sram_bytes * f)
        cfg = with_config(base, weight_sram_bytes=w, act_sram_bytes=a)
        sram.append(SweepPoint("sram", f"{w // 1024}+{a // 1024} KiB", f * 1.0, cfg))
    bandwidth = [
        SweepPoint("bandwidth", f"{b} B/cycle", float(b), with_config(base, dma_bytes_per_cycle=b))
        for b in BANDWIDTHS
    ]
    array = [
        SweepPoint("array", f"{r}x{c}", float(r * c), with_config(base, array_rows=r, array_cols=c))
        for r, c in ARRAYS
    ]
    return {"sram": sram, "bandwidth": bandwidth, "array": array}


def layer_rows(costs: list[LayerCost], cfg: AccelConfig) -> list[dict[str, Any]]:
    """One flat dict per (layer, scheme); ``costs`` holds each layer's schemes consecutively."""
    rows = []
    for i, c in enumerate(costs):
        m, k, n = c.gemm_mnk or (None, None, None)
        tm, tn, tk = c.tile or (None, None, None)
        rows.append(
            {
                "index": i // len(SCHEMES),
                "layer": c.layer,
                "op": c.op,
                "kind": c.kind,
                "scheme": c.scheme,
                "groups": c.groups,
                "gemm_m": m,
                "gemm_k": k,
                "gemm_n": n,
                "tile_m": tm,
                "tile_n": tn,
                "tile_k": tk,
                "tiles": c.tiles,
                "macs": c.macs,
                "vector_ops": c.vector_ops,
                "dram_weight_bytes": c.dram_weight_bytes,
                "dram_act_in_bytes": c.dram_act_in_bytes,
                "dram_act_out_bytes": c.dram_act_out_bytes,
                "dram_psum_bytes": c.dram_psum_bytes,
                "dram_bytes": c.dram_bytes,
                "sram_read_bytes": c.sram_read_bytes,
                "sram_write_bytes": c.sram_write_bytes,
                "compute_cycles": c.compute_cycles,
                "transfer_cycles": c.transfer_cycles,
                "total_cycles_no_overlap": c.total_cycles_no_overlap,
                "total_cycles_double_buffered": c.total_cycles_double_buffered,
                "stall_cycles_no_overlap": c.stall_cycles_no_overlap,
                "stall_cycles_double_buffered": c.stall_cycles_double_buffered,
                "util_no_overlap": _r(c.utilisation(cfg, False)),
                "util_double_buffered": _r(c.utilisation(cfg, True)),
                "bound": c.bound,
                "modelled_us_double_buffered": round(
                    c.total_cycles_double_buffered / cfg.clock_mhz, 3
                ),
            }
        )
    return rows


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 6)


def totals(costs: list[LayerCost], cfg: AccelConfig) -> dict[str, dict[str, Any]]:
    """Whole-network totals per scheme, plus the per-layer best of the two schemes."""
    out: dict[str, dict[str, Any]] = {}
    by_layer = [costs[i : i + len(SCHEMES)] for i in range(0, len(costs), len(SCHEMES))]
    for j, scheme in enumerate(SCHEMES):
        rows = [layer[j] for layer in by_layer]
        out[scheme] = _sum(rows, cfg)
    best = [min(layer, key=lambda c: c.total_cycles_double_buffered) for layer in by_layer]
    out["best_per_layer"] = _sum(best, cfg)
    return out


def _sum(rows: list[LayerCost], cfg: AccelConfig) -> dict[str, Any]:
    no_overlap = sum(c.total_cycles_no_overlap for c in rows)
    double = sum(c.total_cycles_double_buffered for c in rows)
    compute = sum(c.compute_cycles for c in rows)
    macs = sum(c.macs for c in rows)
    stall_no, stall_db = no_overlap - compute, double - compute
    return {
        "layers": len(rows),
        "macs": macs,
        "compute_cycles": compute,
        "transfer_cycles": sum(c.transfer_cycles for c in rows),
        "total_cycles_no_overlap": no_overlap,
        "total_cycles_double_buffered": double,
        "dma_stall_cycles_no_overlap": stall_no,
        "dma_stall_cycles_double_buffered": stall_db,
        "dma_stall_reduction_percent": round(100 * (1 - stall_db / stall_no), 2)
        if stall_no
        else None,
        "total_cycle_reduction_percent": round(100 * (1 - double / no_overlap), 2),
        "dram_bytes": sum(c.dram_bytes for c in rows),
        "utilisation_double_buffered": round(macs / (cfg.peak_macs_per_cycle * double), 6),
        "compute_bound_layers": sum(c.bound == "compute" for c in rows),
        "transfer_bound_layers": sum(c.bound == "transfer" for c in rows),
        "modelled_latency_us_double_buffered": round(double / cfg.clock_mhz, 3),
    }


def run_sweeps(
    work: list[Work], base: AccelConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(per-layer rows with sweep columns, per-point totals rows)."""
    layer_out: list[dict[str, Any]] = []
    total_out: list[dict[str, Any]] = []
    for points in sweep_points(base).values():
        for p in points:
            costs = estimate_work(work, p.config)
            head = {"sweep": p.sweep, "point": p.label, "x": p.x}
            layer_out += [{**head, **r} for r in layer_rows(costs, p.config)]
            for scheme, t in totals(costs, p.config).items():
                total_out.append({**head, "scheme": scheme, **t})
    return layer_out, total_out
