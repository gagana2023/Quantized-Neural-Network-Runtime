"""M5 tests: hand-computed cost-model cases, mapping from graphs, invariants on the real models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from costmodel.mapping import GemmWork, VectorWork, map_graph
from costmodel.model import (
    SCHEMES,
    AccelConfig,
    _gemm_cost,
    _vector_cost,
    cost_gemm,
    cost_vector,
    estimate_graph,
    gemm_tiles,
    load_config,
    schedule,
    tile_fits,
)
from int8rt.graph import Graph, Node, TensorInfo
from int8rt.importer import load_model
from tests.test_executor import CIFAR, KWS

I8 = np.dtype(np.int8)
ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "costmodel" / "configs" / "base.yaml"

# 4x4 array, 4 B/cycle DMA, generous SRAM (so any tile fits), 4 vector lanes.
TINY = AccelConfig("tiny", 4, 4, 4096, 4096, 4, 4, 100.0)


def gemm(m: int, k: int, n: int, groups: int = 1, bias: bool = False) -> GemmWork:
    return GemmWork("g", "QLinearConv", groups, m, k, n, k * n, bias, 1, 1)


def arr(*v: int) -> np.ndarray:
    return np.array(v, dtype=np.int64)


# ---------------------------------------------------------------- schedule


def test_schedule_single_tile_has_nothing_to_overlap() -> None:
    assert schedule(arr(3), arr(6), arr(4)) == (13, 13)


def test_schedule_overlaps_neighbours() -> None:
    # compute 3,3  load 6,6  store 4,4:
    # no overlap 6 + 12 + 8 = 26; double buffered 6 + max(3, 6+0) + max(3, 0+4) + 4 = 20
    assert schedule(arr(3, 3), arr(6, 6), arr(4, 4)) == (26, 20)


def test_schedule_empty() -> None:
    z = np.zeros(0, dtype=np.int64)
    assert schedule(z, z, z) == (0, 0)


# ---------------------------------------------------------------- hand-computed GEMMs


def test_single_tile_gemm() -> None:
    # M=4 K=3 N=4, one tile. compute 1*1*3 = 3; load W 12 B + X 12 B = 24 B -> 6; store 16 B -> 4.
    c = _gemm_cost(gemm(4, 3, 4), TINY, "output_stationary", (4, 4, 3))
    assert (c.compute_cycles, c.transfer_cycles) == (3, 10)
    assert (c.total_cycles_no_overlap, c.total_cycles_double_buffered) == (13, 13)
    assert (c.dram_weight_bytes, c.dram_act_in_bytes, c.dram_act_out_bytes) == (12, 12, 16)
    assert c.utilisation(TINY, True) == pytest.approx(48 / (16 * 13))
    assert c.bound == "transfer"  # 3 compute cycles < 10 transfer cycles


def test_two_tiles_output_vs_weight_stationary() -> None:
    w = gemm(4, 3, 8)  # two N tiles of 4: tile (4, 4, 3)
    # OS: each tile loads W 12 + X 12 = 24 B -> 6 cycles, stores 16 B -> 4. nodb 2*(3+6+4) = 26;
    #     db = 6 + max(3, 6) + max(3, 4) + 4 = 20; weight bytes 24.
    os_ = _gemm_cost(w, TINY, "output_stationary", (4, 4, 3))
    assert (os_.total_cycles_no_overlap, os_.total_cycles_double_buffered) == (26, 20)
    assert os_.dram_weight_bytes == 24
    # WS: W is loaded for the first tile only: loads 6, 3; stores 4, 4; compute 3, 3.
    #     nodb 6 + 9 + 8 = 23; db = 6 + max(3, 3) + max(3, 4) + 4 = 17; weight bytes 12.
    ws = _gemm_cost(w, TINY, "weight_stationary", (4, 4, 3))
    assert (ws.total_cycles_no_overlap, ws.total_cycles_double_buffered) == (23, 17)
    assert ws.dram_weight_bytes == 12
    assert ws.dram_act_in_bytes == os_.dram_act_in_bytes == 24


def test_weight_stationary_spills_partial_sums_when_k_is_cut() -> None:
    w = gemm(4, 4, 4)  # tile (4, 4, 2): two K tiles
    # WS t0: 8 B W + 8 B X -> 4 cycles, writes 4*4*4 = 64 B psums -> 16; t1: 16 B + 64 B psum read
    # -> 20 cycles, stores 16 B -> 4. compute 2, 2. nodb 4 + 24 + 20 = 48;
    # db = 4 + max(2, 20) + max(2, 16) + 4 = 44; psum bytes 128.
    ws = _gemm_cost(w, TINY, "weight_stationary", (4, 4, 2))
    assert (ws.total_cycles_no_overlap, ws.total_cycles_double_buffered) == (48, 44)
    assert ws.dram_psum_bytes == 128
    # OS keeps accumulators on chip: loads 4, 4 cycles; one 16 B store (4). nodb 4 + 8 + 4 = 16;
    # db = 4 + max(2, 4) + max(2, 0) + 4 = 14.
    os_ = _gemm_cost(w, TINY, "output_stationary", (4, 4, 2))
    assert (os_.total_cycles_no_overlap, os_.total_cycles_double_buffered) == (16, 14)
    assert os_.dram_psum_bytes == 0


def test_bias_is_loaded_with_the_first_k_tile() -> None:
    t = gemm_tiles(gemm(4, 4, 4, bias=True), TINY, "output_stationary", 4, 4, 2)
    assert t.dram_weight == 4 * 4 + 4 * 4  # all weights + 4 INT32 biases (16 B), loaded once


def test_depthwise_uses_one_row_of_the_array() -> None:
    # 3 groups, each M=1 K=9 N=4, tile (1, 4, 9): compute 1*1*9 = 9; load 9 + 36 = 45 B -> 12;
    # store 4 B -> 1. Per group 9 + 12 + 1 = 22, three groups 66; MACs 3*36 = 108.
    c = _gemm_cost(gemm(1, 9, 4, groups=3), TINY, "output_stationary", (1, 4, 9))
    assert (c.compute_cycles, c.total_cycles_no_overlap) == (27, 66)
    assert c.macs == 108
    assert c.utilisation(TINY, False) == pytest.approx(108 / (16 * 66))
    # even with free DMA, one of R rows is busy: macs / (R*C*compute) = 1/R
    assert c.macs / (16 * c.compute_cycles) == pytest.approx(1 / TINY.array_rows)


def test_edge_tiles_use_remainder_sizes() -> None:
    # N=6 in tiles of 4 -> 4 and 2 columns; both need one column block. compute 3 + 3.
    t = gemm_tiles(gemm(4, 3, 6), TINY, "output_stationary", 4, 4, 3)
    assert list(t.compute) == [3, 3]
    assert t.dram_act_out == 24 and t.dram_act_in == 18  # 12 + 6 input bytes


def test_input_density_of_a_convolution() -> None:
    # 3x3 conv, 1 channel, 5x5 input and output: K=9, N=25, unique input 25 of K*N = 225.
    w = GemmWork("c", "QLinearConv", 1, 2, 9, 25, 25, False, 1, 1)
    t = gemm_tiles(w, TINY, "output_stationary", 2, 25, 9)
    assert t.dram_act_in == 25


def test_tile_fits_uses_half_of_each_sram() -> None:
    cfg = AccelConfig("c", 4, 4, 64, 256, 4, 4, 100.0)  # halves: 32 B weights, 128 B activations
    assert tile_fits(gemm(4, 8, 4), cfg, 4, 4, 8)  # 32 B weights; 32 B input + 64 B accumulators
    assert not tile_fits(gemm(5, 8, 4), cfg, 5, 4, 8)  # 40 B weights > 32
    assert not tile_fits(gemm(4, 8, 8), cfg, 4, 8, 8)  # 64 B input + 128 B accumulators > 128


def test_search_result_fits_and_is_never_better_than_exhaustive() -> None:
    cfg = AccelConfig("c", 4, 4, 128, 512, 4, 4, 100.0)
    w = gemm(8, 12, 12)
    for scheme in SCHEMES:
        best = cost_gemm(w, cfg, scheme)
        every = [
            _gemm_cost(w, cfg, scheme, (mt, nt, kt)).total_cycles_double_buffered
            for mt in range(1, 9)
            for nt in range(1, 13)
            for kt in range(1, 13)
            if tile_fits(w, cfg, mt, nt, kt)
        ]
        # the search tries only unit * 2^j tile sizes, so it cannot beat the exhaustive minimum
        assert best.total_cycles_double_buffered >= min(every)
        assert best.tile is not None and tile_fits(w, cfg, *best.tile)


def test_nothing_fits() -> None:
    cfg = AccelConfig("c", 4, 4, 2, 2, 4, 4, 100.0)
    with pytest.raises(ValueError, match="not even"):
        cost_gemm(gemm(4, 4, 4), cfg, "output_stationary")


def test_vector_layer() -> None:
    cfg = AccelConfig("v", 4, 4, 4096, 64, 4, 4, 100.0)  # half of act SRAM = 32 B
    w = VectorWork("v", "MaxPool", 64, 64, 128)
    # 4 tiles of 16 B in / 16 B out / 32 ops: compute 8, load 4, store 4 each.
    # nodb 4 * 16 = 64; db = 4 + 4 * max(8, <=8) + 4 = 40.
    c = _vector_cost(w, cfg, "weight_stationary", 4)
    assert (c.compute_cycles, c.transfer_cycles) == (32, 32)
    assert (c.total_cycles_no_overlap, c.total_cycles_double_buffered) == (64, 40)
    assert c.macs == 0 and c.utilisation(cfg, True) is None
    # 8 tiles of 8 B / 8 B / 16 ops: compute 4, load 2, store 2: db = 2 + 8 * 4 + 2 = 36
    assert _vector_cost(w, cfg, "weight_stationary", 8).total_cycles_double_buffered == 36
    # the search takes at least the minimum 4 tiles and finds something no worse than 36
    best = cost_vector(w, cfg, "weight_stationary")
    assert best.tiles >= 4 and best.total_cycles_double_buffered <= 36
    assert best.total_cycles_double_buffered >= best.compute_cycles


def test_flatten_costs_nothing() -> None:
    c = cost_vector(VectorWork("f", "Flatten", 0, 0, 0), TINY, "weight_stationary")
    assert c.total_cycles_no_overlap == c.total_cycles_double_buffered == 0 and c.bound == "none"


def _graph(nodes: list[Node], shapes: dict[str, tuple[int, ...]], consts: set[str]) -> Graph:
    tensors = {
        n: TensorInfo(n, s, I8, data=np.zeros(s, np.int8) if n in consts else None)
        for n, s in shapes.items()
    }
    return Graph(nodes, tensors, [nodes[0].inputs[0]], [nodes[-1].outputs[0]])


def test_map_conv_depthwise_and_gemm() -> None:
    conv = Node(
        "conv", "QLinearConv", "", ["x", "", "", "w", "", "", "", "", "b"], ["y"], {"group": 1}
    )
    dw = Node("dw", "QLinearConv", "", ["y", "", "", "w2", "", "", "", ""], ["z"], {"group": 8})
    fc = Node("fc", "QGemm", "com.microsoft", ["f", "", "", "wf", "", "", ""], ["o"], {"transB": 1})
    shapes = {
        "x": (1, 4, 5, 5), "w": (8, 4, 3, 3), "b": (8,), "y": (1, 8, 5, 5),
        "w2": (8, 1, 3, 3), "z": (1, 8, 5, 5),
        "f": (1, 32), "wf": (10, 32), "o": (1, 10),
    }  # fmt: skip
    g = _graph([conv, dw, fc], shapes, {"w", "b", "w2", "wf"})
    c, d, f = map_graph(g)
    assert isinstance(c, GemmWork) and isinstance(d, GemmWork) and isinstance(f, GemmWork)
    assert (c.groups, c.m, c.k, c.n, c.in_unique, c.has_bias) == (1, 8, 36, 25, 100, True)
    assert (d.groups, d.m, d.k, d.n, d.in_unique, d.has_bias) == (8, 1, 9, 25, 25, False)
    assert (f.groups, f.m, f.k, f.n) == (1, 10, 32, 1)
    assert c.macs == 8 * 36 * 25 and d.macs == 8 * 9 * 25 and f.macs == 320


def test_base_config_loads_and_validates() -> None:
    cfg = load_config(BASE)
    assert cfg.name == "base" and cfg.peak_macs_per_cycle == 256
    with pytest.raises(ValueError, match="positive"):
        AccelConfig("bad", 0, 4, 1, 1, 1, 1, 1.0)


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(
            CIFAR, marks=pytest.mark.skipif(not CIFAR.exists(), reason="models/ is generated")
        ),
        pytest.param(KWS, marks=pytest.mark.skipif(not KWS.exists(), reason="KWS comes from P2")),
    ],
)
def test_invariants_on_real_models(path: Path) -> None:
    cfg = load_config(BASE)
    g = load_model(path)
    rows = estimate_graph(g, cfg)
    assert len(rows) == 2 * len(g.nodes)
    for r in rows:
        assert r.total_cycles_double_buffered <= r.total_cycles_no_overlap
        assert r.total_cycles_no_overlap == r.compute_cycles + r.transfer_cycles
        u = r.utilisation(cfg, True)
        assert u is None or 0 < u <= 1
        if r.gemm_mnk is not None:  # every MAC is computed; weights are read at least once
            assert r.compute_cycles * cfg.peak_macs_per_cycle >= r.macs
            assert r.dram_weight_bytes >= r.groups * r.gemm_mnk[0] * r.gemm_mnk[1]
