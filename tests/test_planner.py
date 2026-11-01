"""M4 tests: liveness, the greedy arena, the overlap check, and the executor with the planner on."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from int8rt import cli
from int8rt.executor import Executor
from int8rt.graph import Graph, Node, TensorInfo
from int8rt.importer import load_model
from int8rt.planner import (
    ALIGN,
    find_overlaps,
    im2col_scratch_bytes,
    liveness,
    plan_greedy,
    weights_bytes,
)
from tests.test_executor import CIFAR, KWS, KWS_FEATURES, tiny_graph

I8 = np.dtype(np.int8)


def synthetic_graph() -> Graph:
    """x(128) -> a(256) -> b(64); (b, a) -> c(192) -> d(64). Sizes are multiples of 64 B.

    Hand-computed lifetimes (node index; inputs at -1; pinned = through the last node, 3):
    x [-1,3]  a [0,2]  b [1,2]  c [2,3]  d [3,3].
    """
    sizes = {"x": 128, "a": 256, "b": 64, "c": 192, "d": 64}
    tensors = {n: TensorInfo(n, (s,), I8) for n, s in sizes.items()}
    nodes = [
        Node("n0", "Op", "", ["x"], ["a"]),
        Node("n1", "Op", "", ["a"], ["b"]),
        Node("n2", "Op", "", ["b", "a"], ["c"]),
        Node("n3", "Op", "", ["c"], ["d"]),
    ]
    return Graph(nodes, tensors, ["x"], ["d"])


def test_liveness_by_hand() -> None:
    lt = liveness(synthetic_graph())
    assert {n: (t.first, t.last) for n, t in lt.items()} == {
        "x": (-1, 3),
        "a": (0, 2),
        "b": (1, 2),
        "c": (2, 3),
        "d": (3, 3),
    }


def test_three_numbers_by_hand() -> None:
    plan = plan_greedy(synthetic_graph())
    assert plan.naive_bytes == 128 + 256 + 64 + 192 + 64  # 704
    assert plan.lower_bound_bytes == 640  # step 2: x + a + b + c
    # largest first: a@0, c@256, x@448, b@576; d (live only at step 3) fits in the hole at 0
    assert plan.offsets == {"a": 0, "c": 256, "x": 448, "b": 576, "d": 0}
    assert plan.arena_bytes == 640
    assert find_overlaps(plan) == []


def test_tensor_is_not_placed_over_its_own_node_inputs() -> None:
    # b (produced by n1) must not share bytes with a (consumed by n1): lifetimes are closed.
    plan = plan_greedy(synthetic_graph())
    a0, b0 = plan.offsets["a"], plan.offsets["b"]
    assert not (a0 < b0 + 64 and b0 < a0 + 256)


def test_offsets_are_aligned_and_arena_covers_everything() -> None:
    g = synthetic_graph()
    g.tensors["b"] = TensorInfo("b", (70,), I8)  # an odd size forces padding for the next tensor
    plan = plan_greedy(g)
    assert all(off % ALIGN == 0 for off in plan.offsets.values())
    assert plan.arena_bytes >= max(plan.offsets[n] + plan.lifetimes[n].nbytes for n in plan.offsets)
    assert find_overlaps(plan) == []


def test_planned_is_between_lower_bound_and_naive_bytes() -> None:
    for g in (synthetic_graph(), tiny_graph()):
        plan = plan_greedy(g)
        assert plan.lower_bound_bytes <= plan.arena_bytes
        assert plan.lower_bound_bytes <= plan.naive_bytes


def test_overlap_check_catches_a_bad_plan() -> None:
    plan = plan_greedy(synthetic_graph())
    bad = replace(plan, offsets={**plan.offsets, "b": plan.offsets["a"] + 32})  # b inside a
    assert [set(p) for p in find_overlaps(bad)] == [{"a", "b"}]
    # sharing bytes is fine when lifetimes are disjoint: d [3,3] and b [1,2]
    ok = replace(plan, offsets={**plan.offsets, "d": plan.offsets["b"]})
    assert find_overlaps(ok) == []


def test_planner_on_and_off_agree_on_the_tiny_graph() -> None:
    x = np.linspace(-3, 3, 16, dtype=np.float32).reshape(1, 1, 4, 4)
    off = Executor(tiny_graph(), planner="naive").run({"f": x})["y"]
    on = Executor(tiny_graph(), planner="greedy").run({"f": x})["y"]
    assert on.tobytes() == off.tobytes()


def test_every_activation_is_a_view_into_the_one_arena() -> None:
    ex = Executor(tiny_graph(), planner="greedy")
    assert ex.plan is not None and ex._arena is not None
    assert ex._arena.dtype == np.int8 and ex._arena.nbytes == ex.plan.arena_bytes
    for name in ex.plan.offsets:
        assert np.shares_memory(ex._view(name), ex._arena)


def test_stale_arena_contents_do_not_leak_into_results() -> None:
    """Fill the arena with garbage before each run: every kernel must overwrite its whole output."""
    x = np.linspace(-3, 3, 16, dtype=np.float32).reshape(1, 1, 4, 4)
    ref = Executor(tiny_graph()).run({"f": x})["y"]
    ex = Executor(tiny_graph(), planner="greedy")
    assert ex._arena is not None
    for fill in (0x55, -1, 127):
        ex._arena[:] = fill
        assert ex.run({"f": x})["y"].tobytes() == ref.tobytes()


def test_dump_dir_with_the_arena_keeps_every_tensor_as_produced(tmp_path: Path) -> None:
    x = np.linspace(-3, 3, 16, dtype=np.float32).reshape(1, 1, 4, 4)
    Executor(tiny_graph(), dump_dir=tmp_path / "naive").run({"f": x})
    Executor(tiny_graph(), planner="greedy", dump_dir=tmp_path / "greedy").run({"f": x})
    a = json.loads((tmp_path / "naive" / "index.json").read_text())
    b = json.loads((tmp_path / "greedy" / "index.json").read_text())
    assert a == b
    for name, fname in a.items():
        left = np.load(tmp_path / "naive" / fname)
        assert left.tobytes() == np.load(tmp_path / "greedy" / fname).tobytes(), name


def test_unknown_planner_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown planner"):
        Executor(tiny_graph(), planner="best")  # type: ignore[arg-type]


# ---------------------------------------------------------------- the real models

needs_cifar = pytest.mark.skipif(not CIFAR.exists(), reason="models/ is generated by tools")
needs_kws = pytest.mark.skipif(
    not (KWS.exists() and KWS_FEATURES.exists()), reason="KWS model comes from Project 2"
)
MODELS = [pytest.param(CIFAR, marks=needs_cifar), pytest.param(KWS, marks=needs_kws)]


@pytest.mark.parametrize("requant", ["float", "fixed"])
@pytest.mark.parametrize("path", MODELS)
def test_real_models_identical_with_and_without_the_planner(path: Path, requant: str) -> None:
    g = load_model(path)
    in_name = g.inputs[0]
    rng = np.random.default_rng(5)
    base = Executor(g, requant=requant)  # type: ignore[arg-type]
    arena = Executor(g, requant=requant, planner="greedy")  # type: ignore[arg-type]
    assert arena._arena is not None
    for _ in range(4):
        x = rng.standard_normal(g.tensors[in_name].shape).astype(np.float32)
        arena._arena[:] = rng.integers(-128, 128, arena._arena.shape, dtype=np.int8)  # garbage
        expected = base.run({in_name: x})
        got = arena.run({in_name: x})
        assert {k: v.tobytes() for k, v in got.items()} == {
            k: v.tobytes() for k, v in expected.items()
        }


@pytest.mark.parametrize("path", MODELS)
def test_real_models_plan_has_no_overlap_and_saves_memory(path: Path) -> None:
    g = load_model(path)
    plan = plan_greedy(g)
    assert find_overlaps(plan) == []
    assert plan.lower_bound_bytes <= plan.arena_bytes < plan.naive_bytes
    assert weights_bytes(g) > 0 and im2col_scratch_bytes(g) > 0


@needs_cifar
def test_cli_memreport_and_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["memreport", "--model", str(CIFAR)])
    report = json.loads(capsys.readouterr().out)
    assert set(report["activation_kib"]) == {"naive", "planned", "lower_bound"}
    x = np.random.default_rng(1).standard_normal((1, 3, 32, 32)).astype(np.float32)
    np.save(tmp_path / "x.npy", x)
    for planner in ("naive", "greedy"):
        cli.main(
            [
                "run",
                "--model", str(CIFAR),
                "--input", str(tmp_path / "x.npy"),
                "--output", str(tmp_path / f"{planner}.npy"),
                "--planner", planner,
                "--dump-all", str(tmp_path / f"d_{planner}"),
            ]
        )  # fmt: skip
    assert np.load(tmp_path / "naive.npy").tobytes() == np.load(tmp_path / "greedy.npy").tobytes()
    assert (tmp_path / "d_greedy" / "index.json").exists()
