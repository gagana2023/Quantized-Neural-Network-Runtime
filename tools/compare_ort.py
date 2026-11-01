"""Full-model parity: int8rt against ONNX Runtime, tensor by tensor (spec M3).

For each input, ORT is run with every node output added to the graph outputs, and int8rt is run
with ``dump_dir`` so every intermediate tensor is written as ``.npy``. The two are compared as
bytes, in execution order; the first tensor that differs is reported with its node, the element
index and both values.

Also measured here (three kinds of number, kept apart in the JSON):

* numerical correctness: float mode (Mode A) against ORT, per tensor, per input;
* Mode B (fixed-point requantisation) deviation from Mode A, both *isolated* (each node fed Mode
  A's inputs, so one node's requantiser is all that differs) and *cumulative* (Mode B free-running);
* end-to-end test accuracy of ORT (FP32 and INT8) and int8rt (both modes) on the full test set.

Usage: python -m tools.compare_ort --model cifar|kws [--inputs 100] [--no-accuracy]
Writes results/parity_<model>.json and results/accuracy_<model>.json.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
from onnx import helper

from int8rt.executor import Executor
from int8rt.graph import Graph, Node
from int8rt.importer import load_model
from int8rt.ops import op_def
from int8rt.requant import RequantMode
from tools import envinfo

ROOT = Path(__file__).resolve().parent.parent
Array = npt.NDArray[Any]


# ---------------------------------------------------------------- models and data


@dataclass(frozen=True)
class ModelSpec:
    name: str
    int8_path: Path
    fp32_path: Path
    input_name: str
    load_test_set: Callable[[], tuple[Array, Array]]  # -> (float32 (N, ...), int64 (N,))


def _cifar_test_set() -> tuple[Array, Array]:
    from torch.utils.data import DataLoader

    from tools.train_cifar_cnn import load_cifar

    dl = DataLoader(load_cifar(ROOT / "data", train=False), 1000)
    images = np.concatenate([x.numpy() for x, _ in dl]).astype(np.float32)
    labels = np.concatenate([y.numpy() for _, y in dl]).astype(np.int64)
    return images, labels


def _kws_dir() -> Path:
    return ROOT / "models" / "kws" / "kws"


def _kws_test_set() -> tuple[Array, Array]:
    d = np.load(_kws_dir() / "kws_test_features.npz")
    return d["x"].astype(np.float32), d["y"].astype(np.int64)


MODELS: dict[str, ModelSpec] = {
    "cifar": ModelSpec(
        "cifar",
        ROOT / "models" / "cifar" / "cifar_int8.onnx",
        ROOT / "models" / "cifar" / "cifar_fp32.onnx",
        "input",
        _cifar_test_set,
    ),
    "kws": ModelSpec(
        "kws",
        _kws_dir() / "dscnn_int8.onnx",
        _kws_dir() / "dscnn_fp32.onnx",
        "mfcc",
        _kws_test_set,
    ),
}


# ---------------------------------------------------------------- ORT with every intermediate


def ort_session(path: Path | str | bytes) -> ort.InferenceSession:
    """Graph optimisations off: the reference must run exactly the nodes in the file."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def tapped_session(path: Path) -> tuple[ort.InferenceSession, list[str]]:
    """An ORT session whose outputs are the graph outputs plus every node output.

    Returns the session and the names of the tapped (extra) tensors.
    """
    model = onnx.load(str(path))
    existing = {o.name for o in model.graph.output}
    taps = [o for n in model.graph.node for o in n.output if o and o not in existing]
    model.graph.output.extend(helper.make_empty_tensor_value_info(t) for t in taps)
    return ort_session(model.SerializeToString()), taps


def ort_all_tensors(sess: ort.InferenceSession, feed: Mapping[str, Array]) -> dict[str, Array]:
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(None, dict(feed)), strict=True))


# ---------------------------------------------------------------- comparing tensors


@dataclass(frozen=True)
class Divergence:
    node: str
    op: str
    tensor: str
    index: tuple[int, ...]
    ref_value: float | int
    got_value: float | int

    def as_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "op": self.op,
            "tensor": self.tensor,
            "element_index": list(self.index),
            "ort_value": self.ref_value,
            "int8rt_value": self.got_value,
        }


def _bits(a: Array) -> Array:
    """Reinterpret as same-width unsigned ints so floats compare by bit pattern (NaN-safe)."""
    return np.ascontiguousarray(a).view(f"u{a.dtype.itemsize}")


def first_diff_index(ref: Array, got: Array) -> tuple[int, ...] | None:
    """Index of the first element (C order) whose bytes differ, or None if identical."""
    if ref.shape != got.shape or ref.dtype != got.dtype:
        raise ValueError(f"shape/dtype mismatch: {ref.dtype}{ref.shape} vs {got.dtype}{got.shape}")
    bad = np.flatnonzero(_bits(ref).reshape(-1) != _bits(got).reshape(-1))
    if bad.size == 0:
        return None
    return tuple(int(i) for i in np.unravel_index(int(bad[0]), ref.shape))


def compare_in_order(
    nodes: Sequence[Node], ref: Mapping[str, Array], got: Mapping[str, Array]
) -> tuple[list[bool], Divergence | None]:
    """Compare the first output of each node, in execution order.

    Returns one exact-match flag per node and the first divergence (None if all match). Every
    node is compared even after a divergence, so the flags show whether a later node recovers.
    """
    flags: list[bool] = []
    first: Divergence | None = None
    for node in nodes:
        t = node.outputs[0]
        idx = first_diff_index(ref[t], got[t])
        flags.append(idx is None)
        if idx is not None and first is None:
            first = Divergence(
                node.name, node.op_type, t, idx, ref[t][idx].item(), got[t][idx].item()
            )
    return flags, first


# ---------------------------------------------------------------- Mode B deviation statistics


class DeviationStats:
    """Element differences (candidate minus reference) accumulated over many inputs."""

    def __init__(self) -> None:
        self.elements = 0
        self.differing = 0
        self.max_abs = 0.0
        self.hist: Counter[int] = Counter()  # integer tensors only: diff in LSBs -> count
        self.is_float = False

    def add(self, ref: Array, got: Array) -> None:
        self.elements += ref.size
        if ref.dtype.kind == "f":
            self.is_float = True
            d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
            self.differing += int(np.count_nonzero(d))
            self.max_abs = max(self.max_abs, float(d.max(initial=0.0)))
            return
        d32 = got.astype(np.int32) - ref.astype(np.int32)
        self.differing += int(np.count_nonzero(d32))
        self.max_abs = max(self.max_abs, float(np.abs(d32).max(initial=0)))
        vals, counts = np.unique(d32, return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist(), strict=True):
            if v != 0:
                self.hist[int(v)] += c

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "elements": self.elements,
            "differing_elements": self.differing,
            "differing_percent": 100.0 * self.differing / self.elements if self.elements else 0.0,
            "max_abs_difference": self.max_abs,
            "unit": "float value" if self.is_float else "LSB",
        }
        if not self.is_float:
            out["difference_histogram"] = {str(k): v for k, v in sorted(self.hist.items())}
        return out


def load_dump(dump_dir: Path) -> dict[str, Array]:
    index = json.loads((dump_dir / "index.json").read_text())
    return {name: np.load(dump_dir / fname) for name, fname in index.items()}


def isolated_runners(graph: Graph, mode: RequantMode) -> list[tuple[Node, list[str], Any]]:
    """Per node: (node, names of its dynamic inputs, runner) for the given requantisation mode."""
    return [
        (
            n,
            [t for t in n.inputs if t and not graph.tensors[t].is_initializer],
            op_def(n).prepare(n, graph, mode),
        )
        for n in graph.nodes
    ]


# ---------------------------------------------------------------- the parity run


@dataclass
class ParityResult:
    report: dict[str, Any]
    all_exact: bool


def run_parity(spec: ModelSpec, images: Array, n_inputs: int) -> ParityResult:
    graph = load_model(spec.int8_path)
    sess, taps = tapped_session(spec.int8_path)
    nodes = graph.nodes
    fixed_nodes = isolated_runners(graph, "fixed")

    ex_float = Executor(graph, "float")
    ex_fixed = Executor(graph, "fixed")

    exact_per_node = [0] * len(nodes)
    first_mismatch: dict[str, Any] | None = None
    inputs_with_mismatch = 0
    isolated = [DeviationStats() for _ in nodes]
    cumulative = [DeviationStats() for _ in nodes]
    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp:
        dir_a, dir_b = Path(tmp) / "float", Path(tmp) / "fixed"
        ex_float.dump_dir, ex_fixed.dump_dir = dir_a, dir_b
        for i in range(n_inputs):
            feed = {spec.input_name: images[i : i + 1]}
            ref = ort_all_tensors(sess, feed)
            ex_float.run(feed)
            a_vals = load_dump(dir_a)
            flags, div = compare_in_order(nodes, ref, a_vals)
            for k, ok in enumerate(flags):
                exact_per_node[k] += int(ok)
            if div is not None:
                inputs_with_mismatch += 1
                if first_mismatch is None:
                    first_mismatch = {"input_index": i, **div.as_dict()}

            # Mode B, cumulative: the whole network free-running in fixed-point.
            ex_fixed.run(feed)
            b_vals = load_dump(dir_b)
            for k, node in enumerate(nodes):
                t = node.outputs[0]
                cumulative[k].add(a_vals[t], b_vals[t])
            # Mode B, isolated: each node sees Mode A's (= ORT's) inputs.
            for k, (node, dyn, runner) in enumerate(fixed_nodes):
                t = node.outputs[0]
                out = np.empty_like(a_vals[t])
                runner([a_vals[d] for d in dyn], out)
                isolated[k].add(a_vals[t], out)
            if (i + 1) % 25 == 0 or i + 1 == n_inputs:
                print(f"  parity {spec.name}: {i + 1}/{n_inputs} inputs", file=sys.stderr)

    layers = [
        {
            "node": n.name,
            "op": n.op_type,
            "tensor": n.outputs[0],
            "shape": list(graph.tensors[n.outputs[0]].shape),
            "dtype": str(graph.tensors[n.outputs[0]].dtype),
            "exact_matches": exact_per_node[k],
            "inputs": n_inputs,
        }
        for k, n in enumerate(nodes)
    ]
    all_exact = all(e == n_inputs for e in exact_per_node)
    report: dict[str, Any] = {
        "what": "int8rt (float requantisation, Mode A) against ONNX Runtime CPU, byte for byte, "
        "on every node output",
        "model": str(spec.int8_path.relative_to(ROOT)),
        "ort_tapped_tensors": len(taps),
        "input_source": f"first {n_inputs} test-set inputs, in dataset order",
        "inputs_tested": n_inputs,
        "layers_compared": len(nodes),
        "tensor_comparisons": len(nodes) * n_inputs,
        "exact_matches": int(sum(exact_per_node)),
        "inputs_with_any_mismatch": inputs_with_mismatch,
        "float_mode_bit_exact_on_every_layer_and_input": all_exact,
        "first_mismatch": first_mismatch,
        "layers": layers,
        "mode_b_vs_mode_a": {
            "note": "Mode A is bit-exact with ORT (above), so these are also deviations from ORT. "
            "'isolated': each node run in fixed-point on Mode A's input tensors, so only that "
            "node's requantiser differs. 'cumulative': the whole network free-running in "
            "fixed-point, so earlier differences propagate. Integer tensors are in LSBs; the "
            "float output (logits) is in value units.",
            "isolated": [
                {"node": n.name, "op": n.op_type, **isolated[k].as_dict()}
                for k, n in enumerate(nodes)
            ],
            "cumulative": [
                {"node": n.name, "op": n.op_type, **cumulative[k].as_dict()}
                for k, n in enumerate(nodes)
            ],
        },
        "wall_seconds_measured_on_host": round(time.perf_counter() - t0, 1),
    }
    return ParityResult(report, all_exact)


# ---------------------------------------------------------------- accuracy on the full test set


def run_accuracy(spec: ModelSpec, images: Array, labels: Array) -> dict[str, Any]:
    graph = load_model(spec.int8_path)
    ex_a, ex_b = Executor(graph, "float"), Executor(graph, "fixed")
    s_fp32, s_int8 = ort_session(spec.fp32_path), ort_session(spec.int8_path)
    out_name = graph.outputs[0]
    n = len(labels)
    correct = {"ort_fp32": 0, "ort_int8": 0, "int8rt_float": 0, "int8rt_fixed": 0}
    float_exact = 0
    flips = 0  # fixed-point prediction differs from float-mode prediction
    fixed_logit_diff = DeviationStats()
    fixed_differing_inputs = 0
    t0 = time.perf_counter()
    for i in range(n):
        feed = {spec.input_name: images[i : i + 1]}
        y = int(labels[i])
        r_fp32 = s_fp32.run(None, feed)[0]
        r_int8 = s_int8.run(None, feed)[0]
        r_a = ex_a.run(feed)[out_name]
        r_b = ex_b.run(feed)[out_name]
        correct["ort_fp32"] += int(np.argmax(r_fp32) == y)
        correct["ort_int8"] += int(np.argmax(r_int8) == y)
        correct["int8rt_float"] += int(np.argmax(r_a) == y)
        correct["int8rt_fixed"] += int(np.argmax(r_b) == y)
        float_exact += int(r_a.tobytes() == r_int8.tobytes())
        flips += int(np.argmax(r_b) != np.argmax(r_a))
        fixed_differing_inputs += int(r_b.tobytes() != r_a.tobytes())
        fixed_logit_diff.add(r_a, r_b)
        if (i + 1) % 1000 == 0 or i + 1 == n:
            print(f"  accuracy {spec.name}: {i + 1}/{n} images", file=sys.stderr)
    return {
        "test_images": n,
        "ort_fp32_accuracy": correct["ort_fp32"] / n,
        "ort_int8_accuracy": correct["ort_int8"] / n,
        "int8rt_float_accuracy": correct["int8rt_float"] / n,
        "int8rt_fixed_accuracy": correct["int8rt_fixed"] / n,
        "correct_counts": correct,
        "float_mode_logits_bit_exact_with_ort_int8": {"inputs": float_exact, "of": n},
        "float_mode_accuracy_equals_ort_int8": correct["int8rt_float"] == correct["ort_int8"],
        "fixed_mode_vs_float_mode": {
            "inputs_whose_logits_differ": fixed_differing_inputs,
            "inputs_whose_predicted_class_differs": flips,
            "logit_deviation": fixed_logit_diff.as_dict(),
            "accuracy_change_points": 100.0
            * (correct["int8rt_fixed"] - correct["int8rt_float"])
            / n,
        },
        "wall_seconds_measured_on_host": round(time.perf_counter() - t0, 1),
    }


def kws_reference_check(images: Array, spec: ModelSpec) -> dict[str, Any]:
    """Compare this ORT run with the logits Project 2 stored next to the features."""
    ref = np.load(_kws_dir() / "kws_test_features.npz")["logits_int8"]
    sess = ort_session(spec.int8_path)
    same = sum(
        sess.run(None, {spec.input_name: images[i : i + 1]})[0].tobytes()
        == ref[i : i + 1].tobytes()
        for i in range(len(images))
    )
    return {"inputs": len(images), "identical_to_project2_stored_logits_int8": same}


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--inputs", type=int, default=100, help="inputs for per-layer parity")
    ap.add_argument("--no-accuracy", action="store_true", help="skip the full test-set pass")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    spec = MODELS[args.model]
    images, labels = spec.load_test_set()
    n_inputs = min(args.inputs, len(labels))
    env = envinfo.collect("python -m tools.compare_ort " + shlex.join(sys.argv[1:]))

    print(f"[{spec.name}] per-layer parity on {n_inputs} inputs", file=sys.stderr)
    parity = run_parity(spec, images, n_inputs)
    args.out_dir.mkdir(exist_ok=True)
    p_path = args.out_dir / f"parity_{spec.name}.json"
    p_path.write_text(json.dumps({"env": env, **parity.report}, indent=2) + "\n")
    print(f"wrote {p_path}")

    if not args.no_accuracy:
        print(f"[{spec.name}] accuracy on {len(labels)} test inputs", file=sys.stderr)
        acc = run_accuracy(spec, images, labels)
        a_path = args.out_dir / f"accuracy_{spec.name}.json"
        old = json.loads(a_path.read_text()) if a_path.exists() else {}
        old.pop("int8rt_accuracy", None)  # placeholder written by export_quantize at M0
        old.pop("ort_fp32_accuracy", None)
        old.pop("ort_int8_accuracy", None)
        old.pop("test_images", None)
        extra: dict[str, Any] = {}
        if spec.name == "kws":
            extra["project2_reference_logits"] = kws_reference_check(images, spec)
        merged = {
            "env_m0_quantisation": old.pop("env", None),
            "env": env,
            "what": "Test accuracy: ORT CPU (graph optimisations disabled) and int8rt, "
            "float and fixed-point requantisation, all measured on the same inputs",
            **old,
            **acc,
            **extra,
        }
        merged = {k: v for k, v in merged.items() if v is not None}
        a_path.write_text(json.dumps(merged, indent=2) + "\n")
        print(f"wrote {a_path}")

    verdict = "bit-exact" if parity.all_exact else "MISMATCH (see first_mismatch)"
    print(f"{spec.name}: float mode vs ORT on {n_inputs} inputs: {verdict}")
    if not parity.all_exact:
        sys.exit(1)


if __name__ == "__main__":
    main()
