"""Build the README results tables from the committed files in results/.

Every number in the tables is read from a results JSON; none is typed by hand. The tables are
written to ``results/readme_tables.md`` (with a provenance footer) and spliced into README.md
between the ``<!-- RESULTS:BEGIN -->`` and ``<!-- RESULTS:END -->`` markers.

Usage: python -m tools.make_readme_tables          # rewrite both files
       python -m tools.make_readme_tables --check  # exit 1 if README.md is out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
README = ROOT / "README.md"
BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
MODELS = {"cifar": "CIFAR-10 CNN", "kws": "KWS DS-CNN"}
SCHEMES = {"weight_stationary": "weight stationary", "output_stationary": "output stationary"}


def _load(name: str) -> dict[str, Any]:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"missing {path.relative_to(ROOT)}; run the tool that writes it first")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _table(header: list[str], rows: list[list[str]], right: tuple[int, ...] = ()) -> str:
    sep = ["---:" if i in right else "---" for i in range(len(header))]
    lines = [header, sep, *rows]
    return "\n".join("| " + " | ".join(r) + " |" for r in lines)


def _pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}%"


def op_parity_table() -> str:
    d = _load("op_parity.json")
    rows = [[op, str(v["cases"]), str(v["matched"])] for op, v in d["per_op"].items()]
    rows.append(["**all**", str(d["cases"]), str(d["matched"])])
    head = (
        f"Single-op parity, float mode, seed {d['seed']}: {d['matched']} of {d['cases']} cases "
        f"bit-exact with ORT (`results/op_parity.json`). The `*Sweep` rows feed arbitrary INT32 "
        "accumulators or float values through ORT's requantiser / quantiser."
    )
    return head + "\n\n" + _table(["op", "cases", "bit-exact"], rows, right=(1, 2))


def parity_table() -> str:
    rows = []
    for key, label in MODELS.items():
        p, a = _load(f"parity_{key}.json"), _load(f"accuracy_{key}.json")
        exact = a["float_mode_logits_bit_exact_with_ort_int8"]
        rows.append(
            [
                label,
                str(p["inputs_tested"]),
                str(p["layers_compared"]),
                f"{p['exact_matches']} / {p['tensor_comparisons']}",
                f"{exact['inputs']} / {exact['of']}",
            ]
        )
    return _table(
        [
            "model",
            "inputs (per layer)",
            "nodes compared",
            "node outputs bit-exact",
            "final logits bit-exact (all test inputs)",
        ],
        rows,
        right=(1, 2),
    )


def accuracy_table() -> str:
    rows = []
    for key, label in MODELS.items():
        a = _load(f"accuracy_{key}.json")
        rows.append(
            [
                label,
                str(a["test_images"]),
                _pct(a["ort_fp32_accuracy"]),
                _pct(a["ort_int8_accuracy"]),
                _pct(a["int8rt_float_accuracy"]),
                _pct(a["int8rt_fixed_accuracy"]),
            ]
        )
    return _table(
        [
            "model",
            "test inputs",
            "FP32 (ORT)",
            "INT8 (ORT)",
            "INT8 int8rt, float requant",
            "INT8 int8rt, fixed-point requant",
        ],
        rows,
        right=(1, 2, 3, 4, 5),
    )


def mode_b_table() -> str:
    rows = []
    for key, label in MODELS.items():
        f = _load(f"accuracy_{key}.json")["fixed_mode_vs_float_mode"]
        dev = f["logit_deviation"]
        n = _load(f"accuracy_{key}.json")["test_images"]
        cum = _load(f"parity_{key}.json")["mode_b_vs_mode_a"]["cumulative"]
        lsb = max(
            (c["max_abs_difference"] for c in cum if c["unit"] == "LSB"),
            default=0.0,
        )
        rows.append(
            [
                label,
                f"{lsb:g}",
                f"{dev['differing_percent']:.1f}%",
                f"{dev['max_abs_difference']:.2f}",
                f"{f['inputs_whose_predicted_class_differs']} / {n}",
                f"{f['accuracy_change_points']:+.2f}",
            ]
        )
    return _table(
        [
            "model",
            "max node-output deviation, first 100 inputs (LSB)",
            "final logits that differ",
            "max logit difference",
            "predicted class changes",
            "accuracy change (points)",
        ],
        rows,
        right=(1, 2, 3, 4, 5),
    )


def memory_table() -> str:
    rows = []
    for key, label in MODELS.items():
        m = _load(f"memory_{key}.json")
        k = m["activation_kib"]
        rows.append(
            [
                label,
                f"{k['naive']:.2f}",
                f"{k['planned']:.2f}",
                f"{k['lower_bound']:.2f}",
                f"{m['reduction_naive_to_planned_percent']:.1f}%",
                f"{m['weights_kib']:.2f}",
            ]
        )
    return _table(
        [
            "model",
            "naive KiB",
            "planned KiB",
            "lower bound KiB",
            "naive to planned",
            "weights KiB (separate)",
        ],
        rows,
        right=(1, 2, 3, 4, 5),
    )


def cost_table() -> str:
    rows = []
    for key, label in MODELS.items():
        d = _load(f"cost_base_{key}.json")
        for scheme, sname in SCHEMES.items():
            t = d["totals"][scheme]
            rows.append(
                [
                    label,
                    sname,
                    f"{t['compute_cycles']:,}",
                    f"{t['total_cycles_no_overlap']:,}",
                    f"{t['total_cycles_double_buffered']:,}",
                    f"{t['dma_stall_reduction_percent']:.1f}%",
                    _pct(t["utilisation_double_buffered"], 1),
                    f"{t['compute_bound_layers']} / {t['transfer_bound_layers']}",
                ]
            )
    return _table(
        [
            "model",
            "loop order",
            "compute cycles",
            "cycles, no overlap",
            "cycles, double buffered",
            "modelled DMA stalls cut",
            "utilisation (DB)",
            "compute- / transfer-bound layers",
        ],
        rows,
        right=(2, 3, 4, 5, 6),
    )


def provenance() -> str:
    lines = []
    for name in sorted(p.name for p in RESULTS.glob("*.json")):
        d = json.loads((RESULTS / name).read_text(encoding="utf-8"))
        env = d.get("env")
        if isinstance(env, dict):
            lines.append(
                f"- `results/{name}`: {env['date'][:10]}, commit `{env['git_commit'][:8]}"
                f"{'+dirty' if str(env['git_commit']).endswith('+dirty') else ''}`, "
                f"`{env['command'].strip()}`"
            )
    return "\n".join(lines)


def build() -> str:
    env = _load("env.json")
    parts = [
        ("Numerical correctness: single operators (vs ONNX Runtime)", op_parity_table()),
        ("Numerical correctness: whole models (vs ONNX Runtime)", parity_table()),
        ("Test accuracy", accuracy_table()),
        (
            "Numerical: fixed-point (Mode B) deviation from float (Mode A, bit-exact with ORT)",
            mode_b_table(),
        ),
        ("Static activation memory (planner; not a host measurement)", memory_table()),
        (
            "MODELLED accelerator cost, base config (analytical; not measured, not a real chip)",
            cost_table(),
        ),
    ]
    out = []
    for i, (title, body) in enumerate(parts, 1):
        out.append(f"**{i}. {title}**\n\n{body}\n")
    out.append(
        f"Generated by `python -m tools.make_readme_tables` from `results/*.json`. Reference "
        f"versions: onnxruntime {env['packages']['onnxruntime']}, numpy "
        f"{env['packages']['numpy']}, onnx {env['packages']['onnx']}. Each file below records its "
        "own date, commit, CPU, OS, package versions and command:\n\n" + provenance() + "\n"
    )
    return "\n".join(out)


def splice(readme: str, tables: str) -> str:
    try:
        head, rest = readme.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
    except ValueError:
        raise SystemExit(f"README.md needs the markers {BEGIN} and {END}") from None
    return f"{head}{BEGIN}\n\n{tables}\n{END}{tail}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if README.md would change")
    args = ap.parse_args()
    tables = build()
    new = splice(README.read_text(encoding="utf-8"), tables)
    if args.check:
        if new != README.read_text(encoding="utf-8"):
            print("README.md results tables are out of date; run tools.make_readme_tables")
            sys.exit(1)
        return
    (RESULTS / "readme_tables.md").write_text(tables, encoding="utf-8")
    README.write_text(new, encoding="utf-8")


if __name__ == "__main__":
    main()
