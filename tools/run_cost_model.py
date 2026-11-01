"""Run the accelerator cost model on a model: per-layer table, sweeps, CSVs, plots, summary JSON.

Writes, for config ``<cfg>`` and model ``<m>``, into ``results/``:
  cost_<cfg>_<m>.csv            per-layer rows, both loop orders (modelled)
  cost_<cfg>_<m>.json           environment, config, whole-network totals and sweep totals
  cost_<cfg>_<m>_sweeps.csv     per-layer rows at every sweep point
  cost_<cfg>_<m>_layers.md      the per-layer table (compute- vs transfer-bound) for docs/RESULTS.md
  cost_<cfg>_<m>_summary.md     whole-network totals and sweep tables for docs/RESULTS.md
  cost_<cfg>_<m>.png            per-layer compute vs transfer cycles
  cost_<cfg>_<m>_sweeps.png     whole-network cycles vs SRAM size, bandwidth and array shape

All numbers are MODELLED cost of a generic configurable accelerator. They are not measured, not host
CPU time, and say nothing about numerical correctness.

Usage: python -m tools.run_cost_model --model cifar|kws [--config costmodel/configs/base.yaml]
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from costmodel.mapping import map_graph  # noqa: E402
from costmodel.model import SCHEMES, AccelConfig, estimate_work, load_config  # noqa: E402
from costmodel.sweep import layer_rows, run_sweeps, totals  # noqa: E402
from int8rt.importer import load_model  # noqa: E402
from tools import envinfo  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATHS = {
    "cifar": ROOT / "models" / "cifar" / "cifar_int8.onnx",
    "kws": ROOT / "models" / "kws" / "kws" / "dscnn_int8.onnx",
}
SHORT = {"weight_stationary": "WS", "output_stationary": "OS"}
# categorical slots 1 and 2 of the dataviz reference palette; dashes and hatching back up colour
BLUE, ORANGE, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
SCHEME_COLOUR = {"weight_stationary": BLUE, "output_stationary": ORANGE}
WHAT = (
    "MODELLED cost of a generic, configurable accelerator (analytical, not cycle-accurate, not any "
    "real chip). Not measured, not host CPU time, not a numerical-correctness result."
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: "" if v is None else v for k, v in r.items()} for r in rows)


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def plot_layers(rows: list[dict[str, Any]], title: str, path: Path) -> None:
    """Compute vs transfer cycles per layer, one panel per loop order (log scale)."""
    fig, axes = plt.subplots(len(SCHEMES), 1, figsize=(11, 7), sharex=True, facecolor=SURFACE)
    for ax, scheme in zip(axes, SCHEMES, strict=True):
        sel = [r for r in rows if r["scheme"] == scheme and r["kind"] != "layout"]
        xs = list(range(len(sel)))
        ax.bar(
            [x - 0.2 for x in xs],
            [r["compute_cycles"] for r in sel],
            0.38,
            color=BLUE,
            label="compute cycles",
        )
        ax.bar(
            [x + 0.2 for x in xs],
            [r["transfer_cycles"] for r in sel],
            0.38,
            color=ORANGE,
            label="transfer cycles",
            hatch="//",
            edgecolor=SURFACE,
        )
        ax.set_yscale("log")
        ax.set_ylabel("cycles (log)", color=MUTED)
        name = scheme.replace("_", " ")
        ax.set_title(f"{SHORT[scheme]}: {name}", loc="left", color=INK, fontsize=10)
        _style(ax)
        top = max(max(r["compute_cycles"], r["transfer_cycles"]) for r in sel)
        ax.set_ylim(0.6, top * 10)  # headroom for the legend
        ax.legend(frameon=False, loc="upper right", ncol=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [f"{r['index']}:{r['op'].replace('QLinear', 'QL')}" for r in sel],
            rotation=60,
            ha="right",
            fontsize=8,
            color=MUTED,
        )
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _is_square(point: str) -> bool:
    r, _, c = point.partition("x")
    return not c or r == c  # non-array sweeps have no 'x' in the label


def plot_sweeps(point_totals: list[dict[str, Any]], title: str, path: Path) -> None:
    names = ["sram", "bandwidth", "array"]
    xlabel = {
        "sram": "SRAM size vs base (weight and activation SRAM together)",
        "bandwidth": "DMA bytes per cycle",
        "array": "MACs per cycle (R x C)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    for ax, name in zip(axes, names, strict=True):
        for scheme in SCHEMES:
            sel = [t for t in point_totals if t["sweep"] == name and t["scheme"] == scheme]
            sel.sort(key=lambda t: t["x"])
            for key, style, label in (
                ("total_cycles_double_buffered", "-", "double buffered"),
                ("total_cycles_no_overlap", "--", "no overlap"),
            ):
                # square arrays form the line; non-square shapes share a MAC count with a square
                # one, so they are drawn as separate, labelled points
                line = [t for t in sel if _is_square(t["point"])]
                kw: dict[str, Any] = {"color": SCHEME_COLOUR[scheme], "markersize": 7}
                ax.plot(
                    [t["x"] for t in line],
                    [t[key] for t in line],
                    style,
                    marker="o",
                    linewidth=2,
                    label=f"{SHORT[scheme]} {label}",
                    **kw,
                )
                for t in sel:
                    if not _is_square(t["point"]):
                        ax.plot(t["x"], t[key], marker="s", linestyle="none", **kw)
                        if key == "total_cycles_double_buffered" and scheme == SCHEMES[0]:
                            dy = 6 if t["point"] == "32x8" else -10
                            ax.annotate(
                                t["point"],
                                (t["x"], t[key]),
                                textcoords="offset points",
                                xytext=(7, dy),
                                fontsize=8,
                                color=MUTED,
                            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel[name], color=MUTED)
        ax.set_ylabel("whole-network cycles (log)", color=MUTED)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _cells(r: dict[str, Any]) -> list[str]:
    tile = "-" if r["tile_m"] is None else f"{r['tile_m']},{r['tile_n']},{r['tile_k']}"
    u = r["util_double_buffered"]
    util = "-" if u is None else f"{100 * u:.1f}%"
    return [tile, str(r["total_cycles_double_buffered"]), util, r["bound"]]


def layers_markdown(rows: list[dict[str, Any]]) -> str:
    head = (
        "| # | op | groups x M,K,N | WS tile (mt,nt,kt) | WS cycles (DB) | WS util | WS bound "
        "| OS tile | OS cycles (DB) | OS util | OS bound |\n"
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |\n"
    )
    by_layer: dict[int, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by_layer.setdefault(r["index"], {})[r["scheme"]] = r
    lines = []
    for i, pair in by_layer.items():
        ws, os_ = pair["weight_stationary"], pair["output_stationary"]
        gemm = "-"
        if ws["gemm_m"] is not None:
            gemm = f"{ws['groups']} x {ws['gemm_m']},{ws['gemm_k']},{ws['gemm_n']}"
        cells = [str(i), ws["op"], gemm, *_cells(ws), *_cells(os_)]
        lines.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def _both(t: dict[str, Any]) -> str:
    return f"{t['total_cycles_double_buffered']} / {t['total_cycles_no_overlap']}"


def summary_markdown(totals_: dict[str, dict[str, Any]], sweep_totals: list[dict[str, Any]]) -> str:
    """Whole-network totals per scheme, then one table per sweep (DB / no-overlap cycles)."""
    out = [
        "| scheme | compute cycles | transfer cycles | no overlap | double buffered "
        "| DMA stalls cut | utilisation (DB) | DRAM KiB | compute-bound / transfer-bound layers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, t in totals_.items():
        cells = [
            name,
            t["compute_cycles"],
            t["transfer_cycles"],
            t["total_cycles_no_overlap"],
            t["total_cycles_double_buffered"],
            f"{t['dma_stall_reduction_percent']}%",
            f"{100 * t['utilisation_double_buffered']:.1f}%",
            f"{t['dram_bytes'] / 1024:.1f}",
            f"{t['compute_bound_layers']} / {t['transfer_bound_layers']}",
        ]
        out.append("| " + " | ".join(str(c) for c in cells) + " |")
    for sweep in ("sram", "bandwidth", "array"):
        out += ["", f"Sweep `{sweep}` (whole-network cycles, double buffered / no overlap):", ""]
        out += [
            "| point | WS | OS | WS DMA stalls cut | OS DMA stalls cut |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        pts: dict[str, dict[str, dict[str, Any]]] = {}
        for t in sweep_totals:
            if t["sweep"] == sweep and t["scheme"] in SCHEMES:
                pts.setdefault(t["point"], {})[t["scheme"]] = t
        for point, by in pts.items():
            ws, os_ = by["weight_stationary"], by["output_stationary"]
            cells = [
                point,
                _both(ws),
                _both(os_),
                f"{ws['dma_stall_reduction_percent']}%",
                f"{os_['dma_stall_reduction_percent']}%",
            ]
            out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def run(model: str, cfg_path: Path, out_dir: Path, command: str) -> dict[str, Any]:
    cfg: AccelConfig = load_config(cfg_path)
    graph = load_model(MODEL_PATHS[model])
    work = map_graph(graph)
    costs = estimate_work(work, cfg)
    rows = layer_rows(costs, cfg)
    sweep_layers, sweep_totals = run_sweeps(work, cfg)
    stem = f"cost_{cfg.name}_{model}"
    out_dir.mkdir(exist_ok=True)
    write_csv(out_dir / f"{stem}.csv", rows)
    write_csv(out_dir / f"{stem}_sweeps.csv", sweep_layers)
    (out_dir / f"{stem}_layers.md").write_text(layers_markdown(rows))
    (out_dir / f"{stem}_summary.md").write_text(summary_markdown(totals(costs, cfg), sweep_totals))
    plot_layers(
        rows, f"{model}: modelled compute vs transfer cycles ({cfg.name})", out_dir / f"{stem}.png"
    )
    plot_sweeps(
        sweep_totals,
        f"{model}: modelled whole-network cycles, one parameter varied at a time",
        out_dir / f"{stem}_sweeps.png",
    )
    summary = {
        "env": envinfo.collect(command),
        "what": WHAT,
        "model": model,
        "model_file": str(MODEL_PATHS[model].relative_to(ROOT)),
        "config": cfg.as_dict(),
        "totals": totals(costs, cfg),
        "sweep_totals": sweep_totals,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(MODEL_PATHS), required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "costmodel" / "configs" / "base.yaml")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    cmd = "python -m tools.run_cost_model " + shlex.join(sys.argv[1:])
    summary = run(args.model, args.config, args.out_dir, cmd)
    for scheme, t in summary["totals"].items():
        print(
            f"{args.model} {scheme}: {t['total_cycles_no_overlap']} cycles no overlap, "
            f"{t['total_cycles_double_buffered']} double buffered "
            f"(DMA stalls -{t['dma_stall_reduction_percent']}%), "
            f"utilisation {100 * t['utilisation_double_buffered']:.1f}%"
        )


if __name__ == "__main__":
    main()
