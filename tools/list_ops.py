"""Print every op type and domain in an ONNX model (with node counts).

Usage: python -m tools.list_ops model.onnx [-o results/ops_x.txt]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import onnx


def count_ops(model: onnx.ModelProto) -> Counter[tuple[str, str]]:
    """Count nodes by (domain, op_type). The default ONNX domain is shown as 'ai.onnx'."""
    return Counter((n.domain or "ai.onnx", n.op_type) for n in model.graph.node)


def format_ops(model: onnx.ModelProto, name: str) -> str:
    opsets = ", ".join(f"{o.domain or 'ai.onnx'}={o.version}" for o in model.opset_import)
    lines = [f"model: {name}", f"opsets: {opsets}", f"nodes: {len(model.graph.node)}", ""]
    lines.append(f"{'domain':<14}{'op_type':<26}count")
    for (domain, op), n in sorted(count_ops(model).items()):
        lines.append(f"{domain:<14}{op:<26}{n}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="also write the listing to this file")
    args = ap.parse_args()
    text = format_ops(onnx.load(str(args.model)), args.model.name)
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
