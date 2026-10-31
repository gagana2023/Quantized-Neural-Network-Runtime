"""Environment/provenance record attached to every results file."""

from __future__ import annotations

import datetime
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["numpy", "onnx", "onnxruntime", "torch", "torchvision", "PyYAML"]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cpu_model() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True,
            )
            return out.strip().splitlines()[0].strip()
        except (OSError, subprocess.CalledProcessError, IndexError):
            pass
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def collect(command: str | None = None) -> dict[str, Any]:
    """Date, git commit, CPU, OS, Python, package versions and the exact command."""
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    dirty = bool(_git("status", "--porcelain"))
    return {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD") + ("+dirty" if dirty else ""),
        "cpu": _cpu_model(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": versions,
        "command": command or " ".join([Path(sys.executable).name, *sys.argv]),
    }


def write_env_json(path: Path | None = None, command: str | None = None) -> Path:
    path = path or ROOT / "results" / "env.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collect(command), indent=2) + "\n")
    return path


if __name__ == "__main__":
    print(write_env_json(command="python -m tools.envinfo"))
