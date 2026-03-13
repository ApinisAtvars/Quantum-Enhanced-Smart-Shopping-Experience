from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "QACF" / "data" / "processed_32m_ubcf"
OUT_PATH = PROCESSED_DIR / "reproducibility_snapshot.json"


TRACKED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "umap",
    "qiskit",
    "pennylane",
    "torch",
    "dimod",
    "dwave",
]

TRACKED_FILES = [
    "ubcf_metrics.csv",
    "ubcf_tuning.csv",
    "hybrid_ubcf_stress.csv",
    "classical_only_stress.csv",
    "hybrid_vs_classical_uplift.csv",
    "research_addendum_ubcf.csv",
    "runtime_complexity_ubcf.csv",
    "privacy_log_ubcf.csv",
    "hybrid_results_summary.json",
    "quantum_metrics.json",
    "quantum_metrics.csv",
]


def _pkg_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path.relative_to(ROOT)),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": _sha256(path),
    }


def build_snapshot() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    packages = {name: _pkg_version(name) for name in TRACKED_PACKAGES}

    files = []
    for fname in TRACKED_FILES:
        path = PROCESSED_DIR / fname
        if path.exists():
            files.append(_file_record(path))

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(ROOT),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "hostname": platform.node(),
        },
        "reproducibility": {
            "default_seed": 42,
            "notes": [
                "Notebook seeds should be fixed to 42 unless explicitly changed per trial.",
                "Use this snapshot together with notebook outputs for auditability.",
            ],
        },
        "package_versions": packages,
        "artifacts": files,
    }


def main() -> None:
    snapshot = build_snapshot()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Saved reproducibility snapshot: {OUT_PATH}")


if __name__ == "__main__":
    main()
