"""Build the bundled Windows application with the current Python environment.

    python build_exe.py
    python build_exe.py --output-dir C:\\path\\to\\output

No installation is performed here. See README_KO.md for CMD setup.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent
REQUIRED = (
    "main.py",
    "gui.py",
    "diagnosis_core.py",
    "assets/fnn_fault_model.pt",
    "assets/scaler.npz",
    "assets/sample.csv",
    "LICENSE_THIRD_PARTY.txt",
    "SEED_FaultDiagnosis.spec",
)


def check_environment() -> None:
    if sys.platform != "win32" or platform.architecture()[0] != "64bit":
        raise RuntimeError("Build this Windows executable with 64-bit Python on Windows.")
    if sys.version_info < (3, 12):
        raise RuntimeError("The pinned NumPy needs Python 3.12 or newer; this release was built with 3.14.")
    # Foreign SDKs on PATH can supply incompatible ICU/UCRT DLLs to the
    # dependency scanner. Python package hooks locate their own native DLLs;
    # Windows system DLLs must resolve through System32, not another app's SDK.
    python_home = Path(sys.executable).parent
    windows_home = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    os.environ["PATH"] = os.pathsep.join(map(str, (
        python_home, python_home / "Scripts", windows_home / "System32", windows_home,
    )))
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError("Required source/assets are missing: " + ", ".join(missing))

    print("Python:", sys.executable, sys.version.split()[0], flush=True)
    for name in ("torch", "numpy", "PyQt6", "matplotlib", "pyinstaller"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"{name} is missing. Run "
                "python -m pip install -r requirements.txt"
            ) from exc
        print(f"  {name}: {version}", flush=True)

    os.environ["QT_API"] = "pyqt6"
    import torch
    from PyQt6 import QtCore  # noqa: F401
    import numpy  # noqa: F401
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: F401

    if torch.version.cuda is not None:
        raise RuntimeError(
            "The build requires CPU-only PyTorch to keep the executable portable. "
            "Install requirements.txt in a fresh virtual environment (see README_KO.md)."
        )
    # Catch accidental syntax errors before PyInstaller starts its slower analysis.
    for filename in ("main.py", "gui.py", "diagnosis_core.py"):
        compile((ROOT / filename).read_text(encoding="utf-8-sig"), filename, "exec")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--incremental", action="store_true", help="Reuse PyInstaller caches after source edits")
    args = parser.parse_args()
    try:
        check_environment()
        from PyInstaller.__main__ import run

        output = args.output_dir.resolve()
        options = [
            str(ROOT / "SEED_FaultDiagnosis.spec"),
            "--noconfirm",
            "--distpath", str(output),
            "--workpath", str(ROOT / "build"),
        ]
        if not args.incremental:
            options.append("--clean")
        run(options)
        exe = output / "SEED_FaultDiagnosis.exe"
        if not exe.is_file():
            raise RuntimeError("PyInstaller finished without the expected executable.")
        print(f"\nBuild complete: {exe}\nSize: {exe.stat().st_size / 1024**2:.1f} MiB")
        return 0
    except Exception as exc:
        print(f"\nBUILD FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
