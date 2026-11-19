# -*- mode: python ; coding: utf-8 -*-
# Standard PyInstaller torch/numpy/PyQt6 hooks collect required native DLLs.
from pathlib import Path

root = Path(SPECPATH)

a = Analysis(
    [str(root / "main.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[
        (str(root / "assets" / "fnn_fault_model.pt"), "assets"),
        (str(root / "assets" / "scaler.npz"), "assets"),
        (str(root / "assets" / "sample.csv"), "assets"),
        (str(root / "LICENSE_THIRD_PARTY.txt"), "."),
    ],
    hiddenimports=[
        "matplotlib.backends.backend_qtagg",
        "matplotlib.backends.backend_agg",
        "PyQt6.sip",
    ],
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["QtAgg"]}},
    runtime_hooks=[],
    excludes=[
        "PyQt5", "PySide2", "PySide6", "tkinter",
        "tensorflow", "keras", "jax", "jaxlib",
        "notebook", "jupyter", "IPython", "pytest",
        "pandas", "scipy", "sklearn",
        "torchvision", "torchaudio", "tensorboard",
    ],
    noarchive=False,
)
# Qt 6 uses Windows 10/11's system ICU. Other applications may put an
# incompatible ICU (or old UCRT/API shim DLLs) on PATH. Never redistribute
# those OS-provided DLLs from such unrelated SDKs, even with a cached analysis.
system_dlls = {"icuuc.dll", "icuin.dll", "icudt.dll", "icudt78.dll", "ucrtbase.dll"}
a.binaries = [
    item for item in a.binaries
    if Path(item[0]).name.lower() not in system_dlls
    and not Path(item[0]).name.lower().startswith(("api-ms-win-", "ext-ms-win-"))
]
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="SEED_FaultDiagnosis",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
