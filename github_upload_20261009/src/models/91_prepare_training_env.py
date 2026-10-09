from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SITE = ROOT / ".venv" / "Lib" / "site-packages"
TARGET_SITE = ROOT / "数据整理" / ".venv" / "Lib" / "site-packages"
TARGET_PYTHON = ROOT / "数据整理" / ".venv" / "Scripts" / "python.exe"


PACKAGES_TO_COPY = [
    "matplotlib",
    "matplotlib-3.10.9.dist-info",
    "mpl_toolkits",
    "contourpy",
    "contourpy-1.3.3.dist-info",
    "cycler",
    "cycler-0.12.1.dist-info",
    "fontTools",
    "fonttools-4.63.0.dist-info",
    "kiwisolver",
    "kiwisolver-1.5.0.dist-info",
    "PIL",
    "pillow-12.2.0.dist-info",
    "pyparsing",
    "pyparsing-3.3.2.dist-info",
]


def copy_one(name: str, force: bool) -> str:
    src = SOURCE_SITE / name
    dst = TARGET_SITE / name
    if not src.exists():
        return f"MISS source: {src}"
    if dst.exists():
        if not force:
            return f"SKIP exists: {dst}"
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return f"COPY {src.name}"


def test_target_python() -> None:
    code = (
        "import sys; "
        "import torch; "
        "import numpy; "
        "import pandas; "
        "import sklearn; "
        "import matplotlib; "
        "print(sys.executable); "
        "print('torch', torch.__version__, 'no_grad', hasattr(torch, 'no_grad')); "
        "print('matplotlib', matplotlib.__version__); "
        "print('training env ok')"
    )
    subprocess.run([str(TARGET_PYTHON), "-X", "utf8", "-c", code], cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the working training virtual environment.")
    parser.add_argument("--force", action="store_true", help="Overwrite copied matplotlib-related packages.")
    args = parser.parse_args()

    if not SOURCE_SITE.exists():
        raise FileNotFoundError(f"Source site-packages not found: {SOURCE_SITE}")
    if not TARGET_SITE.exists():
        raise FileNotFoundError(f"Target site-packages not found: {TARGET_SITE}")
    if not TARGET_PYTHON.exists():
        raise FileNotFoundError(f"Target Python not found: {TARGET_PYTHON}")

    print("Preparing training environment...")
    print(f"Source site-packages: {SOURCE_SITE}")
    print(f"Target site-packages: {TARGET_SITE}")
    for name in PACKAGES_TO_COPY:
        print(copy_one(name, args.force))

    print("\nTesting target Python imports...")
    test_target_python()


if __name__ == "__main__":
    main()
