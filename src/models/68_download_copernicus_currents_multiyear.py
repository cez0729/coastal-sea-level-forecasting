from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COPERNICUSMARINE_EXE = ROOT / ".venv" / "Scripts" / "copernicusmarine.exe"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Copernicus Marine surface current uo/vo data for the project region."
    )
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--out-dir", default="data/raw/copernicus_currents_2023_2025")
    parser.add_argument("--dataset-id", default="cmems_mod_glo_phy_my_0.083deg_P1D-m")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = f"surface_currents_uo_vo_{args.start_date}_{args.end_date}.nc".replace("-", "")

    cmd = [
        str(COPERNICUSMARINE_EXE),
        "subset",
        "-i",
        args.dataset_id,
        "-v",
        "uo",
        "-v",
        "vo",
        "-x",
        "-76",
        "-X",
        "-70",
        "-y",
        "38",
        "-Y",
        "42.5",
        "-z",
        "0",
        "-Z",
        "1",
        "-t",
        args.start_date,
        "-T",
        args.end_date,
        "-o",
        str(out_dir),
        "-f",
        out_file,
        "--file-format",
        "netcdf",
        "--coordinates-selection-method",
        "nearest",
        "--overwrite",
        "split-on",
        "--on-time",
        "month",
        "--concurrent-processes",
        "1",
    ]

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved monthly files under: {out_dir}")


if __name__ == "__main__":
    main()
