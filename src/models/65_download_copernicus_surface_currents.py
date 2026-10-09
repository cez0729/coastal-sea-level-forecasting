import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "copernicus_currents"
OUT_FILE = "surface_currents_uo_vo_20250401_20251230.nc"
COPERNICUSMARINE_EXE = ROOT / ".venv" / "Scripts" / "copernicusmarine.exe"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(COPERNICUSMARINE_EXE),
        "subset",
        "-i",
        "cmems_mod_glo_phy_my_0.083deg_P1D-m",
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
        "2025-04-01",
        "-T",
        "2025-12-30",
        "-o",
        str(OUT_DIR),
        "-f",
        OUT_FILE,
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
    print(f"Saved to {OUT_DIR / OUT_FILE}")


if __name__ == "__main__":
    main()
