from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "autodl_bundle" / "sea_level"
ARCHIVE = ROOT / "autodl_sea_level_bundle.zip"

SCRIPTS = [
    "74_multistate_physics_loss_gnn_bigru_v2.py",
    "76_enhanced_forcing_physics_loss_ablation_v3.py",
    "77_v4_physics_loss_weight_strategy_search.py",
    "78_final_four_models_enhanced_data.py",
    "80_priority_top3_convincing_experiments.py",
    "84_causal_preprocessing_robustness.py",
    "85_simple_baselines_for_paper.py",
    "94_event_based_extreme_evaluation.py",
    "95_block_bootstrap_significance.py",
    "96_rolling_origin_validation.py",
    "100_merge_autodl_results.py",
    "101_priority1_publication_experiments.py",
    "102_priority2_publication_experiments.py",
]

DATA_FILES = [
    "processed_multiyear_2023_2025/water_tide_residual_long.csv",
    "processed_multiyear_2023_2025/era5_station_hourly.csv",
    "processed_multiyear_2023_2025/surface_currents_station_daily.csv",
    "processed_multiyear_2023_2025/wave_direction_speed_station_3hourly.csv",
    "processed_multiyear_2023_2025/noaa_coops_met_station_hourly.csv",
    "processed/gebco_station_depth.csv",
]


def copy_relative(source_root: Path, relative: str, destination_root: Path) -> None:
    source = source_root / relative
    if not source.exists():
        raise FileNotFoundError(source)
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    if STAGING.parent.exists():
        shutil.rmtree(STAGING.parent)
    (STAGING / "scripts").mkdir(parents=True)
    for name in SCRIPTS:
        copy_relative(ROOT / "数据整理", name, STAGING / "scripts")
    for relative in DATA_FILES:
        copy_relative(ROOT / "data", relative, STAGING / "data")
    for path in (ROOT / "autodl").iterdir():
        if path.is_file():
            copy_relative(ROOT, f"autodl/{path.name}", STAGING)

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    shutil.make_archive(str(ARCHIVE.with_suffix("")), "zip", STAGING.parent, STAGING.name)
    size_mb = ARCHIVE.stat().st_size / 1024**2
    print(f"Bundle directory: {STAGING}")
    print(f"Upload archive: {ARCHIVE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
