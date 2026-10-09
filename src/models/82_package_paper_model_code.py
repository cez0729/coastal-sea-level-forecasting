from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "数据整理"
PKG = ROOT / "paper_model_code_package"
ZIP_PATH = ROOT / "paper_model_code_package.zip"


FILES = [
    (
        "67_download_noaa_multiyear_raw.py",
        "00_data_download_NOAA_water_level_and_tide_2023_2025.py",
        "Download NOAA observed water level and tide prediction data for 2023-2025.",
    ),
    (
        "69_make_noaa_multiyear_residual.py",
        "01_data_process_NOAA_water_tide_residual.py",
        "Build water level, tide, and residual sea-level matrices from NOAA data.",
    ),
    (
        "70_download_era5_multiyear.py",
        "02_data_download_ERA5_wind_pressure_2023_2025.py",
        "Download ERA5 atmospheric forcing, including wind and sea-level pressure.",
    ),
    (
        "72_make_era5_multiyear_station_features.py",
        "03_data_process_ERA5_station_hourly_features.py",
        "Interpolate/align ERA5 atmospheric variables to tide-gauge stations.",
    ),
    (
        "68_download_copernicus_currents_multiyear.py",
        "04_data_download_Copernicus_surface_currents_2023_2025.py",
        "Download Copernicus Marine surface current data.",
    ),
    (
        "71_make_copernicus_multiyear_station_features.py",
        "05_data_process_Copernicus_current_wave_station_features.py",
        "Create station-level current and wave features from Copernicus data.",
    ),
    (
        "75_download_noaa_ndbc_enhanced_forcing.py",
        "06_data_download_NOAA_COOPS_enhanced_local_forcing.py",
        "Download and align enhanced local NOAA CO-OPS meteorological forcing.",
    ),
    (
        "73_multistate_graph_shallow_water_baseline.py",
        "10_physics_baseline_multistate_shallow_water_ODE.py",
        "Construct the multistate shallow-water-inspired physical ODE baseline.",
    ),
    (
        "74_multistate_physics_loss_gnn_bigru_v2.py",
        "20_model_PhysicalLoss_GNN_BiGRU_multistate_V2.py",
        "Train the multistate physics-loss GNN-BiGRU model.",
    ),
    (
        "76_enhanced_forcing_physics_loss_ablation_v3.py",
        "21_ablation_physics_states_eta_uv_wave_enhanced_forcing.py",
        "Ablate physics states: eta-only, eta+u+v, eta+u+v+wave.",
    ),
    (
        "77_v4_physics_loss_weight_strategy_search.py",
        "22_ablation_physics_loss_weight_strategy_search.py",
        "Search physics-loss weight, lead weighting, and extreme weighting strategies.",
    ),
    (
        "78_final_four_models_enhanced_data.py",
        "30_final_four_models_GNN_BiGRU_LearnableGraph_ODE_PhysicalLoss.py",
        "Run the four final models under the same enhanced-data setting.",
    ),
    (
        "79_run_final_validation_tasks.py",
        "31_final_validation_multiseed_horizon_extreme_event.py",
        "Run multi-seed stability, horizon comparison, and extreme-event validation.",
    ),
    (
        "80_priority_top3_convincing_experiments.py",
        "32_optimization_top3_lambda_stride_extreme_weight.py",
        "Run the top-three credibility/accuracy experiments: lambda, stride, extreme weighting.",
    ),
]


README = """# Paper Model Code Package

This package contains the model and experiment code reflected in the current paper-style summary.

The files were copied from the original project script folder and renamed so the file names directly show what each script does. The original scripts remain unchanged.

## File Map

| Packaged file | Purpose |
|---|---|
{rows}

## Suggested Reading Order

1. Start with `00` to `06` to understand the data pipeline.
2. Read `10` for the standalone multistate physics baseline.
3. Read `20` to `22` for the physics-loss model and ablations.
4. Read `30` and `31` for the final four-model comparison and validation.
5. Read `32` for the latest optimization experiments: physics-loss weight, denser rolling-window training, and extreme-event weighting.

## Notes

- The code depends on the processed data already present in the project `data/` folder.
- No passwords, API keys, or account credentials are included in this package.
- The packaged scripts are intended for paper organization and review. For exact historical reproduction, the original scripts in the project script folder are still the authoritative source.
"""


def main() -> None:
    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir(parents=True)

    rows = []
    for original, renamed, purpose in FILES:
        src = SRC / original
        if not src.exists():
            raise FileNotFoundError(src)
        dst = PKG / renamed
        shutil.copy2(src, dst)
        rows.append(f"| `{renamed}` | {purpose} |")

    (PKG / "README.md").write_text(README.format(rows="\n".join(rows)), encoding="utf-8")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PKG.rglob("*")):
            zf.write(path, path.relative_to(ROOT))
    print(ZIP_PATH)


if __name__ == "__main__":
    main()
