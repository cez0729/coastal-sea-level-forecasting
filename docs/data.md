# Data requirements

Full scientific training needs these files under `data/processed_multiyear_2023_2025/`:

- `water_tide_residual_long.csv`
- `era5_station_hourly.csv`
- `surface_currents_station_daily.csv`
- `wave_direction_speed_station_3hourly.csv`
- `noaa_coops_met_station_hourly.csv`

The repository includes station metadata at `data/processed/stations.csv` and GEBCO depth metadata at `data/processed/gebco_station_depth.csv`. Run `python scripts/check_data.py` before training. Missing inputs are not silently replaced with synthetic data.

## Sources

| Input | Source |
|---|---|
| Observed level, tide, local weather | https://tidesandcurrents.noaa.gov/api/ |
| Buoy meteorology | https://www.ndbc.noaa.gov/ |
| ERA5 wind and pressure | https://cds.climate.copernicus.eu/ |
| Surface current and wave products | https://data.marine.copernicus.eu/ |
| Bathymetry | https://www.gebco.net/ |

Acquisition and station-feature scripts are grouped in `src/data/`. ERA5 and Copernicus require user accounts and source-product access. Install `requirements-data.txt` only if running those scripts. Local observation gaps and lower-frequency ocean products require the same alignment rules as the original experiment; fresh downloads alone do not guarantee bitwise reproduction.

The original download provenance is included separately in `data_sources_original.md`. Its historical script names can be traced through `file_mapping.csv`. The repository does not redistribute the full forcing products or provide a checkpoint download service.
