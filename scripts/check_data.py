"""Fail before training if any required processed input is unavailable."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    'data/processed_multiyear_2023_2025/water_tide_residual_long.csv',
    'data/processed_multiyear_2023_2025/era5_station_hourly.csv',
    'data/processed_multiyear_2023_2025/surface_currents_station_daily.csv',
    'data/processed_multiyear_2023_2025/wave_direction_speed_station_3hourly.csv',
    'data/processed_multiyear_2023_2025/noaa_coops_met_station_hourly.csv',
    'data/processed/gebco_station_depth.csv',
]

if __name__ == '__main__':
    missing = [p for p in REQUIRED if not (ROOT / p).is_file()]
    for p in REQUIRED:
        print(('MISSING ' if p in missing else 'OK      ') + p)
    if missing:
        raise SystemExit('Training inputs incomplete. See docs/data.md. Synthetic tests do not require these files.')
    print('All required files present; this checks availability, not scientific validity.')
