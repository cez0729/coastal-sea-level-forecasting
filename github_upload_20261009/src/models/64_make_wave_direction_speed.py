from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WAVE_PATH = ROOT / "data" / "processed" / "wave_station_hourly_valid.csv"
DEPTH_PATH = ROOT / "data" / "processed" / "gebco_station_depth.csv"
OUT_PATH = ROOT / "data" / "processed" / "wave_direction_speed_station_hourly.csv"

GRAVITY = 9.80665


def solve_wavenumber(period_seconds: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    """Solve omega^2 = g k tanh(k h) with Newton iteration."""
    period_seconds = np.asarray(period_seconds, dtype=np.float64)
    depth_m = np.maximum(np.asarray(depth_m, dtype=np.float64), 0.1)
    omega = 2.0 * np.pi / np.maximum(period_seconds, 0.1)

    k = np.maximum((omega ** 2) / GRAVITY, 1e-6)
    for _ in range(30):
        kh = k * depth_m
        tanh_kh = np.tanh(kh)
        f = GRAVITY * k * tanh_kh - omega ** 2
        df = GRAVITY * (tanh_kh + kh * (1.0 - tanh_kh ** 2))
        k_next = k - f / np.maximum(df, 1e-12)
        k = np.maximum(k_next, 1e-8)
    return k


def phase_speed(period_seconds: pd.Series, depth_m: pd.Series) -> np.ndarray:
    period = period_seconds.to_numpy(dtype=np.float64)
    depth = depth_m.to_numpy(dtype=np.float64)
    omega = 2.0 * np.pi / np.maximum(period, 0.1)
    k = solve_wavenumber(period, depth)
    return omega / k


def main() -> None:
    wave = pd.read_csv(WAVE_PATH)
    depth = pd.read_csv(DEPTH_PATH)[["station_id", "depth"]]

    out = wave.merge(depth, on="station_id", how="left")
    out["wave_speed_from_peak_period_mps"] = phase_speed(
        out["wave_peak_period"], out["depth"]
    )
    out["wave_speed_from_mean_period_mps"] = phase_speed(
        out["wave_mean_period"], out["depth"]
    )

    # Components are useful for ML models and avoid circular direction discontinuity.
    direction_rad = np.deg2rad(out["wave_direction"].to_numpy(dtype=np.float64))
    out["wave_dir_x_from_mps"] = out["wave_speed_from_peak_period_mps"] * np.sin(direction_rad)
    out["wave_dir_y_from_mps"] = out["wave_speed_from_peak_period_mps"] * np.cos(direction_rad)

    columns = [
        "datetime",
        "station_id",
        "station_name",
        "station_lat",
        "station_lon",
        "wave_lat",
        "wave_lon",
        "distance_km",
        "depth",
        "wave_height",
        "wave_peak_period",
        "wave_mean_period",
        "wave_direction",
        "wave_speed_from_peak_period_mps",
        "wave_speed_from_mean_period_mps",
        "wave_dir_x_from_mps",
        "wave_dir_y_from_mps",
    ]
    out[columns].to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
