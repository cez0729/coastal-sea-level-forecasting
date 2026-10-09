"""Structure audit for the saved CF-MOR-GWN P0 OOF residuals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DEFAULT = ROOT / "results" / "cf_mor_gwn_p0_residual_diagnostic_20260811" / "oof_diagnostic_predictions.npz"
OUTPUT_DEFAULT = ROOT / "results" / "cf_mor_gwn_p0_residual_diagnostic_20260811"
STATIONS = ["New London", "Montauk", "Kings Point", "The Battery", "Sandy Hook", "Atlantic City", "Cape May"]


def autocorrelation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return np.nan
    left, right = values[:-lag], values[lag:]
    if left.std() < 1e-12 or right.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_DEFAULT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    saved = np.load(args.input)
    error = saved["oof_error"]
    rows = []
    for station in range(error.shape[1]):
        for lead in range(error.shape[2]):
            values = error[:, station, lead]
            blocks = np.array_split(values, args.folds)
            block_means = np.asarray([block.mean() for block in blocks])
            rows.append({
                "station": STATIONS[station],
                "lead": lead + 1,
                "mean_error": float(values.mean()),
                "std_error": float(values.std()),
                "rmse": float(np.sqrt(np.mean(values ** 2))),
                "acf_8h": autocorrelation(values, 1),
                "acf_24h": autocorrelation(values, 3),
                "block_mean_std": float(block_means.std()),
                "block_sign_consistency": float(max(np.mean(block_means >= 0), np.mean(block_means <= 0))),
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(out / "oof_residual_station_horizon_structure.csv", index=False)
    per_horizon = detail.groupby("lead").agg(
        mean_abs_bias=("mean_error", lambda x: float(np.mean(np.abs(x)))),
        mean_std=("std_error", "mean"),
        mean_acf_8h=("acf_8h", "mean"),
        mean_acf_24h=("acf_24h", "mean"),
        mean_block_sign_consistency=("block_sign_consistency", "mean"),
    ).reset_index()
    per_station = detail.groupby("station").agg(
        mean_abs_bias=("mean_error", lambda x: float(np.mean(np.abs(x)))),
        mean_std=("std_error", "mean"),
        mean_acf_8h=("acf_8h", "mean"),
        mean_acf_24h=("acf_24h", "mean"),
        mean_block_sign_consistency=("block_sign_consistency", "mean"),
    ).reset_index()
    per_horizon.to_csv(out / "oof_residual_per_horizon.csv", index=False)
    per_station.to_csv(out / "oof_residual_per_station.csv", index=False)
    summary = {
        "mean_abs_station_horizon_bias": float(detail["mean_error"].abs().mean()),
        "mean_station_horizon_std": float(detail["std_error"].mean()),
        "mean_acf_8h": float(detail["acf_8h"].mean()),
        "mean_acf_24h": float(detail["acf_24h"].mean()),
        "fraction_abs_acf_8h_above_0_2": float(np.mean(detail["acf_8h"].abs() > 0.2)),
        "fraction_abs_acf_24h_above_0_2": float(np.mean(detail["acf_24h"].abs() > 0.2)),
        "fraction_block_sign_consistency_1": float(np.mean(detail["block_sign_consistency"] == 1.0)),
    }
    (out / "P0_STRUCTURE_AUDIT.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print("\nPer station")
    print(per_station.to_string(index=False))
    print("\nPer horizon")
    print(per_horizon.to_string(index=False))


if __name__ == "__main__":
    main()
