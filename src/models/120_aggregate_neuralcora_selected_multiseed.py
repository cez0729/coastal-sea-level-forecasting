from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCREENING = ROOT / "results" / "neuralcora_surge_model_screening_seed42"
DEFAULT_SELECTED = ROOT / "results" / "neuralcora_surge_selected_multiseed"
SELECTED_MODELS = ["climatology", "persistence", "unet", "regional_unet", "extreme_unet", "fno", "geognn"]


def combine(screening_paths: list[Path], selected_paths: list[Path], keys: list[str], models: list[str]) -> pd.DataFrame:
    screening = pd.concat([pd.read_csv(path) for path in screening_paths], ignore_index=True)
    selected = pd.concat([pd.read_csv(path) for path in selected_paths], ignore_index=True)
    screening = screening[screening["model"].isin(models)].copy()
    selected = selected[selected["model"].isin(models)].copy()
    # Explicit screening reruns should replace an older row with the same key.
    combined = pd.concat([selected, screening], ignore_index=True)
    return combined.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)


def mean_std(summary: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in summary.select_dtypes(include=[np.number]).columns if column not in {"seed", "parameters"}]
    rows: list[dict[str, object]] = []
    for model, group in summary.groupby("model", sort=False):
        row: dict[str, object] = {"model": model, "seeds": int(group["seed"].nunique())}
        for column in numeric:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine screening rows with supplemental NeuralCORA multiseed results")
    parser.add_argument("--screening-dir", type=Path, nargs="+", default=[DEFAULT_SCREENING])
    parser.add_argument("--selected-dir", type=Path, nargs="+", default=[DEFAULT_SELECTED])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models", nargs="+", default=SELECTED_MODELS)
    args = parser.parse_args()

    output_dir = args.output_dir or args.selected_dir[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = combine(
        [path / "seed_summary.csv" for path in args.screening_dir],
        [path / "seed_summary.csv" for path in args.selected_dir],
        ["model", "seed"],
        args.models,
    )
    events = combine(
        [path / "per_event_all_models.csv" for path in args.screening_dir],
        [path / "per_event_all_models.csv" for path in args.selected_dir],
        ["model", "seed", "event_group"],
        args.models,
    )
    noaa = combine(
        [path / "noaa_station_metrics.csv" for path in args.screening_dir],
        [path / "noaa_station_metrics.csv" for path in args.selected_dir],
        ["model", "seed", "station_name"],
        args.models,
    )
    summary.to_csv(output_dir / "seed_summary.csv", index=False)
    events.to_csv(output_dir / "per_event_all_models.csv", index=False)
    noaa.to_csv(output_dir / "noaa_station_metrics.csv", index=False)
    mean_std(summary).to_csv(output_dir / "mean_std.csv", index=False)

    counts = summary.groupby("model")["seed"].nunique()
    expected = {
        model: 1 if model in {"climatology", "persistence"} or model.startswith("era5_future_") else 5
        for model in sorted(args.models)
    }
    actual = counts.to_dict()
    if actual != expected:
        raise RuntimeError(f"Unexpected seed counts after aggregation: {actual}; expected {expected}")
    print(counts.to_string())
    print(f"Saved combined tables in {output_dir}")


if __name__ == "__main__":
    main()
