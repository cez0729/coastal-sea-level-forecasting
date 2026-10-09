from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "neuralcora_surge"
DEFAULT_INPUT = DATA_ROOT / "processed" / "storm_driven_event_catalog_train_locked_1999_2022.csv"
DEFAULT_OUTPUT = DATA_ROOT / "processed" / "publication_event_catalog_1999_2022.csv"

NON_HURDAT_PER_YEAR = {
    "train": 4,
    "validation": 6,
    "test": 6,
}


def select_events(catalog: pd.DataFrame) -> pd.DataFrame:
    catalog = catalog.copy()
    catalog["core_start"] = pd.to_datetime(catalog["core_start"], utc=True)
    catalog["year"] = catalog["core_start"].dt.year
    catalog["is_hurdat"] = catalog["hurdat_storm_id"].notna()
    selected_parts = [catalog[catalog["is_hurdat"]].copy()]
    non_hurdat = catalog[~catalog["is_hurdat"]].copy()
    for (split, year), group in non_hurdat.groupby(["split", "year"], sort=True):
        count = NON_HURDAT_PER_YEAR[split]
        selected_parts.append(
            group.sort_values(["peak_residual_m", "core_start"], ascending=[False, True]).head(count).copy()
        )
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.drop_duplicates("event_id").sort_values("core_start").reset_index(drop=True)
    selected["selection_reason"] = selected["is_hurdat"].map(
        {True: "all_hurdat_matched", False: "annual_top_non_hurdat_peak"}
    )
    return selected.drop(columns=["year", "is_hurdat"])


def unique_days(frame: pd.DataFrame) -> int:
    dates: set[object] = set()
    for row in frame.itertuples(index=False):
        start = pd.Timestamp(row.window_start).floor("D")
        end = pd.Timestamp(row.window_end).ceil("D")
        dates.update(pd.date_range(start, end, freq="D").date)
    return len(dates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the NeuralCORA-Surge publication event subset")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    catalog = pd.read_csv(args.input)
    selected = select_events(catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_catalog": str(args.input),
        "selection_locked_before_model_comparison": True,
        "rules": {
            "hurdat": "include every HURDAT2-matched episode",
            "non_hurdat": "within each split and calendar year, retain the largest peak residual episodes",
            "non_hurdat_per_year": NON_HURDAT_PER_YEAR,
        },
        "events": int(len(selected)),
        "events_by_split": selected.groupby("split").size().astype(int).to_dict(),
        "hurdat_by_split": selected[selected["hurdat_storm_id"].notna()].groupby("split").size().astype(int).to_dict(),
        "unique_download_days": unique_days(selected),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
