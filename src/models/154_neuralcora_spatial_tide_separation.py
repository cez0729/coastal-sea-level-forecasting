from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOTAL_CACHE = ROOT / "data" / "neuralcora_surge" / "processed" / "publication_field_grid_32x48.npz"
DEFAULT_NOAA = ROOT / "data" / "neuralcora_surge" / "processed" / "noaa_station_water_tide_residual_1999_2022.csv.gz"
DEFAULT_STATIONS = ROOT / "results" / "neuralcora_surge_station_audit" / "station_wet_node_mapping.csv"
DEFAULT_RESIDUAL_CACHE = ROOT / "data" / "neuralcora_surge" / "processed" / "publication_residual_field_grid_32x48.npz"
DEFAULT_OUTPUT = ROOT / "results" / "neuralcora_spatial_tide_separation"


@dataclass
class FieldCache:
    source: np.lib.npyio.NpzFile
    event_ids: list[str]
    splits: dict[str, str]
    labels: dict[str, str]
    fields: dict[str, np.ndarray]
    time_arrays: dict[str, np.ndarray]
    sea_mask: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    mapped_lat: np.ndarray
    mapped_lon: np.ndarray

    def field(self, event: str) -> np.ndarray:
        return self.fields[event]

    def times(self, event: str) -> np.ndarray:
        return self.time_arrays[event]


def load_field_cache(path: Path) -> FieldCache:
    source = np.load(path, allow_pickle=False)
    event_ids = source["event_ids"].astype(str).tolist()
    splits = dict(zip(event_ids, source["event_splits"].astype(str).tolist()))
    labels = dict(zip(event_ids, source["event_labels"].astype(str).tolist()))
    # Decompress once; repeated NPZ member reads are prohibitively slow for LOO audits.
    fields = {event: source[f"field__{event}"].astype(np.float32) for event in event_ids}
    time_arrays = {event: source[f"time__{event}"].astype("datetime64[ns]") for event in event_ids}
    return FieldCache(
        source=source,
        event_ids=event_ids,
        splits=splits,
        labels=labels,
        fields=fields,
        time_arrays=time_arrays,
        sea_mask=source["sea_mask"].astype(bool),
        lat=source["lat"].astype(np.float32),
        lon=source["lon"].astype(np.float32),
        mapped_lat=source["mapped_lat"].astype(np.float32),
        mapped_lon=source["mapped_lon"].astype(np.float32),
    )


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2:
        return float("nan")
    true = y_true[valid].astype(np.float64)
    pred = y_pred[valid].astype(np.float64)
    denominator = np.square(true - true.mean()).sum()
    return float(1.0 - np.square(true - pred).sum() / denominator) if denominator > 0 else float("nan")


def metric_row(scope: str, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    true = y_true[valid].astype(np.float64)
    pred = y_pred[valid].astype(np.float64)
    if not len(true):
        return {"scope": scope, "split": split, "n": 0, "rmse_m": np.nan, "mae_m": np.nan, "r2": np.nan, "correlation": np.nan}
    correlation = float(np.corrcoef(true, pred)[0, 1]) if len(true) > 1 else np.nan
    return {
        "scope": scope,
        "split": split,
        "n": int(len(true)),
        "rmse_m": float(np.sqrt(np.mean(np.square(true - pred)))),
        "mae_m": float(np.mean(np.abs(true - pred))),
        "r2": r2_score(true, pred),
        "correlation": correlation,
    }


def load_noaa(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = ["station_id", "station_name", "time", "observed_msl_m", "tide_msl_m", "residual_m"]
    frame = pd.read_csv(path, compression="infer", usecols=columns)
    frame["station_id"] = frame["station_id"].astype(str)
    frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None)
    frame = frame.drop_duplicates(["time", "station_id"])
    tide = frame.pivot(index="time", columns="station_id", values="tide_msl_m").sort_index()
    residual = frame.pivot(index="time", columns="station_id", values="residual_m").sort_index()
    return frame, tide, residual


def build_features(
    times: np.ndarray,
    tide: pd.DataFrame,
    station_ids: list[str],
    lags: list[int],
) -> np.ndarray:
    index = pd.DatetimeIndex(times.astype("datetime64[ns]"))
    parts: list[np.ndarray] = []
    for lag in lags:
        lookup = index + pd.to_timedelta(lag, unit="h")
        parts.append(tide.reindex(lookup)[station_ids].to_numpy(dtype=np.float64))
    return np.concatenate(parts, axis=1)


def concatenate_split(cache: FieldCache, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    times: list[np.ndarray] = []
    fields: list[np.ndarray] = []
    events: list[str] = []
    for event in cache.event_ids:
        if cache.splits[event] != split:
            continue
        event_times = cache.times(event)
        times.append(event_times)
        fields.append(cache.field(event)[:, cache.sea_mask])
        events.extend([event] * len(event_times))
    return np.concatenate(times), np.concatenate(fields), events


def standardize_design(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train_x, axis=0)
    std = np.nanstd(train_x, axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return mean, std


def design_matrix(raw_x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    scaled = (raw_x - mean[None]) / std[None]
    return np.column_stack([np.ones(len(scaled), dtype=np.float64), scaled])


def train_weights(times: np.ndarray, residual: pd.DataFrame) -> tuple[np.ndarray, float]:
    intensity = residual.reindex(pd.DatetimeIndex(times)).abs().median(axis=1, skipna=True).to_numpy(dtype=np.float64)
    finite = np.isfinite(intensity)
    threshold = float(np.nanquantile(intensity[finite], 0.75))
    filled = np.where(finite, intensity, threshold)
    weights = np.minimum(1.0, np.square(threshold / np.maximum(filled, threshold)))
    return np.maximum(weights, 0.05), threshold


def fit_ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    root_w = np.sqrt(weights)[:, None]
    xw = x * root_w
    yw = y * root_w
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(xw.T @ xw + penalty, xw.T @ yw)


def station_grid_mapping(cache: FieldCache, stations: pd.DataFrame) -> pd.DataFrame:
    wet_rows, wet_cols = np.where(cache.sea_mask)
    wet_lat = cache.mapped_lat[cache.sea_mask]
    wet_lon = cache.mapped_lon[cache.sea_mask]
    rows: list[dict[str, object]] = []
    for station in stations.itertuples(index=False):
        lon_scale = np.cos(np.deg2rad(float(station.station_lat)))
        distance = np.square(wet_lat - station.station_lat) + np.square((wet_lon - station.station_lon) * lon_scale)
        position = int(np.argmin(distance))
        rows.append(
            {
                "station_id": str(station.station_id),
                "station_name": station.station_name,
                "station_lat": float(station.station_lat),
                "station_lon": float(station.station_lon),
                "grid_row": int(wet_rows[position]),
                "grid_col": int(wet_cols[position]),
                "mapped_lat": float(wet_lat[position]),
                "mapped_lon": float(wet_lon[position]),
                "wet_position": position,
            }
        )
    return pd.DataFrame(rows)


def station_matches(
    cache: FieldCache,
    tide: pd.DataFrame,
    noaa: pd.DataFrame,
    mapping: pd.DataFrame,
    coefficients: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    station_ids: list[str],
    lags: list[int],
    splits: tuple[str, ...] = ("train", "validation", "test"),
    estimate_column: str = "estimated_tide_m",
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    by_station = mapping.set_index("station_id")
    for split in splits:
        times, fields, events = concatenate_split(cache, split)
        raw_x = build_features(times, tide, station_ids, lags)
        valid_x = np.isfinite(raw_x).all(axis=1)
        predictions = np.full((len(times), coefficients.shape[1]), np.nan, dtype=np.float64)
        predictions[valid_x] = design_matrix(raw_x[valid_x], feature_mean, feature_std) @ coefficients
        for station_id in station_ids:
            station = by_station.loc[station_id]
            position = int(station.wet_position)
            parts.append(
                pd.DataFrame(
                    {
                        "station_id": station_id,
                        "station_name": station.station_name,
                        "time": pd.DatetimeIndex(times),
                        "split": split,
                        "event_id": events,
                        "cora_total_m": fields[:, position],
                        estimate_column: predictions[:, position],
                    }
                )
            )
    matched = pd.concat(parts, ignore_index=True)
    matched = matched.merge(noaa, on=["station_id", "station_name", "time"], how="left")
    return matched


def validation_score(matched: pd.DataFrame) -> float:
    subset = matched[matched["split"] == "validation"]
    valid = subset["estimated_tide_m"].notna() & subset["tide_msl_m"].notna()
    return float(np.sqrt(np.mean(np.square(subset.loc[valid, "estimated_tide_m"] - subset.loc[valid, "tide_msl_m"]))))


def add_affine_residual_calibration(matched: pd.DataFrame, source: str, output: str) -> pd.DataFrame:
    matched = matched.copy()
    matched[output] = np.nan
    for station_id, group in matched.groupby("station_id"):
        train = group[group["split"] == "train"]
        valid = train[source].notna() & train["residual_m"].notna()
        x = train.loc[valid, source].to_numpy(dtype=np.float64)
        y = train.loc[valid, "residual_m"].to_numpy(dtype=np.float64)
        design = np.column_stack([np.ones(len(x)), x])
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        select = matched["station_id"] == station_id
        matched.loc[select, output] = intercept + slope * matched.loc[select, source]
    return matched


def build_station_metrics(matched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        current = matched[matched["split"] == split]
        rows.append(metric_row("total_vs_observed", split, current["observed_msl_m"].to_numpy(), current["cora_total_m"].to_numpy()))
        rows.append(metric_row("total_vs_residual_negative_control", split, current["residual_m"].to_numpy(), current["cora_total_m"].to_numpy()))
        rows.append(metric_row("estimated_tide_vs_noaa_tide", split, current["tide_msl_m"].to_numpy(), current["estimated_tide_m"].to_numpy()))
        rows.append(metric_row("detided_residual_raw", split, current["residual_m"].to_numpy(), current["cora_residual_m"].to_numpy()))
        rows.append(metric_row("detided_residual_affine", split, current["residual_m"].to_numpy(), current["cora_residual_affine_m"].to_numpy()))
        if "loo_cora_residual_m" in current:
            rows.append(metric_row("detided_residual_leave_one_station_out", split, current["residual_m"].to_numpy(), current["loo_cora_residual_m"].to_numpy()))
    return pd.DataFrame(rows)


def build_residual_cache(
    path: Path,
    cache: FieldCache,
    tide: pd.DataFrame,
    station_ids: list[str],
    lags: list[int],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    coefficients: np.ndarray,
    alpha: float,
) -> tuple[list[str], dict[str, np.ndarray]]:
    valid_events: list[str] = []
    residual_fields: dict[str, np.ndarray] = {}
    for event in cache.event_ids:
        times = cache.times(event)
        raw_x = build_features(times, tide, station_ids, lags)
        if not np.isfinite(raw_x).all():
            continue
        tide_wet = design_matrix(raw_x, feature_mean, feature_std) @ coefficients
        tide_field = np.full(cache.field(event).shape, np.nan, dtype=np.float32)
        tide_field[:, cache.sea_mask] = tide_wet.astype(np.float32)
        residual_fields[event] = (cache.field(event) - tide_field).astype(np.float32)
        valid_events.append(event)

    train_values = np.concatenate(
        [residual_fields[event][:, cache.sea_mask].ravel() for event in valid_events if cache.splits[event] == "train"]
    )
    train_values = train_values[np.isfinite(train_values)]
    payload: dict[str, np.ndarray] = {
        "event_ids": np.asarray(valid_events),
        "event_splits": np.asarray([cache.splits[event] for event in valid_events]),
        "event_labels": np.asarray([cache.labels[event] for event in valid_events]),
        "lat": cache.lat,
        "lon": cache.lon,
        "mapped_lat": cache.mapped_lat,
        "mapped_lon": cache.mapped_lon,
        "sea_mask": cache.sea_mask.astype(np.float32),
        "train_mean": np.asarray(train_values.mean(), dtype=np.float32),
        "train_std": np.asarray(max(train_values.std(), 1e-6), dtype=np.float32),
        "train_q95": np.asarray(np.quantile(train_values, 0.95), dtype=np.float32),
        "train_q99": np.asarray(np.quantile(train_values, 0.99), dtype=np.float32),
        "completeness": np.asarray("complete"),
        "target_definition": np.asarray("CORA zeta minus train-fitted NOAA astronomical-tide basis"),
        "tide_alpha": np.asarray(alpha, dtype=np.float64),
        "source_total_cache": np.asarray(str(DEFAULT_TOTAL_CACHE)),
    }
    for event in valid_events:
        payload[f"field__{event}"] = residual_fields[event]
        payload[f"time__{event}"] = cache.times(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return valid_events, residual_fields


def persistence_metrics(
    fields: dict[str, np.ndarray],
    cache: FieldCache,
    events: list[str],
    train_q95: float,
    input_hours: int = 24,
    lead_hours: int = 24,
) -> dict[str, object]:
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    peak_errors: list[float] = []
    for event in events:
        if cache.splits[event] != "test":
            continue
        field = fields[event]
        count = len(field) - input_hours - lead_hours + 1
        for start in range(max(count, 0)):
            prediction = field[start + input_hours - 1, cache.sea_mask]
            target = field[start + input_hours + lead_hours - 1, cache.sea_mask]
            valid = np.isfinite(prediction) & np.isfinite(target)
            true_parts.append(target[valid])
            pred_parts.append(prediction[valid])
            peak_errors.append(float(abs(np.nanmax(target) - np.nanmax(prediction))))
    true = np.concatenate(true_parts)
    pred = np.concatenate(pred_parts)
    observed_extreme = true >= train_q95
    predicted_extreme = pred >= train_q95
    tp = int(np.sum(observed_extreme & predicted_extreme))
    fp = int(np.sum(~observed_extreme & predicted_extreme))
    fn = int(np.sum(observed_extreme & ~predicted_extreme))
    return {
        "model": "persistence",
        "target": "spatial_non_tidal_residual",
        "input_hours": input_hours,
        "lead_hours": lead_hours,
        "n": int(len(true)),
        "rmse_m": float(np.sqrt(np.mean(np.square(true - pred)))),
        "mae_m": float(np.mean(np.abs(true - pred))),
        "r2": r2_score(true, pred),
        "q95_csi": float(tp / max(tp + fp + fn, 1)),
        "field_peak_mae_m": float(np.mean(peak_errors)),
    }


def make_figures(matched: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    test = matched[matched["split"] == "test"].dropna(
        subset=["tide_msl_m", "estimated_tide_m", "residual_m", "cora_residual_m"]
    )
    sample = test.iloc[:: max(len(test) // 8000, 1)]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hexbin(sample["tide_msl_m"], sample["estimated_tide_m"], gridsize=55, mincnt=1, cmap="viridis")
    limit = float(np.nanmax(np.abs(sample[["tide_msl_m", "estimated_tide_m"]].to_numpy())))
    axes[0, 0].plot([-limit, limit], [-limit, limit], color="black", linewidth=1)
    axes[0, 0].set(title="Test astronomical tide", xlabel="NOAA tide (m)", ylabel="Estimated CORA-grid tide (m)")
    axes[0, 1].hexbin(sample["residual_m"], sample["cora_residual_m"], gridsize=55, mincnt=1, cmap="magma")
    limit = float(np.nanquantile(np.abs(sample[["residual_m", "cora_residual_m"]].to_numpy()), 0.995))
    axes[0, 1].plot([-limit, limit], [-limit, limit], color="black", linewidth=1)
    axes[0, 1].set(title="Test non-tidal residual", xlabel="NOAA observed - tide (m)", ylabel="CORA zeta - estimated tide (m)")
    test_metrics = metrics[metrics["split"] == "test"].set_index("scope")
    scopes = ["total_vs_residual_negative_control", "detided_residual_raw", "detided_residual_affine", "detided_residual_leave_one_station_out"]
    labels = ["Total vs residual\n(negative control)", "De-tided raw", "De-tided +\ntrain affine", "Leave-one-station-out"]
    present = [(scope, label) for scope, label in zip(scopes, labels) if scope in test_metrics.index]
    axes[1, 0].bar([label for _, label in present], [test_metrics.loc[scope, "rmse_m"] for scope, _ in present], color=["#8c8c8c", "#26734d", "#315f8c", "#b05a2b"][: len(present)])
    axes[1, 0].set(title="NOAA station residual RMSE", ylabel="RMSE (m)")
    axes[1, 0].tick_params(axis="x", labelsize=8)
    station_rmse = []
    station_names = []
    for station_name, group in test.groupby("station_name"):
        station_names.append(station_name)
        station_rmse.append(float(np.sqrt(np.mean(np.square(group["residual_m"] - group["cora_residual_m"])))))
    axes[1, 1].barh(station_names, station_rmse, color="#3f7f76")
    axes[1, 1].set(title="Raw de-tided RMSE by station", xlabel="RMSE (m)")
    figure.suptitle("CORA spatial tide-separation audit", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_dir / "spatial_tide_separation_audit.png", dpi=180)
    plt.close(figure)

    peak_row = test.loc[test["residual_m"].abs().idxmax()]
    station = peak_row["station_id"]
    event = peak_row["event_id"]
    example = matched[(matched["station_id"] == station) & (matched["event_id"] == event)].sort_values("time")
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(example["time"], example["residual_m"], label="NOAA residual", color="#111111", linewidth=2)
    axis.plot(example["time"], example["cora_residual_m"], label="CORA de-tided residual", color="#167c80", linewidth=1.5)
    axis.plot(example["time"], example["cora_total_m"], label="CORA total water level", color="#b05a2b", alpha=0.55)
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set(title=f"Largest test residual case: {peak_row['station_name']} ({event})", ylabel="Water level (m)", xlabel="UTC")
    axis.legend(ncol=3, fontsize=8)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_dir / "largest_test_residual_timeseries.png", dpi=180)
    plt.close(figure)


def write_report(
    output_dir: Path,
    best_alpha: float,
    threshold: float,
    metrics: pd.DataFrame,
    persistence: dict[str, object],
    valid_events: list[str],
    excluded_events: list[str],
) -> None:
    test = metrics[metrics["split"] == "test"].set_index("scope")
    negative = test.loc["total_vs_residual_negative_control"]
    raw = test.loc["detided_residual_raw"]
    affine = test.loc["detided_residual_affine"]
    loo = test.loc["detided_residual_leave_one_station_out"]
    report = f"""# CORA 空间潮汐分离与非潮汐残差审计

## 结论

本实验解决了原空间模型把 CORA `zeta` 总水位直接当作 storm-surge field 的定义错误。新的标签为：

`CORA non-tidal residual = CORA zeta - estimated astronomical tide field`。

潮汐场使用 NOAA 七站可提前获得的天文潮预测作为时间基，只用 1999--2016 训练事件拟合空间系数；ridge 强度只由 2017--2019 验证集选择，最终 `alpha={best_alpha:g}`。训练期 NOAA 网络残差 q75 为 `{threshold:.4f} m`，高残差时段在拟合中被连续降权。

## 测试期 NOAA 站点审计

| 对照 | RMSE (m) | R2 | correlation |
|---|---:|---:|---:|
| CORA总水位直接对NOAA residual（负对照） | {negative.rmse_m:.4f} | {negative.r2:.4f} | {negative.correlation:.4f} |
| 去潮后的CORA residual | {raw.rmse_m:.4f} | {raw.r2:.4f} | {raw.correlation:.4f} |
| 去潮后再做训练期仿射校准 | {affine.rmse_m:.4f} | {affine.r2:.4f} | {affine.correlation:.4f} |
| 留一站空间检验 | {loo.rmse_m:.4f} | {loo.r2:.4f} | {loo.correlation:.4f} |

24小时 residual persistence：RMSE `{persistence['rmse_m']:.4f} m`，R2 `{persistence['r2']:.4f}`，训练期 q95 CSI `{persistence['q95_csi']:.4f}`。

新缓存保留 `{len(valid_events)}` 个完整事件组；因潮汐基滞后特征不完整排除 `{len(excluded_events)}` 个事件：`{', '.join(excluded_events) if excluded_events else 'none'}`。

## 科学边界

1. 该结果现在可以严格称为 **storm-event non-tidal residual field**，不能仅凭去潮就称为“solely storm-caused surge”。非潮汐残差仍包含风压、风应力、波浪、河流、季节背景、海流和模型误差。
2. 若要严格得到 counterfactual storm surge，需要 CORA 同网格、同初始/边界条件的 tide-only ADCIRC 对照运行；官方公开桶当前只列出后验 `fort.63/zeta`、最大水位和波浪产品，没有现成 tide-only 场。
3. NOAA 潮汐预测在未来可提前获得，因此作为预测期确定性输入不构成 future-target leakage；空间回归系数和仿射校准均仅由训练期拟合。
4. 七站验证能约束站点附近的去潮质量，但不能完全证明离岸每一个网格单元的潮汐场。留一站结果是对空间外推风险的直接检查。

## 后续模型实验

新 residual 缓存可以直接传给 `118_neuralcora_publication_field_models.py --cache ...`，在完全相同事件切分和预算下重训 Persistence、FNO、Past-ERA5 FNO 和 U-Net。总水位旧模型结果不能直接拿来充当 residual 模型结果。
"""
    (output_dir / "REPORT_CN.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and audit a train-locked CORA non-tidal residual field")
    parser.add_argument("--total-cache", type=Path, default=DEFAULT_TOTAL_CACHE)
    parser.add_argument("--noaa", type=Path, default=DEFAULT_NOAA)
    parser.add_argument("--station-mapping", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--residual-cache", type=Path, default=DEFAULT_RESIDUAL_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lags", type=int, nargs="+", default=[-3, 0, 3])
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[154] loading total-water-level cache", flush=True)
    cache = load_field_cache(args.total_cache)
    print(f"[154] loaded {len(cache.event_ids)} events and {int(cache.sea_mask.sum())} wet cells", flush=True)
    noaa, tide, noaa_residual = load_noaa(args.noaa)
    print(f"[154] loaded NOAA tide table: {len(tide)} hours", flush=True)
    station_metadata = pd.read_csv(args.station_mapping, dtype={"station_id": str})
    station_ids = station_metadata["station_id"].astype(str).tolist()
    mapping = station_grid_mapping(cache, station_metadata)
    mapping.to_csv(args.output_dir / "station_grid_mapping.csv", index=False)

    train_times, train_y, _ = concatenate_split(cache, "train")
    print(f"[154] assembled training matrix: {train_y.shape}", flush=True)
    train_raw_x = build_features(train_times, tide, station_ids, args.lags)
    valid_train = np.isfinite(train_raw_x).all(axis=1) & np.isfinite(train_y).all(axis=1)
    feature_mean, feature_std = standardize_design(train_raw_x[valid_train])
    train_x = design_matrix(train_raw_x[valid_train], feature_mean, feature_std)
    weights, residual_q75 = train_weights(train_times[valid_train], noaa_residual)

    alpha_rows: list[dict[str, float]] = []
    coefficient_candidates: dict[float, np.ndarray] = {}
    for alpha in args.alphas:
        print(f"[154] fitting alpha={alpha:g}", flush=True)
        coefficients = fit_ridge(train_x, train_y[valid_train].astype(np.float64), weights, alpha)
        coefficient_candidates[alpha] = coefficients
        matched = station_matches(cache, tide, noaa, mapping, coefficients, feature_mean, feature_std, station_ids, args.lags, ("validation",))
        score = validation_score(matched)
        alpha_rows.append({"alpha": alpha, "validation_tide_rmse_m": score})
    alpha_frame = pd.DataFrame(alpha_rows).sort_values("validation_tide_rmse_m")
    alpha_frame.to_csv(args.output_dir / "alpha_selection.csv", index=False)
    best_alpha = float(alpha_frame.iloc[0].alpha)
    coefficients = coefficient_candidates[best_alpha]
    print(f"[154] selected alpha={best_alpha:g}", flush=True)

    matched = station_matches(cache, tide, noaa, mapping, coefficients, feature_mean, feature_std, station_ids, args.lags)
    matched["cora_residual_m"] = matched["cora_total_m"] - matched["estimated_tide_m"]

    loo_parts: list[pd.DataFrame] = []
    for omitted in station_ids:
        print(f"[154] leave-one-station-out: omit {omitted}", flush=True)
        included = [station for station in station_ids if station != omitted]
        loo_train_raw = build_features(train_times, tide, included, args.lags)
        loo_valid = np.isfinite(loo_train_raw).all(axis=1) & np.isfinite(train_y).all(axis=1)
        loo_mean, loo_std = standardize_design(loo_train_raw[loo_valid])
        loo_x = design_matrix(loo_train_raw[loo_valid], loo_mean, loo_std)
        loo_weights, _ = train_weights(train_times[loo_valid], noaa_residual)
        loo_coefficients = fit_ridge(loo_x, train_y[loo_valid].astype(np.float64), loo_weights, best_alpha)
        omitted_position = int(mapping.loc[mapping["station_id"] == omitted, "wet_position"].iloc[0])
        omitted_rows: list[pd.DataFrame] = []
        for split in ("train", "validation", "test"):
            current_times, current_fields, current_events = concatenate_split(cache, split)
            current_raw = build_features(current_times, tide, included, args.lags)
            current_valid = np.isfinite(current_raw).all(axis=1)
            current_prediction = np.full(len(current_times), np.nan, dtype=np.float64)
            current_prediction[current_valid] = (
                design_matrix(current_raw[current_valid], loo_mean, loo_std) @ loo_coefficients[:, omitted_position]
            )
            station_name = str(mapping.loc[mapping["station_id"] == omitted, "station_name"].iloc[0])
            omitted_rows.append(
                pd.DataFrame(
                    {
                        "station_id": omitted,
                        "station_name": station_name,
                        "time": pd.DatetimeIndex(current_times),
                        "split": split,
                        "event_id": current_events,
                        "cora_total_m": current_fields[:, omitted_position],
                        "estimated_tide_m": current_prediction,
                    }
                )
            )
        omitted_match = pd.concat(omitted_rows, ignore_index=True).merge(
            noaa, on=["station_id", "station_name", "time"], how="left"
        )
        omitted_match["loo_cora_residual_m"] = omitted_match["cora_total_m"] - omitted_match["estimated_tide_m"]
        loo_parts.append(omitted_match[["station_id", "time", "split", "event_id", "loo_cora_residual_m"]])
    loo = pd.concat(loo_parts, ignore_index=True)
    matched = matched.merge(loo, on=["station_id", "time", "split", "event_id"], how="left")
    matched = add_affine_residual_calibration(matched, "cora_residual_m", "cora_residual_affine_m")
    matched.to_csv(args.output_dir / "station_hourly_matched.csv.gz", index=False, compression="gzip")
    metrics = build_station_metrics(matched)
    metrics.to_csv(args.output_dir / "station_metrics.csv", index=False)

    valid_events, residual_fields = build_residual_cache(
        args.residual_cache, cache, tide, station_ids, args.lags, feature_mean, feature_std, coefficients, best_alpha
    )
    print(f"[154] residual cache events: {len(valid_events)}", flush=True)
    excluded_events = [event for event in cache.event_ids if event not in valid_events]
    residual_source = np.load(args.residual_cache, allow_pickle=False)
    persistence = persistence_metrics(
        residual_fields, cache, valid_events, float(residual_source["train_q95"]), input_hours=24, lead_hours=24
    )
    pd.DataFrame([persistence]).to_csv(args.output_dir / "residual_persistence_24h.csv", index=False)

    np.savez_compressed(
        args.output_dir / "spatial_tide_model.npz",
        coefficients=coefficients.astype(np.float32),
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        station_ids=np.asarray(station_ids),
        lags=np.asarray(args.lags, dtype=np.int16),
        alpha=np.asarray(best_alpha),
        sea_mask=cache.sea_mask.astype(np.uint8),
    )
    make_figures(matched, metrics, args.output_dir)
    write_report(args.output_dir, best_alpha, residual_q75, metrics, persistence, valid_events, excluded_events)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "train-locked robust ridge spatial mapping from lagged NOAA astronomical tide predictions",
        "target": "storm-event non-tidal residual field, not strict storm-only surge",
        "total_cache": str(args.total_cache.resolve()),
        "residual_cache": str(args.residual_cache.resolve()),
        "noaa": str(args.noaa.resolve()),
        "station_ids": station_ids,
        "lags_hours": args.lags,
        "alpha_candidates": args.alphas,
        "selected_alpha": best_alpha,
        "selection_split": "validation (2017-2019)",
        "fit_split": "train (1999-2016)",
        "test_split": "test (2020-2022)",
        "valid_events": len(valid_events),
        "excluded_events": excluded_events,
        "evidence_status": "formal preprocessing audit; downstream residual deep models not yet trained",
    }
    (args.output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(pd.DataFrame([persistence]).to_string(index=False))


if __name__ == "__main__":
    main()
