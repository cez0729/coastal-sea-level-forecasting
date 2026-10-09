from __future__ import annotations

from statistics import NormalDist

import numpy as np
from scipy.special import erf


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]
    if not true.size:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan}
    error = pred - true
    denominator = np.sum((true - true.mean()) ** 2)
    r2 = 1.0 - np.sum(error ** 2) / denominator if denominator > 0 else np.nan
    return {
        "R2": float(r2),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
    }


def point_summary(official, true: np.ndarray, pred: np.ndarray, tide: np.ndarray) -> dict[str, float]:
    return official.final4.summarize_single(true, pred, tide)


def fit_global_sigma_scale(y_val, mu_val, sigma_val, clip=(0.25, 4.0)) -> dict[str, float]:
    y = np.asarray(y_val, dtype=np.float64)
    mu = np.asarray(mu_val, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma_val, dtype=np.float64), 1e-12)
    if y.shape != mu.shape or y.shape != sigma.shape:
        raise ValueError("Validation y, mu and sigma must have identical shapes")
    raw = float(np.sqrt(np.mean(((y - mu) / sigma) ** 2)))
    return {"raw_scale": raw, "scale": float(np.clip(raw, clip[0], clip[1]))}


def gaussian_nll(y, mu, sigma) -> float:
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    return float(np.mean(0.5 * ((y - mu) / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2.0 * np.pi)))


def gaussian_crps(y, mu, sigma) -> float:
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    z = (y - mu) / sigma
    pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def probabilistic_metrics(y, mu, sigma, levels=(0.50, 0.80, 0.95)) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    out = {"gaussian_nll": gaussian_nll(y, mu, sigma), "crps": gaussian_crps(y, mu, sigma)}
    normal = NormalDist()
    for level in levels:
        z = normal.inv_cdf(0.5 + float(level) / 2.0)
        lower = mu - z * sigma
        upper = mu + z * sigma
        label = int(round(level * 100))
        out[f"coverage_{label}"] = float(np.mean((y >= lower) & (y <= upper)))
        out[f"width_{label}"] = float(np.mean(upper - lower))
    out["mean_sigma"] = float(np.mean(sigma))
    return out


def pit_histogram(y, mu, sigma, bins: int = 10) -> list[dict[str, float]]:
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    pit = 0.5 * (1.0 + erf(((y - mu) / sigma) / np.sqrt(2.0)))
    counts, edges = np.histogram(pit, bins=bins, range=(0.0, 1.0))
    total = max(1, int(counts.sum()))
    return [
        {
            "bin": index + 1,
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": int(counts[index]),
            "frequency": float(counts[index] / total),
        }
        for index in range(bins)
    ]


def metrics_by_horizon(y, mu, sigma=None) -> list[dict[str, float]]:
    y = np.asarray(y)
    mu = np.asarray(mu)
    if y.shape != mu.shape or y.ndim != 3:
        raise ValueError("Expected [sample, station, horizon]")
    rows = []
    for lead in range(y.shape[-1]):
        row = {"lead_hour": lead + 1, **regression_metrics(y[..., lead], mu[..., lead])}
        if sigma is not None:
            row.update(probabilistic_metrics(y[..., lead], mu[..., lead], np.asarray(sigma)[..., lead]))
        rows.append(row)
    return rows


def metrics_by_station(y, mu, sigma=None, station_ids=None) -> list[dict[str, float]]:
    y = np.asarray(y)
    mu = np.asarray(mu)
    if y.shape != mu.shape or y.ndim != 3:
        raise ValueError("Expected [sample, station, horizon]")
    station_ids = list(range(y.shape[1])) if station_ids is None else list(station_ids)
    rows = []
    for station, station_id in enumerate(station_ids):
        row = {"station_id": station_id, **regression_metrics(y[:, station], mu[:, station])}
        row.update({f"lead24_{k}": value for k, value in regression_metrics(y[:, station, -1], mu[:, station, -1]).items()})
        if sigma is not None:
            row.update(probabilistic_metrics(y[:, station], mu[:, station], np.asarray(sigma)[:, station]))
        rows.append(row)
    return rows


def _moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]


def paired_block_bootstrap_r2(
    y_true,
    pred_left,
    pred_right,
    *,
    block_length: int = 168,
    replicates: int = 2000,
    seed: int = 20260816,
) -> dict[str, float]:
    """Paired moving-block bootstrap for R2(left)-R2(right) along forecast origins."""
    y = np.asarray(y_true)
    left = np.asarray(pred_left)
    right = np.asarray(pred_right)
    if y.shape != left.shape or y.shape != right.shape or y.ndim < 2:
        raise ValueError("Paired arrays must share shape [origin, ...]")
    n = y.shape[0]
    if n < 2:
        raise ValueError("At least two origins are required")
    point = regression_metrics(y, left)["R2"] - regression_metrics(y, right)["R2"]
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = _moving_block_indices(n, min(block_length, n), rng)
        values[index] = regression_metrics(y[sample], left[sample])["R2"] - regression_metrics(y[sample], right[sample])["R2"]
    return {
        "delta_r2": float(point),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "prob_delta_gt_zero": float(np.mean(values > 0.0)),
        "block_length": int(min(block_length, n)),
        "replicates": int(replicates),
    }
