"""Apply a validation-locked horizon safety mask to a residual GWN.

The residual correction is retained only at forecast leads where it lowers
validation MSE relative to the frozen horizon anchor. Test targets are used
only after the binary lead mask has been locked.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
INPUT_DEFAULT = ROOT / "results" / "horizon_protected_residual_gwn_screen_20260811" / "predictions.npz"
OUTPUT_DEFAULT = ROOT / "results" / "validation_safe_horizon_mask_20260811"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


p174 = load_module("p174_validation_safe", HERE / "174_horizon_residual_boosted_gwn_screen.py")


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    denominator = np.sum((true - true.mean()) ** 2)
    return float(1.0 - np.sum((true - pred) ** 2) / max(float(denominator), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_DEFAULT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DEFAULT))
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = np.load(args.input)

    val_true = saved["val_true"]
    val_anchor = saved["val_anchor"]
    val_boosted = saved["val_boosted"]
    val_anchor_mse = np.mean((val_anchor - val_true) ** 2, axis=(0, 1))
    val_boosted_mse = np.mean((val_boosted - val_true) ** 2, axis=(0, 1))
    lead_mask = val_boosted_mse < val_anchor_mse

    rows = []
    outputs = {"lead_mask": lead_mask.astype(np.int8)}
    for split in ("val", "test"):
        true = saved[f"{split}_true"]
        tide = saved[f"{split}_tide"]
        anchor = saved[f"{split}_anchor"]
        boosted = saved[f"{split}_boosted"]
        safe = np.where(lead_mask[None, None, :], boosted, anchor)
        outputs[f"{split}_true"] = true
        outputs[f"{split}_tide"] = tide
        outputs[f"{split}_anchor"] = anchor
        outputs[f"{split}_boosted"] = boosted
        outputs[f"{split}_validation_safe"] = safe
        for model_name, pred in (
            ("horizon_anchor", anchor),
            ("residual_boosted_adaptive_gwn", boosted),
            ("validation_safe_horizon_mask_gwn", safe),
        ):
            rows.append({"split": split, "model": model_name, **p174.p170.summarize(true, pred, tide)})

    per_lead = pd.DataFrame({
        "lead": np.arange(1, len(lead_mask) + 1),
        "selected_by_validation": lead_mask,
        "validation_anchor_mse": val_anchor_mse,
        "validation_boosted_mse": val_boosted_mse,
        "validation_mse_reduction": val_anchor_mse - val_boosted_mse,
        "test_anchor_mse": np.mean((saved["test_anchor"] - saved["test_true"]) ** 2, axis=(0, 1)),
        "test_boosted_mse": np.mean((saved["test_boosted"] - saved["test_true"]) ** 2, axis=(0, 1)),
    })
    per_lead["test_mse_reduction"] = per_lead["test_anchor_mse"] - per_lead["test_boosted_mse"]
    per_lead.to_csv(out / "per_lead_safety_mask.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "evaluation_metrics.csv", index=False)
    np.savez_compressed(out / "predictions.npz", **outputs)

    test_safe = outputs["test_validation_safe"]
    test_anchor = outputs["test_anchor"]
    audit = {
        "input": str(Path(args.input).resolve()),
        "selected_leads_1based": (np.flatnonzero(lead_mask) + 1).tolist(),
        "selected_count": int(lead_mask.sum()),
        "terminal_selected": bool(lead_mask[-1]),
        "max_abs_test_terminal_difference_from_anchor": float(
            np.max(np.abs(test_safe[..., -1] - test_anchor[..., -1]))
        ),
        "validation_safe_sequence_r2_recomputed": r2(outputs["val_true"], outputs["val_validation_safe"]),
        "test_safe_sequence_r2_recomputed": r2(outputs["test_true"], test_safe),
    }
    (out / "selection_audit.json").write_text(json.dumps(audit, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
