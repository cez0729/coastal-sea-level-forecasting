# Reproduction levels

1. **No data required:** synthetic invariant tests and saved-result viewer.
2. **Processed data required:** training the two GWN experts and deterministic HS-DT.
3. **Expert checkpoints required:** training the aligned C4 extension.
4. **Prediction caches required:** recomputing ensemble and scale-control intervals.

The release does not claim that level 1 reproduces the empirical results. Use the original preprocessing and checkpoint provenance for levels 2–4.

For level 4, put validation and test NPZ caches under:

```text
data/prediction_caches/
  historical_7/seed_42/{validation_predictions,test_predictions}.npz
  external_10/seed_42/{validation_predictions,test_predictions}.npz
```

Repeat for seeds 123, 2024, 2025, and 3407. Required keys are `true_residual`, `pred_residual`, `eta_pred`, `multi_pred_states`, `sigma_residual`, and `station_ids`. These are C4 aligned caches, not arbitrary same-named model outputs.

Run `python src/evaluation/ensemble_scale_controls.py`. The script refuses absent files rather than fabricating predictions. Bootstrap intervals describe temporal sampling uncertainty conditional on the fixed expert bank.
