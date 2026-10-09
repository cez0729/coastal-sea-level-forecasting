# Post-freeze technical repair log

## Repair 1: station identifier dtype in wave/depth merge

- Trigger: preprocessing stopped before model training because Pandas inferred `station_id` as `int64` in the generated depth CSV while the official station metadata retained it as `str`.
- Error: `ValueError: You are trying to merge on str and int64 columns for key 'station_id'`.
- Change: `数据整理/71_make_copernicus_multiyear_station_features.py` now reads the depth station identifier with `dtype={"station_id": str}` and asserts string conversion before the merge.
- Scientific effect: none. No stations, target values, features, time splits, models, hyperparameters, metrics, or acceptance criteria changed. No model was trained and no target metric was viewed before this repair.
- Frozen downloader hash before repair: `10929C1FB5F8F63B85A02C03CFB0F2547E2E76D810CCEF62B99E5327644184D9` (unchanged; repair is in shared helper 71).

## Repair 2: parallelize identical local-meteorology requests

- Trigger: the sequential implementation downloaded only about 20 of 1,440 fixed monthly requests in 25 seconds.
- Change: the same prospectively locked station/product/month jobs now use four download workers and reuse valid cached files.
- Scientific effect: none. URLs, products, station set, periods, parsing, causal filling, feature schema, models, and metrics are unchanged. No model score was available before this repair.

## Repair 3: bound retries for unavailable local products

- Trigger: some PORTS station/product combinations returned no response and caused repeated 90-second waits; NOAA does not accept a one-year request for this product, so monthly requests remain required.
- Change: local-meteorology requests use at most two 30-second attempts and six workers. Every failed station/product/month remains in the inventory as an explicit error and its feature stays missing.
- Scientific effect: no target, retained station, model, split, metric, or imputation rule changed. Local meteorology was prospectively declared as a potentially missing input and is causally filled only when an observation exists. No model score was available before this repair.

## Repair 4: bounded download budget and explicit cached-only completion

- Trigger: after 976 of 1,440 monthly local-product files were saved, groups of unavailable requests repeatedly exhausted the bounded timeout.
- Change: the final preparation pass can consume the 976 cached files and records every absent month as `missing_after_bounded_download_attempt` instead of silently retrying indefinitely.
- Scientific effect: the five local meteorology variable classes remain in the 34-class schema, with unavailable observations represented as missing and handled by the frozen causal missingness rule. Water level, tide, ERA5, currents, waves and bathymetry are complete and unchanged. No model score was available before this repair.

## Repair 5: progress-only runner instrumentation

- Trigger: the first VARX invocation produced no buffered output and was manually stopped before any metric file was created; process inspection could not distinguish CSV construction from a stalled fit.
- Change: progress messages now mark feature construction, missing-window filtering, matrix dimensions and each frozen alpha candidate.
- Scientific effect: none. The original frozen runner hash was `C1388F7321C1453162372C17F53D47C18B02905417D243838D81249431952E7B`; only output instrumentation was added. No model score was available before this repair.

## Repair 6: reuse immutable preprocessing across neural seeds

- Trigger: the complete strict feature build took about 13 minutes; rebuilding the same arrays for every seed would add roughly one hour without changing any value.
- Change: Eta and Multistate models for all seeds receive the same in-memory, read-only preprocessed arrays and fixed dataset indices already used by VARX.
- Scientific effect: none. Random initialization and data-loader shuffling remain seed-specific; scalers, graph priors, samples, hyperparameters and test metrics are identical to independent rebuilds.

## Repair 7: parallel seed scheduling with checkpoint resume

- Trigger: CPU-only execution of five prospectively locked seeds is substantially longer than feature preparation and deterministic VARX fitting.
- Change: the final post-repair run schedules the five fixed seeds as five non-overlapping single-seed processes and uses the runner's epoch-level checkpoint resume. The final merger waits for every seed-level metric, prediction, per-lead, and per-station file and for all training processes to exit before reading results.
- Scientific effect: none. The seed list, initialization for each seed, shuffled-loader generator state, model definitions, training/validation/test windows, stopping rule, and metrics are unchanged. Checkpoints include the optimizer, scheduler, best state, bad-epoch count, data-loader generator, and Python/NumPy/PyTorch random states.

## Download-inventory interpretation

- `download_failures.json` is a first-pass retry diagnostic and is not the final availability ledger.
- The authoritative final local-meteorology status is `data/external_region_delaware_bay_2023_2025/processed/coops_met_download_inventory.csv`: 700 monthly station-product requests returned observations, 276 returned an explicit NOAA `no_data` response, and 464 remained absent after the bounded download budget.
- The authoritative retained-station list and hashes of all processed model inputs are in `data_manifest.json`. Target coverage is separately reported in `target_coverage_by_split.csv`; all ten retained stations have 100% water-level, tide, and residual coverage in every split.

## Analysis implementation clarification

- The frozen protocol referred to an "early lead group" without assigning its numerical boundary. Before any external-region neural test metric or prediction file existed, analysis script 188 fixed that group as Lead 1--8 and compares its mean HS-DT-minus-VARX lead-wise R2 with the Lead-24 difference.
- The hierarchy criterion passes when those two differences have opposite signs or when the external sequence ordering is HS-DT above VARX, which differs from the original-region 2025 H2 ordering.
- This clarification does not change a model, station, feature, split, fusion weight, or test metric. It is reported because the exact early-lead boundary was not encoded in the original local hash lock.

## Repair 8: prevent a second six-hour carry on derived local meteorology

- Trigger: a code audit found that the external downloader used a six-hour forward-held pressure/wind series to calculate three derived columns, after which the shared causal preprocessor could forward-fill those already derived values for another six hours.
- Measured pre-repair extent: five pressure-anomaly values, five pressure-tendency values, and seven wind-tendency values had source-observation ages above six hours; the maximum was 12 hours. No external neural metric or prediction file existed when this was found. The deterministic VARX score had been generated, so the repair is explicitly code-rule motivated rather than score blind.
- Change: derived local-meteorology values are now emitted only at timestamps with an actual source observation. The shared causal preprocessor then performs the single allowed six-hour forward hold. Raw pressure and wind remain unchanged.
- Scientific effect: the station set, target, 34-class feature names, split, models, seeds, hyperparameters, fusion rule, metrics, and acceptance criteria are unchanged. All pre-repair VARX output and incomplete neural checkpoints are quarantined and excluded; every formal model is rerun from the regenerated data manifest.
- Post-repair hashes: downloader `FC445F3B42869A437B49DE8DE64436F78D6AC7850DCB86F0E5EB08C0542E9B34`; runner `F4859F7A8999B48F48A4C0E52BD55052FDBAA21BB400D788966BDA750EAAB9D9`; analysis script `C9F93300ACE097E6D037E3CF78D6C748E82D5F6171A68440D501DD25B9667DD2`; processed-data manifest `FF5C5367FF592A2FFDFAF3B28F6724C29D58A91F7048ABEC607A26A1ADA0F056`.

## Repair 9: sufficient-statistic block bootstrap implementation

- Trigger: the original analysis implementation copied the complete station-by-lead target and prediction tensors for every one of 2,000 bootstrap draws and did not finish after 15 minutes.
- Change: squared errors, target sums, and target squared sums are first aggregated by forecast origin. The same moving-block indices and seed resampling are then applied to these sufficient statistics.
- Scientific effect: none. The point effect, block length, number of replicates, random seeds, sampled forecast origins, sampled model seeds, R2 denominator, intervals, and decision criteria are unchanged. This is an algebraically equivalent implementation of the same bootstrap statistic.
